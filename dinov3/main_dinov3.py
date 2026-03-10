# DINOv3 in-domain self-supervised adaptation (teacher-student) using OFFICIAL dinov3 backbone
# Supports: full finetune OR lightweight finetune (last N blocks + norms + head)

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import inspect

import numpy as np
from PIL import Image, ImageFile, PngImagePlugin

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils

# ---- Official DINOv3 backbone + head ----
from dinov3.hub import backbones
from dinov3.layers.dino_head import DINOHead

# optional: RMSNorm in dinov3
try:
    from dinov3.layers.rms_norm import RMSNorm  # type: ignore
except Exception:
    RMSNorm = None

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # 100MB
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

torchvision_archs = sorted(
    name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)


def make_dino_head(in_dim: int, out_dim: int, use_bn_in_head: bool, norm_last_layer: bool):
    """
    Robust DINOHead builder across dinov3 versions.
    It inspects DINOHead signature and only passes supported kwargs.
    """
    sig = inspect.signature(DINOHead)
    kwargs = {}

    # BN flag
    if "use_bn" in sig.parameters:
        kwargs["use_bn"] = use_bn_in_head
    elif "use_bn_in_head" in sig.parameters:
        kwargs["use_bn_in_head"] = use_bn_in_head

    # last-layer norm flag (some versions don't have it; then ignored)
    if "norm_last_layer" in sig.parameters:
        kwargs["norm_last_layer"] = norm_last_layer
    elif "normalize_last_layer" in sig.parameters:
        kwargs["normalize_last_layer"] = norm_last_layer

    return DINOHead(in_dim=in_dim, out_dim=out_dim, **kwargs)


def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def apply_finetune_policy(student_backbone: nn.Module, student_head: nn.Module, args):
    """
    Full finetune: do nothing (train everything).
    Lightweight finetune:
      - freeze entire backbone
      - unfreeze: DINOHead (always)
      - optionally unfreeze: last N blocks
      - optionally unfreeze: norm layers (LayerNorm/RMSNorm)
      - optionally unfreeze: patch_embed / cls_token / register_tokens
    """
    if not args.lightweight_finetune:
        if utils.is_main_process():
            print("[finetune] FULL finetune: training all backbone params + head.")
        return

    # 1) freeze all backbone
    _set_requires_grad(student_backbone, False)

    # 2) always train head
    _set_requires_grad(student_head, True)

    # 3) unfreeze last N blocks
    n = max(int(args.train_last_n_blocks), 0)
    if n > 0 and hasattr(student_backbone, "blocks"):
        blocks = getattr(student_backbone, "blocks")
        try:
            total_blocks = len(blocks)
        except Exception:
            total_blocks = None
        if total_blocks is not None:
            n = min(n, total_blocks)
        for blk in blocks[-n:]:
            _set_requires_grad(blk, True)

    # 4) unfreeze norm layers
    if args.train_norm_layers:
        norm_types = (nn.LayerNorm,)
        if RMSNorm is not None:
            norm_types = norm_types + (RMSNorm,)

        for m in student_backbone.modules():
            if isinstance(m, norm_types):
                _set_requires_grad(m, True)

    # 5) optional: patch embed
    if args.train_patch_embed and hasattr(student_backbone, "patch_embed"):
        _set_requires_grad(getattr(student_backbone, "patch_embed"), True)

    # 6) optional: tokens (they are Parameters)
    if args.train_cls_token and hasattr(student_backbone, "cls_token"):
        tok = getattr(student_backbone, "cls_token")
        if isinstance(tok, torch.nn.Parameter):
            tok.requires_grad = True

    if args.train_register_tokens and hasattr(student_backbone, "register_tokens"):
        tok = getattr(student_backbone, "register_tokens")
        if isinstance(tok, torch.nn.Parameter):
            tok.requires_grad = True

    if utils.is_main_process():
        bb_params = list(student_backbone.parameters())
        hd_params = list(student_head.parameters())
        total = sum(p.numel() for p in bb_params + hd_params)
        trainable = sum(p.numel() for p in bb_params + hd_params if p.requires_grad)
        pct = 100.0 * trainable / max(total, 1)
        print(f"[finetune] LIGHTWEIGHT finetune enabled.")
        print(f"          trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
        print(f"          last_n_blocks={args.train_last_n_blocks}, train_norm={args.train_norm_layers}, "
              f"train_patch_embed={args.train_patch_embed}, train_cls_token={args.train_cls_token}, "
              f"train_register_tokens={args.train_register_tokens}")


def get_args_parser():
    parser = argparse.ArgumentParser("DINOv3 In-domain Self-supervised Adaptation", add_help=False)

    # ---- backbone ----
    parser.add_argument(
        "--arch", default="dinov3_vits16plus", type=str,
        choices=[
            "dinov3_vits16plus", "dinov3_vits16", "dinov3_vitb16",
            "dinov3_vitl16", "dinov3_vitl16plus", "dinov3_vith16plus", "dinov3_vit7b16",
        ] + torchvision_archs,
        help="Backbone architecture."
    )
    parser.add_argument("--patch_size", default=16, type=int)

    # ---- DINO head / loss ----
    parser.add_argument("--out_dim", default=65536, type=int)
    parser.add_argument("--norm_last_layer", default=True, type=utils.bool_flag)
    parser.add_argument("--use_bn_in_head", default=False, type=utils.bool_flag)

    parser.add_argument("--warmup_teacher_temp", default=0.04, type=float)
    parser.add_argument("--teacher_temp", default=0.04, type=float)
    parser.add_argument("--warmup_teacher_temp_epochs", default=0, type=int)
    parser.add_argument("--momentum_teacher", default=0.996, type=float)

    # ---- optimization ----
    parser.add_argument("--use_fp16", type=utils.bool_flag, default=True)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--weight_decay_end", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=3.0)
    parser.add_argument("--batch_size_per_gpu", default=128, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--freeze_last_layer", default=1, type=int)

    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--optimizer", default="adamw", type=str, choices=["adamw", "sgd", "lars"])
    parser.add_argument("--drop_path_rate", type=float, default=0.1)

    # ---- multi-crop ----
    parser.add_argument("--global_crops_scale", type=float, nargs="+", default=(0.4, 1.0))
    parser.add_argument("--local_crops_number", type=int, default=8)
    parser.add_argument("--local_crops_scale", type=float, nargs="+", default=(0.05, 0.4))

    # ---- misc ----
    parser.add_argument("--data_path", default="/Pathology_data/UCEC/patch_data_whole_slide_20X_256", type=str)
    parser.add_argument(
        "--pretrained_weights",
        default="/root/code/PRISM-main/dino/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        type=str,
        help="Local path to official dinov3 pretrained weights (.pth)."
    )
    parser.add_argument("--output_dir", default="/Pathology_data/UCEC/log_UCEC_dinov3_adapt_official", type=str)
    parser.add_argument("--saveckp_freq", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)

    # ---- NEW: finetune policy switches ----
    parser.add_argument("--lightweight_finetune", default=False, type=utils.bool_flag,
                        help="If true: freeze most backbone, only train head + selected parts.")
    parser.add_argument("--train_last_n_blocks", default=0, type=int,
                        help="In lightweight mode, unfreeze last N transformer blocks.")
    parser.add_argument("--train_norm_layers", default=True, type=utils.bool_flag,
                        help="In lightweight mode, also train LayerNorm/RMSNorm params.")
    parser.add_argument("--train_patch_embed", default=False, type=utils.bool_flag,
                        help="In lightweight mode, also train patch embedding (usually False).")
    parser.add_argument("--train_cls_token", default=False, type=utils.bool_flag,
                        help="In lightweight mode, also train cls_token (usually False).")
    parser.add_argument("--train_register_tokens", default=False, type=utils.bool_flag,
                        help="In lightweight mode, also train register_tokens if present.")

    return parser


def build_backbone(args):
    if not hasattr(backbones, args.arch):
        raise ValueError(f"Unknown dinov3 arch: {args.arch}")

    fn = getattr(backbones, args.arch)
    kwargs = {"drop_path_rate": args.drop_path_rate}

    if args.pretrained_weights and args.pretrained_weights.strip():
        try:
            model = fn(pretrained=True, weights=args.pretrained_weights, check_hash=False, **kwargs)
        except TypeError:
            model = fn(pretrained=True, weights=args.pretrained_weights, check_hash=False)
    else:
        try:
            model = fn(pretrained=True, weights="LVD1689M", check_hash=False, **kwargs)
        except TypeError:
            model = fn(pretrained=True, weights="LVD1689M", check_hash=False)

    embed_dim = getattr(model, "embed_dim", None) or getattr(model, "dim", None)
    if embed_dim is None:
        raise RuntimeError("Cannot infer embed_dim from backbone; inspect model attributes.")
    return model, embed_dim


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)

    if utils.is_main_process():
        print("git:\n  {}\n".format(utils.get_sha()))
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
        print("DINOHead signature:", inspect.signature(DINOHead))

    cudnn.benchmark = True

    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if utils.is_main_process():
        print(f"Data loaded: there are {len(dataset)} images.")

    # ---- build official backbone ----
    student_backbone, embed_dim = build_backbone(args)
    teacher_backbone, _ = build_backbone(args)

    # ---- heads (keep handle for freezing policy) ----
    student_head = make_dino_head(embed_dim, args.out_dim, args.use_bn_in_head, args.norm_last_layer)
    teacher_head = make_dino_head(embed_dim, args.out_dim, args.use_bn_in_head, True)

    # ---- apply finetune policy BEFORE optimizer creation ----
    apply_finetune_policy(student_backbone, student_head, args)

    # ---- wrap with MultiCrop ----
    student = utils.MultiCropWrapper(student_backbone, student_head)
    teacher = utils.MultiCropWrapper(teacher_backbone, teacher_head)

    student, teacher = student.cuda(), teacher.cuda()

    # teacher DDP only if BN exists (usually false for ViT)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])

    # init teacher = student
    teacher_without_ddp.load_state_dict(student.module.state_dict())

    # freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    if utils.is_main_process():
        print(f"Student and Teacher are built: {args.arch} | embed_dim={embed_dim}")

    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # optimizer (should pick up requires_grad flags if utils.get_params_groups filters)
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    else:
        raise ValueError(args.optimizer)

    fp16_scaler = torch.cuda.amp.GradScaler() if args.use_fp16 else None

    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )

    if utils.is_main_process():
        print("Loss, optimizer and schedulers ready.")

    # resume
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    if utils.is_main_process():
        print("Starting DINO adaptation training!")

    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args
        )

        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "dino_loss": dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()

        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    if utils.is_main_process():
        print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))


def train_one_epoch(
    student, teacher, teacher_without_ddp, dino_loss, data_loader,
    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
    fp16_scaler, args
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}/{args.epochs}]"

    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        it = len(data_loader) * epoch + it

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]

        autocast_enabled = fp16_scaler is not None
        with torch.amp.autocast("cuda", enabled=autocast_enabled):
            teacher_output = teacher(images[:2])
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training", force=True)
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)

        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                _ = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                _ = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA teacher update
        with torch.no_grad():
            m = momentum_schedule[it]
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    metric_logger.synchronize_between_processes()
    if utils.is_main_process():
        print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(
        self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
        warmup_teacher_temp_epochs, nepochs, student_temp=0.1, center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        student_out = (student_output / self.student_temp).chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DINOv3 Adaptation", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)

    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
