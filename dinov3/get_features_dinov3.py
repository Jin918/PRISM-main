# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the Apache License, Version 2.0

import os
import argparse
import json

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms

import utils  # PRISM-main/dino/utils.py (init_distributed_mode, MetricLogger, bool_flag)

from PIL import Image, ImageFile, PngImagePlugin

# ---- PIL safety knobs for huge pathology patches ----
PngImagePlugin.MAX_TEXT_CHUNK = 500 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---- Official DINOv3 hub entry ----
from dinov3.hub import backbones


def load_backbone_from_checkpoint(backbone: nn.Module, ckpt_path: str, checkpoint_key: str = "teacher"):
    """
    Load a backbone-only state_dict from a checkpoint that may contain:
      - {'teacher': state_dict, 'student': state_dict, ...}
      - {'model': state_dict, ...}
      - state_dict directly
    This ignores non-backbone keys (e.g., DINO head) via strict=False.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    sd = ckpt
    if isinstance(ckpt, dict):
        if checkpoint_key in ckpt:
            sd = ckpt[checkpoint_key]
        elif "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]

    if not isinstance(sd, dict):
        raise ValueError(f"Unexpected checkpoint format: {type(sd)}")

    cleaned = {}
    for k, v in sd.items():
        kk = k.replace("module.", "")
        kk = kk.replace("backbone.", "")  # MultiCropWrapper/your checkpoints often use this prefix
        cleaned[kk] = v

    if "storage_tokens" in cleaned and "register_tokens" not in cleaned:
        cleaned["register_tokens"] = cleaned.pop("storage_tokens")

    msg = backbone.load_state_dict(cleaned, strict=False)
    print(f"[load_backbone_from_checkpoint] loaded from {ckpt_path} key={checkpoint_key} msg={msg}")


def build_dinov3_backbone(args) -> nn.Module:
    if not hasattr(backbones, args.arch):
        avail = [n for n in dir(backbones) if n.startswith("dinov3_vit")]
        raise ValueError(f"Unknown arch: {args.arch}. Available: {avail}")

    fn = getattr(backbones, args.arch)

    # Mode 1: load from your finetuned checkpoint (teacher/student checkpoint)
    if args.checkpoint_pth and args.checkpoint_pth.strip():
        backbone = fn(pretrained=False)  # build architecture only
        load_backbone_from_checkpoint(backbone, args.checkpoint_pth, args.checkpoint_key)
        return backbone

    # Mode 2: official pretrained weights (recommended baseline)
    # weights can be local .pth path string, or Weights enum name like 'LVD1689M'
    return fn(pretrained=True, weights=args.weights, check_hash=False)


def extract_feature_pipeline(args):
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = ReturnIndexDataset(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded: {len(dataset)} images.")

    model = build_dinov3_backbone(args).cuda().eval()

    if utils.get_rank() == 0:
        dim = getattr(model, "embed_dim", None)
        print(f"Using backbone: {args.arch} | embed_dim={dim} | weights={args.weights} | ckpt={args.checkpoint_pth}")

    feats = extract_features(model, loader, use_cuda=args.use_cuda, amp=args.amp)

    if utils.get_rank() == 0:
        feats = nn.functional.normalize(feats, dim=1, p=2)

    paths = [p for p, _ in dataset.samples]  # keep alignment with features rows

    if utils.get_rank() == 0 and args.dump_features:
        os.makedirs(args.dump_features, exist_ok=True)
        torch.save(feats.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        with open(os.path.join(args.dump_features, "train_paths.json"), "w") as f:
            json.dump(paths, f)

        meta = {
            "arch": args.arch,
            "weights": args.weights,
            "checkpoint_pth": args.checkpoint_pth,
            "checkpoint_key": args.checkpoint_key,
            "n_images": len(paths),
            "feat_dim": int(feats.shape[1]),
            "transform": "Resize(256)->CenterCrop(224)->ImageNetNorm",
        }
        with open(os.path.join(args.dump_features, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print("Saved:")
        print(" -", os.path.join(args.dump_features, "trainfeat.pth"))
        print(" -", os.path.join(args.dump_features, "train_paths.json"))
        print(" -", os.path.join(args.dump_features, "meta.json"))


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, amp=True):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None

    for samples, index in metric_logger.log_every(data_loader, 50):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            feats = model(samples).clone()

        if dist.get_rank() == 0 and features is None:
            device = "cuda" if use_cuda else "cpu"
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1], device=device)
            print(f"Storing features into tensor of shape {tuple(features.shape)}")

        # gather indices
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        h1 = dist.all_gather(y_l, index, async_op=True)
        h1.wait()
        index_all = torch.cat(y_l)

        # gather feats
        feats_all = torch.empty(dist.get_world_size(), feats.size(0), feats.size(1), dtype=feats.dtype, device=feats.device)
        out_l = list(feats_all.unbind(0))
        h2 = dist.all_gather(out_l, feats, async_op=True)
        h2.wait()

        if dist.get_rank() == 0:
            features.index_copy_(0, index_all, torch.cat(out_l))

    return features


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        return img, idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Official DINOv3 Feature Extraction (WSI patches)")

    parser.add_argument("--batch_size_per_gpu", default=256, type=int)
    parser.add_argument("--num_workers", default=10, type=int)

    parser.add_argument("--data_path", default="/Pathology_data/UCEC/patch_data_whole_slide_20X_256", type=str)
    parser.add_argument("--dump_features", default="/Pathology_data/UCEC/UCEC_features_dinov3_pretrain", type=str)

    parser.add_argument("--arch", default="dinov3_vits16plus", type=str)

    # Baseline: official pretrained weights (local path or enum like 'LVD1689M')
    parser.add_argument(
        "--weights",
        default="/root/code/dinov3-main/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        type=str
    )

    # Optional: load from your finetuned checkpoint instead
    parser.add_argument("--checkpoint_pth", default="", type=str)
    parser.add_argument("--checkpoint_key", default="teacher", type=str)

    parser.add_argument("--use_cuda", default=True, type=utils.bool_flag)
    parser.add_argument("--amp", default=True, type=utils.bool_flag)

    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    utils.init_distributed_mode(args)
    cudnn.benchmark = True

    if utils.get_rank() == 0:
        print("git:\n  {}\n".format(utils.get_sha()))
        print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(vars(args).items())))

    extract_feature_pipeline(args)

    dist.barrier()
