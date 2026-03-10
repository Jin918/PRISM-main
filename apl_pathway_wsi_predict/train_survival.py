import os
import math
import argparse
import shutil
import random
import numpy as np
import inspect

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torch.nn import DataParallel

from wsi_gene_dataset import MyDataSet as WSI_Gene_DataSet
from vit_model_gene_wsi_concat import my_model as create_model_wsi_gene
from vit_model_gene_wsi_concat_no_contrastive_loss import my_model as create_model_wsi_gene_no_contrastive_loss
from vit_model_one_cls import my_model as create_model_wsi
from utils_cox import train_one_epoch, evaluate
from gene_only import my_gene_only_model


def call_with_sig(fn, **kwargs):
    sig = inspect.signature(fn)

    # 如果函数/类构造器支持 **kwargs，就不要过滤，直接传
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)

    # 否则只保留签名里明确存在的参数
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


# -------------------------
# Helpers for checkpointing
# -------------------------
def unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def strip_module_prefix(state_dict):
    """Convert a DataParallel state_dict (module.xxx) -> plain (xxx)"""
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def is_full_checkpoint(obj):
    """Our full ckpt contains model/optimizer/scheduler/epoch at least."""
    return (
        isinstance(obj, dict)
        and ("model" in obj)
        and ("optimizer" in obj)
        and ("scheduler" in obj)
        and ("epoch" in obj)
    )


def load_surv_txt(path):
    """
    读取 cox train/val txt：每行至少包含 [id, futime, fustat]
    fustat: 1=event, 0=censored
    """
    times = []
    events = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            futime = float(parts[1])
            fustat = int(float(parts[2]))
            times.append(futime)
            events.append(fustat)
    return np.asarray(times, dtype=float), np.asarray(events, dtype=int)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 尽量可复现（会略降速）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    os.makedirs(args.log_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    model_file = None  # for finally

    try:
        # -------------------------
        # Dataset / DataLoader
        # -------------------------
        train_dataset = WSI_Gene_DataSet(
            args.wsi_train_feat_dir,
            args.gene_train_feat_dir,
            args.cox_train_path,
            mode="train",
            train_flag=args.train_flag,
            view_seed=args.view_seed,
        )
        print("train patient count: {}".format(str(len(train_dataset))))

        val_dataset = WSI_Gene_DataSet(
            args.wsi_valid_feat_dir,
            args.gene_valid_feat_dir,
            args.cox_val_path,
            mode="valid",
            train_flag=args.train_flag,
            view_seed=args.view_seed,
        )
        print("valid patient count: {}".format(str(len(val_dataset))))

        # -------------------------
        # IPCW: censoring distribution estimated from TRAIN set
        # -------------------------
        train_times_ipcw, train_events_ipcw = load_surv_txt(args.cox_train_path)

        roc_times = [float(y) * float(args.days_per_year) for y in args.roc_years]
        print("[TD-ROC] eval times:", roc_times, "(same unit as futime)")

        batch_size = args.batch_size
        nw = min([os.cpu_count() or 0, batch_size if batch_size > 1 else 0, 8])
        print("Using {} dataloader workers every process".format(nw))

        # train loader (shuffle=True for training)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=nw,
            drop_last=False,
            collate_fn=train_dataset.collate_fn,
        )

        # 单独构造一个“确定性”的 train-eval dataset（mode="valid"）
        train_dataset_eval = WSI_Gene_DataSet(
            args.wsi_train_feat_dir,
            args.gene_train_feat_dir,
            args.cox_train_path,
            mode="valid",
            train_flag=args.train_flag,
            view_seed=args.view_seed,
        )

        train_loader_eval = torch.utils.data.DataLoader(
            train_dataset_eval,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=nw,
            drop_last=False,
            collate_fn=train_dataset_eval.collate_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=nw,
            drop_last=False,
            collate_fn=val_dataset.collate_fn,
        )

        # -------------------------
        # Build model
        # -------------------------
        model_dpr = args.dpr
        wsi_block = args.wsi_block
        gene_block = args.gene_block

        proto_k = int(getattr(args, "proto_k", 0) or 0)
        proto_tau = float(getattr(args, "proto_tau", 0.07))
        use_apl = (proto_k > 0)

        # proto_k<=0 时强制把 APL 正则清零
        if not use_apl:
            args.lambda_div = 0.0
            args.lambda_bal = 0.0

        if args.train_flag == 0:
            if args.contrastive_loss_flag == 1:
                model = call_with_sig(
                    create_model_wsi_gene,
                    num_classes=args.num_classes,
                    has_logits=False,
                    wsi_block=wsi_block,
                    gene_block=gene_block,
                    dpr=model_dpr,
                    proto_k=(proto_k if use_apl else 0),
                    proto_tau=proto_tau,
                    return_contrastive=True,
                )
            else:
                model = call_with_sig(
                    create_model_wsi_gene_no_contrastive_loss,
                    num_classes=args.num_classes,
                    has_logits=False,
                    wsi_block=wsi_block,
                    gene_block=gene_block,
                    dpr=model_dpr,
                    proto_k=(proto_k if use_apl else 0),
                    proto_tau=proto_tau,
                )

        elif args.train_flag == 1:
            model = call_with_sig(
                create_model_wsi,
                num_classes=args.num_classes,
                has_logits=False,
                wsi_block=wsi_block,
                dpr=model_dpr,
                proto_k=(proto_k if use_apl else 0),
                proto_tau=proto_tau,
            )

        elif args.train_flag == 2:
            args.contrastive_loss_flag = 0
            model = my_gene_only_model(
                num_classes=args.num_classes,
                gene_block=gene_block,
                dpr=model_dpr,
            )
        else:
            raise ValueError(f"Invalid train flag: {args.train_flag}")

        # -------------------------
        # Model summary log
        # -------------------------
        model_log_path = os.path.join(args.log_dir, "model_log.txt")
        with open(model_log_path, "w") as f:
            f.write(str(model))
            f.write("\n")
            f.write("Total params: {:.2f}M\n".format(sum(p.numel() for p in model.parameters()) / 1e6))

        model = model.to(device)

        try:
            with open(model_log_path, "a") as f:
                if args.train_flag == 0:
                    f.write(str(summary(model, [(1, 500, 384), (1, 236, 6292)], device="cuda")))
                elif args.train_flag == 1:
                    f.write(str(summary(model, input_size=(1, 500, 384), device="cuda")))
                elif args.train_flag == 2:
                    f.write(str(summary(model, input_size=(1, 236, 6292), device="cuda")))
        except Exception as e:
            print("[WARN] torchinfo summary failed:", e)

        print("HAS APL:", getattr(model, "apl_wsi", None) is not None, "proto_k:", getattr(model, "proto_k", None))

        # Wrap with DataParallel
        if device.type == "cuda" and torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model = model.to(device)

        # -------------------------
        # Resume logic
        # -------------------------
        ckpt = None
        start_epoch = 0
        best_val_cindex = 0.0
        best_sum_cindex = 0.0

        ckpt_latest_path = os.path.join(args.log_dir, "ckpt-latest.pt")

        resume_path = ""
        if not args.no_resume:
            if args.resume:
                resume_path = args.resume
            elif os.path.exists(ckpt_latest_path):
                resume_path = ckpt_latest_path

        if resume_path:
            assert os.path.exists(resume_path), f"resume file not exist: {resume_path}"
            loaded = torch.load(resume_path, map_location="cpu")

            if is_full_checkpoint(loaded):
                ckpt = loaded
                sd = strip_module_prefix(ckpt["model"])
                unwrap_model(model).load_state_dict(sd, strict=True)

                best_val_cindex = float(ckpt.get("best_val_cindex", 0.0))
                best_sum_cindex = float(ckpt.get("best_sum_cindex", 0.0))
                start_epoch = int(ckpt.get("epoch", -1)) + 1

                print(f"[RESUME] full checkpoint loaded: {resume_path}")
                print(f"[RESUME] start_epoch={start_epoch}, best_val={best_val_cindex}, best_sum={best_sum_cindex}")
            else:
                sd = strip_module_prefix(loaded)
                unwrap_model(model).load_state_dict(sd, strict=False)
                start_epoch = int(args.start_epoch)

                print(f"[RESUME] legacy weights loaded: {resume_path}")
                print(f"[RESUME] start_epoch(from args.start_epoch)={start_epoch}")

        did_resume = bool(resume_path)

        # -------------------------
        # Optional initial weights (ONLY when not resuming)
        # -------------------------
        if (not did_resume) and args.weights != "":
            assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
            weights_dict = torch.load(args.weights, map_location="cpu")

            base_model = unwrap_model(model)
            del_keys = ["head.weight", "head.bias"] if getattr(base_model, "has_logits", False) else [
                "pre_logits.fc.weight",
                "pre_logits.fc.bias",
                "head.weight",
                "head.bias",
                "patch_embed.proj.bias",
                "patch_embed.proj.weight",
            ]
            for k in del_keys:
                if k in weights_dict:
                    del weights_dict[k]

            try:
                torch.nn.init.kaiming_normal_(base_model.patch_embed.proj_conv.weight, mode="fan_out", nonlinearity="relu")
                torch.nn.init.kaiming_normal_(base_model.patch_embed.proj_lin.weight, mode="fan_out", nonlinearity="relu")
            except Exception as e:
                print("[WARN] kaiming init failed:", e)

            base_model = unwrap_model(model)
            weights_dict = strip_module_prefix(weights_dict)
            msg = base_model.load_state_dict(weights_dict, strict=False)
            print(msg)

        # -------------------------
        # Freeze layers (optional)
        # -------------------------
        if args.freeze_layers:
            for name, para in model.named_parameters():
                if ("head" not in name) and ("pre_logits" not in name):
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

        # -------------------------
        # Optimizer / Scheduler / Criterion
        # -------------------------
        pg = [p for p in model.parameters() if p.requires_grad]
        regular_loss = None

        optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.weight_decay)

        total_epochs_for_schedule = int(ckpt.get("total_epochs", args.epochs)) if ckpt else int(args.epochs)

        def lf(x):
            if x <= total_epochs_for_schedule:
                return ((1 + math.cos(x * math.pi / total_epochs_for_schedule)) / 2) * (1 - args.lrf) + args.lrf
            else:
                return args.lrf

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=-1)

        if ckpt is not None and is_full_checkpoint(ckpt):
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"[RESUME] optimizer/scheduler restored. scheduler.last_epoch={scheduler.last_epoch}")
        else:
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf, last_epoch=start_epoch - 1)

        criterion = torch.nn.BCEWithLogitsLoss().to(device)

        # -------------------------
        # Helper: dump best preds (train/valid) with sid
        # -------------------------
        def dump_best_preds(epoch_idx: int):
            """
            在 best-val epoch 触发：
              - pred_train_best.tsv
              - pred_valid_best.tsv
            """
            out_train = os.path.join(args.log_dir, "pred_train_best.tsv")
            out_valid = os.path.join(args.log_dir, "pred_valid_best.tsv")
            out_ep = os.path.join(args.log_dir, "best_val_epoch.txt")

            evaluate(
                model=model,
                topK=args.topK,
                criterion=criterion,
                data_loader=train_loader_eval,
                epoch=epoch_idx,
                json_path=os.path.join(args.log_dir, "train_eval_log.txt"),
                split_name="train",
                reg_loss=regular_loss,
                train_flag=args.train_flag,
                contrastive_loss_flag=args.contrastive_loss_flag,
                lambda_div=args.lambda_div,
                lambda_bal=args.lambda_bal,
                train_times_ipcw=None,
                train_events_ipcw=None,
                roc_times=None,
                save_roc_path=None,
                save_pred_path=out_train,
            )

            evaluate(
                model=model,
                topK=args.topK,
                criterion=criterion,
                data_loader=val_loader,
                epoch=epoch_idx,
                json_path=os.path.join(args.log_dir, "valid_log.txt"),
                split_name="valid",
                reg_loss=regular_loss,
                train_flag=args.train_flag,
                contrastive_loss_flag=args.contrastive_loss_flag,
                lambda_div=args.lambda_div,
                lambda_bal=args.lambda_bal,
                train_times_ipcw=None,
                train_events_ipcw=None,
                roc_times=None,
                save_roc_path=None,
                save_pred_path=out_valid,
            )

            with open(out_ep, "w") as f:
                f.write(str(epoch_idx + 1) + "\n")

            print(f"[DUMP] best preds saved: {out_train}, {out_valid}", flush=True)

        # -------------------------
        # Train loop
        # -------------------------
        save_name_txt = os.path.join(args.log_dir, "train_valid_acc.txt")
        model_file = open(save_name_txt, "a" if start_epoch > 0 else "w")

        for epoch in range(start_epoch, args.epochs):
            do_roc = ((epoch + 1) % int(args.roc_every) == 0)
            save_roc_path = os.path.join(args.log_dir, f"tdroc_epoch{epoch+1:03d}.png") if do_roc else None
            roc_times_pass = roc_times if do_roc else None

            train_loss, train_cox_acc, train_p_value, train_c_index = train_one_epoch(
                model=model,
                topK=args.topK,
                criterion=criterion,
                optimizer=optimizer,
                data_loader=train_loader,
                epoch=epoch,
                reg_loss=regular_loss,
                train_flag=args.train_flag,
                contrastive_loss_flag=args.contrastive_loss_flag,
                lambda_div=args.lambda_div,
                lambda_bal=args.lambda_bal,
            )

            scheduler.step()

            val_loss, val_cox_acc, val_p_value, val_c_index, tdroc = evaluate(
                model=model,
                topK=args.topK,
                criterion=criterion,
                data_loader=val_loader,
                epoch=epoch,
                json_path=os.path.join(args.log_dir, "valid_log.txt"),
                split_name="valid",
                reg_loss=regular_loss,
                train_flag=args.train_flag,
                contrastive_loss_flag=args.contrastive_loss_flag,
                lambda_div=args.lambda_div,
                lambda_bal=args.lambda_bal,
                train_times_ipcw=train_times_ipcw,
                train_events_ipcw=train_events_ipcw,
                roc_times=roc_times_pass,
                save_roc_path=save_roc_path,
                save_pred_path=None,
            )

            # -------------------------
            # Log time-dependent AUCs
            # -------------------------
            auc_str = ""
            if tdroc is not None:
                for y in args.roc_years:
                    t = float(y) * float(args.days_per_year)
                    auc = tdroc.get(t, {}).get("auc", float("nan"))
                    tb_writer.add_scalar(f"val_auc_{y}y", auc, epoch)
                    auc_str += f" AUC{y}y={auc:.3f}"

            # TensorBoard scalars
            tb_writer.add_scalar("train_loss", train_loss, epoch)
            tb_writer.add_scalar("train_cox_acc", train_cox_acc, epoch)
            tb_writer.add_scalar("train_p_value", train_p_value, epoch)
            tb_writer.add_scalar("train_c_index", train_c_index, epoch)

            tb_writer.add_scalar("val_loss", val_loss, epoch)
            tb_writer.add_scalar("val_cox_acc", val_cox_acc, epoch)
            tb_writer.add_scalar("val_p_value", val_p_value, epoch)
            tb_writer.add_scalar("val_c_index", val_c_index, epoch)

            tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

            msg = (
                f"[Epoch {epoch+1:03d}/{args.epochs:03d}] "
                f"train: loss={train_loss:.4f} acc={train_cox_acc:.4f} cindex={train_c_index:.4f} p={train_p_value:.3g} | "
                f"val: loss={val_loss:.4f} acc={val_cox_acc:.4f} cindex={val_c_index:.4f} p={val_p_value:.3g}{auc_str} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e}"
            )
            print(msg, flush=True)

            # Text log
            model_file.write(
                f"Train-Epoch-{epoch} : train loss : {train_loss} ; train cox acc : {train_cox_acc}"
                f" ; train p value : {train_p_value} ; train c index : {train_c_index}\n"
            )
            model_file.write(
                f"Valid-Epoch-{epoch} : valid loss : {val_loss} ; valid cox acc : {val_cox_acc}"
                f" ; valid p value : {val_p_value} ; valid c index : {val_c_index}\n"
            )
            if tdroc is not None:
                model_file.write(f"TDROC-Epoch-{epoch} :" + auc_str + "\n")
            model_file.write(f"lrlrl-Epoch-{epoch} : learning rate : {optimizer.param_groups[0]['lr']}\n")
            model_file.flush()

            # ---------
            # Save latest legacy weights
            # ---------
            torch.save(unwrap_model(model).state_dict(), os.path.join(args.log_dir, "model-latest.pth"))

            # ---------
            # Decide whether improves best (BEFORE updating best_*)
            # ---------
            val_sum = float(val_c_index + train_c_index)
            improved_val = float(val_c_index) > float(best_val_cindex)
            improved_sum = val_sum > float(best_sum_cindex)

            # ---------
            # Update bests + save best weights
            # ---------
            if improved_val:
                best_val_path = os.path.join(args.log_dir, "model-val-best.pth")
                torch.save(unwrap_model(model).state_dict(), best_val_path)

                if train_c_index >= 0.7:
                    snap_path = os.path.join(args.log_dir, f"model-val-{val_c_index:.4f}.pth")
                    shutil.copy2(best_val_path, snap_path)

                best_val_cindex = float(val_c_index)
                model_file.write(f"save best val c_index {val_c_index} checkpoint\n")
                model_file.flush()

                dump_best_preds(epoch)

            if improved_sum:
                best_sum_path = os.path.join(args.log_dir, "model-sum-best.pth")
                torch.save(unwrap_model(model).state_dict(), best_sum_path)

                if train_c_index >= 0.7:
                    snap_path = os.path.join(args.log_dir, f"model-sum-{val_c_index:.4f}.pth")
                    shutil.copy2(best_sum_path, snap_path)

                best_sum_cindex = float(val_sum)
                model_file.write(f"save best sum c_index {best_sum_cindex} checkpoint\n")
                model_file.flush()

            # ---------
            # Save FULL checkpoint (always latest)
            # ---------
            full_ckpt = {
                "epoch": epoch,
                "total_epochs": int(args.epochs),
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_cindex": float(best_val_cindex),
                "best_sum_cindex": float(best_sum_cindex),
                "args": vars(args),
            }
            torch.save(full_ckpt, os.path.join(args.log_dir, "ckpt-latest.pt"))

            # Save FULL best ckpts ONLY when improved
            if improved_val:
                torch.save(full_ckpt, os.path.join(args.log_dir, "ckpt-val-best.pt"))
            if improved_sum:
                torch.save(full_ckpt, os.path.join(args.log_dir, "ckpt-sum-best.pt"))

            # -------------------------
            # Save per-epoch ckpt/weights (0-based epoch index)
            # -------------------------
            save_every = max(1, int(getattr(args, "save_every", 1)))
            if (getattr(args, "save_epoch_ckpt", False) or getattr(args, "save_epoch_weights", False)) and (epoch % save_every == 0):
                ep_tag = f"{epoch:03d}"  # 0-based: 000..017

                if getattr(args, "save_epoch_ckpt", False):
                    torch.save(full_ckpt, os.path.join(args.log_dir, f"ckpt-epoch-{ep_tag}.pt"))

                if getattr(args, "save_epoch_weights", False):
                    torch.save(unwrap_model(model).state_dict(), os.path.join(args.log_dir, f"model-epoch-{ep_tag}.pth"))

    finally:
        try:
            if model_file is not None:
                model_file.close()
        except Exception:
            pass
        try:
            tb_writer.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--topK", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lrf", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--log_dir",
        type=str,
        default="/root/code/PRISM-main/datasets/pamt_287/log/run_wsi6_2026021012",
        help="log directory",
    )

    parser.add_argument(
        "--train_flag",
        type=int,
        default=0,
        help="train mode, 0: wsi + gene, 1: wsi, 2: gene",
    )
    parser.add_argument(
        "--contrastive_loss_flag",
        type=int,
        default=1,
        help="0: no contrastive loss, 1: contrastive loss",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--view_seed", type=int, default=42)

    parser.add_argument("--cox_train_path", type=str, default="/root/code/PRISM-main/datasets/pamt_287/cox/train.txt")
    parser.add_argument("--cox_val_path", type=str, default="/root/code/PRISM-main/datasets/pamt_287/cox/val.txt")

    parser.add_argument("--wsi_train_feat_dir", type=str, default="/root/code/PRISM-main/datasets/pamt_287/wsi/train")
    parser.add_argument("--wsi_valid_feat_dir", type=str, default="/root/code/PRISM-main/datasets/pamt_287/wsi/val")

    parser.add_argument("--gene_train_feat_dir", type=str, default="/root/code/PRISM-main/datasets/pamt_287/gene/train")
    parser.add_argument("--gene_valid_feat_dir", type=str, default="/root/code/PRISM-main/datasets/pamt_287/gene/val")

    parser.add_argument("--weights", type=str, default="", help="initial weights path (only used when not resuming)")
    parser.add_argument("--freeze_layers", action="store_true")

    parser.add_argument("--device", default="0,1,2,3,4,5,6,7", type=str, help="device id (unused here)")

    parser.add_argument("--resume", type=str, default="", help="path to full ckpt-latest.pt OR legacy model-latest.pth")
    parser.add_argument("--start_epoch", type=int, default=0, help="only used when resuming legacy .pth")
    parser.add_argument("--no_resume", action="store_true", help="do not auto-resume from log_dir/ckpt-latest.pt")

    # APL
    parser.add_argument("--proto_k", type=int, default=4)
    parser.add_argument("--proto_tau", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.0)
    parser.add_argument("--lambda_bal", type=float, default=0.1)

    parser.add_argument("--wsi_block", type=int, default=2, help="num transformer blocks for WSI branch")
    parser.add_argument("--gene_block", type=int, default=1, help="num transformer blocks for Gene branch")
    parser.add_argument("--dpr", type=float, default=0.3, help="drop_path_rate (stochastic depth), passed as dpr")

    # AUC
    parser.add_argument(
        "--roc_years",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="time-dependent ROC at these years",
    )
    parser.add_argument(
        "--days_per_year",
        type=float,
        default=12.0,
        help="set 365 if futime is in days; set 12 if futime is in months",
    )
    parser.add_argument(
        "--roc_every",
        type=int,
        default=1,
        help="compute td-ROC every N epochs (reduce overhead)",
    )

    parser.add_argument("--save_epoch_ckpt", action="store_true",
                        help="save full checkpoint per epoch as ckpt-epoch-XXX.pt (XXX is 0-based epoch)")
    parser.add_argument("--save_epoch_weights", action="store_true",
                        help="save model weights per epoch as model-epoch-XXX.pth (XXX is 0-based epoch)")
    parser.add_argument("--save_every", type=int, default=1,
                        help="save per-epoch checkpoint/weights every N epochs")

    opt = parser.parse_args()
    main(opt)
    main(opt)
