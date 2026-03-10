#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import inspect
import shutil
from glob import glob

import torch
from torchinfo import summary

from predict_wsi_gene_dataset import MyDataSet as WSI_Gene_DataSet
from vit_model_gene_wsi_concat_predict import my_model as create_model_wsi_gene
from vit_model_gene_wsi_concat_no_contrastive_loss import my_model as create_model_wsi_gene_no_contrastive_loss
from vit_model_one_cls import my_model as create_model_wsi
from gene_only import my_gene_only_model
from utils_cox_predict import predict


def call_with_sig(fn, **kwargs):
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def sample_key(s: str) -> str:
    """
    Normalize sample IDs / filenames into canonical keys.

    Rules:
    - TCGA-XX-YYYY-... -> TCGA-XX-YYYY
    - C3L-00001-...    -> C3L-00001
    - C3N-00001-...    -> C3N-00001
    - otherwise: remove suffix and split at first '.' or '_'
    """
    if s is None:
        return ""

    s = os.path.basename(str(s))
    s = re.sub(r"\.(npy|pth|pt|txt)$", "", s)

    if s.startswith("TCGA-"):
        parts = s.split("-")
        if len(parts) >= 3:
            return "-".join(parts[:3])
        return s[:12]

    if s.startswith("C3L-") or s.startswith("C3N-"):
        parts = s.split("-")
        if len(parts) >= 2:
            return "-".join(parts[:2])
        return s

    s = re.split(r"[._]", s)[0]
    return s


def build_keymap_from_dir(feat_dir: str, exts):
    key2path = {}
    if (not feat_dir) or (not os.path.exists(feat_dir)):
        return key2path

    candidates = []
    for ext in exts:
        candidates.extend(glob(os.path.join(feat_dir, f"*{ext}")))

    for p in candidates:
        k = sample_key(p)
        if not k:
            continue
        if k not in key2path:
            key2path[k] = p
        else:
            if len(os.path.basename(p)) < len(os.path.basename(key2path[k])):
                key2path[k] = p
    return key2path


def make_symlink_or_copy(src: str, dst: str):
    if os.path.islink(dst) or os.path.exists(dst):
        try:
            if os.path.islink(dst):
                old = os.readlink(dst)
                if os.path.abspath(old) == os.path.abspath(src):
                    return
            os.remove(dst)
        except Exception:
            pass

    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rewrite_cox_and_make_symlinks(
    cox_in: str,
    wsi_dir: str,
    gene_dir: str,
    out_root: str,
    train_flag: int,
):
    """
    Normalize IDs in cox txt and build canonical feature links.

    Outputs:
      out_root/
        cox_norm.txt
        features_link/
          wsi/{sid}.txt
          gene/{sid}.npy
    """
    os.makedirs(out_root, exist_ok=True)

    feat_root = os.path.join(out_root, "features_link")
    wsi_link_dir = os.path.join(feat_root, "wsi")
    gene_link_dir = os.path.join(feat_root, "gene")
    os.makedirs(wsi_link_dir, exist_ok=True)
    os.makedirs(gene_link_dir, exist_ok=True)

    need_wsi = train_flag in (0, 1)
    need_gene = train_flag in (0, 2)

    wsi_map = build_keymap_from_dir(wsi_dir, exts=(".txt",)) if need_wsi else {}
    gene_map = build_keymap_from_dir(gene_dir, exts=(".npy",)) if need_gene else {}

    wsi_keys = set(wsi_map.keys()) if need_wsi else set()
    gene_keys = set(gene_map.keys()) if need_gene else set()

    if need_wsi and need_gene:
        keep_keys = wsi_keys & gene_keys
    elif need_wsi:
        keep_keys = wsi_keys
    elif need_gene:
        keep_keys = gene_keys
    else:
        keep_keys = set()

    cox_norm_path = os.path.join(out_root, "cox_norm.txt")

    total = 0
    kept = 0
    with open(cox_in, "r") as f_in, open(cox_norm_path, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            total += 1
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            raw_id, futime, fustat = parts[0], parts[1], parts[2]
            sid = sample_key(raw_id)

            if sid not in keep_keys:
                continue

            f_out.write(f"{sid}\t{futime}\t{fustat}\n")
            kept += 1

            if need_wsi:
                src = os.path.abspath(wsi_map[sid])
                dst = os.path.join(wsi_link_dir, f"{sid}.txt")
                make_symlink_or_copy(src, dst)

            if need_gene:
                src = os.path.abspath(gene_map[sid])
                dst = os.path.join(gene_link_dir, f"{sid}.npy")
                make_symlink_or_copy(src, dst)

    print(f"[COX rewrite] in={cox_in} total={total} kept={kept} out={cox_norm_path}")
    print(f"[COX rewrite] need_wsi={need_wsi} need_gene={need_gene} "
          f"wsi_keys={len(wsi_keys)} gene_keys={len(gene_keys)} keep={len(keep_keys)}")

    if kept == 0:
        print("[WARN] No matched samples after ID normalization.")
        if need_wsi and os.path.exists(wsi_dir):
            ex = os.listdir(wsi_dir)[:5]
            print("  WSI examples:", ex)
        if need_gene and os.path.exists(gene_dir):
            ex = os.listdir(gene_dir)[:5]
            print("  GENE examples:", ex)

    return cox_norm_path, wsi_link_dir, gene_link_dir


def ensure_head_dirs(save_attn_dir: str, heads: int = 16):
    os.makedirs(save_attn_dir, exist_ok=True)
    for h in range(heads):
        os.makedirs(os.path.join(save_attn_dir, f"head{h}"), exist_ok=True)


def build_model(args):
    if args.train_flag == 0:
        if args.contrastive_loss_flag == 1:
            model = call_with_sig(
                create_model_wsi_gene,
                num_classes=args.num_classes,
                has_logits=False,
                wsi_block=args.wsi_block,
                gene_block=args.gene_block,
                dpr=args.dpr,
                proto_k=args.proto_k,
                proto_tau=args.proto_tau,
                return_attn=True,
            )
        else:
            model = call_with_sig(
                create_model_wsi_gene_no_contrastive_loss,
                num_classes=args.num_classes,
                has_logits=False,
                wsi_block=args.wsi_block,
                gene_block=args.gene_block,
                dpr=args.dpr,
                proto_k=args.proto_k,
                proto_tau=args.proto_tau,
            )

    elif args.train_flag == 1:
        model = call_with_sig(
            create_model_wsi,
            num_classes=args.num_classes,
            has_logits=False,
            wsi_block=args.wsi_block,
            dpr=args.dpr,
            proto_k=args.proto_k,
            proto_tau=args.proto_tau,
        )

    elif args.train_flag == 2:
        model = my_gene_only_model(
            num_classes=args.num_classes,
            gene_block=args.gene_block,
            dpr=args.dpr,
        )

    else:
        raise ValueError(f"Invalid train_flag: {args.train_flag}")

    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.save_attn_dir, exist_ok=True)

    print("output_dir:", os.path.abspath(args.output_dir))
    print("save_attn_dir:", os.path.abspath(args.save_attn_dir))
    print("cox_path:", os.path.abspath(args.cox_path))
    print("wsi_feat_dir:", os.path.abspath(args.wsi_feat_dir))
    print("gene_feat_dir:", os.path.abspath(args.gene_feat_dir))
    print("weights:", os.path.abspath(args.weights))

    cox_norm, wsi_link_dir, gene_link_dir = rewrite_cox_and_make_symlinks(
        cox_in=args.cox_path,
        wsi_dir=args.wsi_feat_dir,
        gene_dir=args.gene_feat_dir,
        out_root=args.output_dir,
        train_flag=args.train_flag,
    )

    dataset = WSI_Gene_DataSet(
        wsi_link_dir,
        gene_link_dir,
        cox_norm,
        mode="valid",
    )
    print("inference patient count:", len(dataset))

    batch_size = args.batch_size
    nw = min([os.cpu_count() or 0, batch_size if batch_size > 1 else 0, 8])
    print(f"Using {nw} dataloader workers")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=nw,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    model = build_model(args)

    model_log_path = os.path.join(args.output_dir, "model_log.txt")
    with open(model_log_path, "w") as f:
        f.write(str(model) + "\n")
        f.write("Total params: {:.2f}M\n".format(sum(p.numel() for p in model.parameters()) / 1e6))

    try:
        with open(model_log_path, "a") as f:
            if args.train_flag == 0:
                f.write(str(summary(model, [(1, 500, 384), (1, 236, 6292)], device="cpu")))
            elif args.train_flag == 1:
                f.write(str(summary(model, input_size=(1, 500, 384), device="cpu")))
            elif args.train_flag == 2:
                f.write(str(summary(model, input_size=(1, 236, 6292), device="cpu")))
    except Exception as e:
        print("[WARN] torchinfo summary failed:", e)

    model = model.to(device)

    assert os.path.exists(args.weights), f"weights file not exist: {args.weights}"
    sd = torch.load(args.weights, map_location="cpu")

    if isinstance(sd, dict) and ("model" in sd):
        sd = sd["model"]

    if isinstance(sd, dict) and len(sd) > 0:
        k0 = next(iter(sd.keys()))
        if k0.startswith("module."):
            sd = {k[len("module."):]: v for k, v in sd.items()}

    msg = model.load_state_dict(sd, strict=False)
    print("[load_state_dict]", msg)

    model.eval()

    heads = 16
    try:
        heads = int(getattr(getattr(model, "gene_guided_wsi_fusion", None), "num_heads", heads))
    except Exception:
        pass
    ensure_head_dirs(args.save_attn_dir, heads=heads)

    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    metrics_path = os.path.join(args.output_dir, "predict_log.txt")

    with torch.no_grad():
        val_loss, val_cox_acc, val_p_value, val_c_index = predict(
            model=model,
            topK=args.topK,
            criterion=criterion,
            data_loader=data_loader,
            json_path=metrics_path,
            save_attn_dir=args.save_attn_dir,
            reg_loss=False,
            train_flag=args.train_flag,
            contrastive_loss_flag=args.contrastive_loss_flag,
        )

    print(
        f"loss={val_loss:.6f} "
        f"cox_acc={val_cox_acc:.6f} "
        f"p_value={val_p_value:.6g} "
        f"c_index={val_c_index:.6f}"
    )

    with open(metrics_path, "w") as f:
        f.write(f"loss\t{val_loss}\n")
        f.write(f"cox_acc\t{val_cox_acc}\n")
        f.write(f"p_value\t{val_p_value}\n")
        f.write(f"c_index\t{val_c_index}\n")

    print("saved metrics to:", os.path.abspath(metrics_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and attention export for trained PRISM survival model")

    parser.add_argument("--cox_path", type=str, required=True, help="cox txt path: id time event")
    parser.add_argument("--wsi_feat_dir", type=str, required=True, help="directory of WSI feature txt files")
    parser.add_argument("--gene_feat_dir", type=str, required=True, help="directory of gene feature npy files")
    parser.add_argument("--weights", type=str, required=True, help="model weights or checkpoint path")

    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    parser.add_argument("--save_attn_dir", type=str, default="", help="directory for exported attention weights")

    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--topK", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--train_flag", type=int, default=0, help="0: wsi+gene, 1: wsi-only, 2: gene-only")
    parser.add_argument("--contrastive_loss_flag", type=int, default=1, help="0: no contrastive loss, 1: contrastive loss")

    parser.add_argument("--wsi_block", type=int, default=2)
    parser.add_argument("--gene_block", type=int, default=1)
    parser.add_argument("--dpr", type=float, default=0.3)
    parser.add_argument("--proto_k", type=int, default=4)
    parser.add_argument("--proto_tau", type=float, default=0.1)

    args = parser.parse_args()

    if not args.save_attn_dir:
        args.save_attn_dir = os.path.join(args.output_dir, "attn")

    main(args)
