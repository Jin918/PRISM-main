import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"

expr_path = DATASETS_DIR / "processed" / "TCGA_UCEC_log_FPKM_pathway_normolized.csv"
pwy_path = DATASETS_DIR / "processed" / "MSigDB_2sets_co_genes.csv"

split_dir = DATASETS_DIR / "model_inputs" / "pamt_287" / "split_seed302"
train_txt = split_dir / "train.txt"
val_txt = split_dir / "val.txt"

out_root = DATASETS_DIR / "model_inputs" / "pamt_287" / "gene"
(out_root / "train").mkdir(parents=True, exist_ok=True)
(out_root / "val").mkdir(parents=True, exist_ok=True)


def to_pid(x: str) -> str:
    return str(x)[:12]


def read_ids_from_txt(path: Path):
    ids = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) >= 1:
                ids.append(parts[0])
    return ids


# 1) load
expr = pd.read_csv(expr_path, index_col=0)
expr.index = expr.index.astype(str)       # genes
expr.columns = expr.columns.astype(str)   # samples

pwy = pd.read_csv(pwy_path, index_col=0)
pwy.index = pwy.index.astype(str)         # pathways
pwy.columns = pwy.columns.astype(str)     # genes

# 2) pathway membership -> 0/1
pwy_num = pd.to_numeric(pwy.stack(), errors="coerce").unstack()
if pwy_num.notna().sum().sum() == 0:
    pwy_bin = pwy.notna().astype(np.float32)
else:
    pwy_bin = (pwy_num.fillna(0) > 0).astype(np.float32)

# 3) 对齐基因顺序
gene_order = list(pwy_bin.columns)
expr2 = expr.reindex(gene_order).fillna(0.0)

# 4) 汇总到病人层面：同一病人多个 barcode 取均值
expr2.columns = [to_pid(c) for c in expr2.columns]
expr_pid = expr2.T.groupby(level=0).mean().T   # (G, patients)

train_ids = read_ids_from_txt(train_txt)
val_ids = read_ids_from_txt(val_txt)

train_ids = [i for i in train_ids if i in expr_pid.columns]
val_ids = [i for i in val_ids if i in expr_pid.columns]

print("pathways:", pwy_bin.shape[0], "genes:", pwy_bin.shape[1], "patients:", expr_pid.shape[1])
print("train keep:", len(train_ids), "val keep:", len(val_ids))

# 5) 用 TRAIN 做 gene-wise z-score
train_mat = expr_pid[train_ids].values
mu = train_mat.mean(axis=1, keepdims=True)
sd = train_mat.std(axis=1, keepdims=True) + 1e-8

expr_z = (expr_pid.values - mu) / sd
expr_z = pd.DataFrame(expr_z, index=expr_pid.index, columns=expr_pid.columns)

# 6) 保存 names
(out_root / "pathway_names.json").write_text(
    json.dumps(list(pwy_bin.index), ensure_ascii=False, indent=2)
)
(out_root / "gene_names.json").write_text(
    json.dumps(gene_order, ensure_ascii=False, indent=2)
)

M = pwy_bin.values.astype(np.float32)  # (P, G)


def dump(pids, subdir):
    for pid in pids:
        g = expr_z[pid].values.astype(np.float32)   # (G,)
        mat = M * g[None, :]                        # (P, G)
        np.save(out_root / subdir / f"{pid}.npy", mat.astype(np.float16))


dump(train_ids, "train")
dump(val_ids, "val")

print("OK wrote pathway×gene matrices to:", out_root)
