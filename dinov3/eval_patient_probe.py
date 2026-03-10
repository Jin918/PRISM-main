import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

TCGA_RE = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", re.I)


def path_to_pid(p: str) -> str:
    m = TCGA_RE.search(p)
    if m:
        return m.group(1).upper()
    parent = os.path.basename(os.path.dirname(p))
    return parent[:12]


def load_patch_feats_and_aggregate(feat_pth: str, paths_json: str, save_npz: str = ""):
    feats = torch.load(feat_pth, map_location="cpu")
    if isinstance(feats, torch.Tensor):
        feats = feats.float().numpy()
    feats = np.asarray(feats, dtype=np.float32)
    N, D = feats.shape
    print(f"[load] feats: {N} x {D} from {feat_pth}")

    with open(paths_json, "r") as f:
        paths = json.load(f)
    if len(paths) != N:
        raise ValueError(f"paths_json length={len(paths)} != feats rows={N}")

    pids = [path_to_pid(p) for p in paths]
    uniq, codes = np.unique(np.array(pids, dtype=object), return_inverse=True)
    K = len(uniq)
    print(f"[aggregate] unique patients: {K}")

    sums = np.zeros((K, D), dtype=np.float64)
    cnts = np.zeros((K,), dtype=np.int64)
    np.add.at(sums, codes, feats)
    np.add.at(cnts, codes, 1)
    patient_feats = (sums / np.maximum(cnts[:, None], 1)).astype(np.float32)

    if save_npz:
        os.makedirs(os.path.dirname(save_npz), exist_ok=True)
        np.savez_compressed(save_npz, ids=uniq, feats=patient_feats, counts=cnts)
        print(f"[save] patient feats -> {save_npz}")

    id2i = {pid: i for i, pid in enumerate(uniq.astype(str))}
    return uniq.astype(str), patient_feats, id2i


def load_patient_npz(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    ids = z["ids"].astype(str)
    feats = z["feats"].astype(np.float32)
    id2i = {pid: i for i, pid in enumerate(ids)}
    print(f"[load] patient_npz: {len(ids)} patients, feat_dim={feats.shape[1]} from {npz_path}")
    return ids, feats, id2i


def align_X(df: pd.DataFrame, id2i: dict, feats: np.ndarray, id_col: str):
    keep = []
    X = []
    for pid in df[id_col].astype(str).tolist():
        if pid in id2i:
            keep.append(True)
            X.append(feats[id2i[pid]])
        else:
            keep.append(False)
    keep = np.array(keep, dtype=bool)
    X = np.stack(X, axis=0) if len(X) else None
    return X, keep


def build_lr_binary():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])


def build_lr_multiclass():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])


def _binary_metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    out = {
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan,
        "bal_acc": float(balanced_accuracy_score(y_true, pred)),
        "acc": float(accuracy_score(y_true, pred)),
        "n": int(len(y_true)),
        "pos": int(np.sum(y_true == 1)),
        "neg": int(np.sum(y_true == 0)),
    }
    return out


def _multiclass_metrics(y_true, pred):
    out = {
        "bal_acc": float(balanced_accuracy_score(y_true, pred)),
        "acc": float(accuracy_score(y_true, pred)),
        "n": int(len(y_true)),
    }
    return out


def eval_holdout_binary(train_df, val_df, X_train, X_val, col):
    y_tr = pd.to_numeric(train_df[col], errors="coerce")
    y_va = pd.to_numeric(val_df[col], errors="coerce")
    m_tr = ~y_tr.isna()
    m_va = ~y_va.isna()

    if m_tr.sum() < 10 or m_va.sum() < 10:
        return None

    y_tr = y_tr[m_tr].astype(int).values
    y_va = y_va[m_va].astype(int).values
    Xt = X_train[m_tr.values]
    Xv = X_val[m_va.values]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return None

    clf = build_lr_binary()
    clf.fit(Xt, y_tr)

    prob_tr = clf.predict_proba(Xt)[:, 1]
    prob_va = clf.predict_proba(Xv)[:, 1]

    trm = _binary_metrics(y_tr, prob_tr)
    vam = _binary_metrics(y_va, prob_va)

    return {
        "task": col, "type": "binary",
        "train_auc": trm["auc"], "train_bal_acc": trm["bal_acc"], "train_acc": trm["acc"],
        "train_n": trm["n"], "train_pos": trm["pos"], "train_neg": trm["neg"],
        "val_auc": vam["auc"], "val_bal_acc": vam["bal_acc"], "val_acc": vam["acc"],
        "val_n": vam["n"], "val_pos": vam["pos"], "val_neg": vam["neg"],
    }


def eval_holdout_multiclass(train_df, val_df, X_train, X_val, col):
    y_tr = train_df[col].astype(str)
    y_va = val_df[col].astype(str)

    m_tr = ~(y_tr.isna() | (y_tr == "Unknown"))
    m_va = ~(y_va.isna() | (y_va == "Unknown"))
    if m_tr.sum() < 10 or m_va.sum() < 10:
        return None

    y_tr = y_tr[m_tr].values
    y_va = y_va[m_va].values
    Xt = X_train[m_tr.values]
    Xv = X_val[m_va.values]

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)

    known = set(le.classes_)
    keep = np.array([c in known for c in y_va], dtype=bool)
    if keep.sum() < 10:
        return None
    y_va = y_va[keep]
    Xv = Xv[keep]
    y_va_enc = le.transform(y_va)

    if len(np.unique(y_tr_enc)) < 2 or len(np.unique(y_va_enc)) < 2:
        return None

    clf = build_lr_multiclass()
    clf.fit(Xt, y_tr_enc)

    pred_tr = clf.predict(Xt)
    pred_va = clf.predict(Xv)

    trm = _multiclass_metrics(y_tr_enc, pred_tr)
    vam = _multiclass_metrics(y_va_enc, pred_va)

    return {
        "task": col, "type": "multiclass", "n_cls": int(len(le.classes_)),
        "train_bal_acc": trm["bal_acc"], "train_acc": trm["acc"], "train_n": trm["n"],
        "val_bal_acc": vam["bal_acc"], "val_acc": vam["acc"], "val_n": vam["n"],
    }


def eval_cv_binary(df, X, col, n_splits=5, seed=42):
    y = pd.to_numeric(df[col], errors="coerce")
    m = ~y.isna()
    if m.sum() < 30:
        return None
    y = y[m].astype(int).values
    Xc = X[m.values]

    if len(np.unique(y)) < 2:
        return None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, balaccs = [], []
    for tr, te in skf.split(Xc, y):
        clf = build_lr_binary()
        clf.fit(Xc[tr], y[tr])
        prob = clf.predict_proba(Xc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        aucs.append(roc_auc_score(y[te], prob))
        balaccs.append(balanced_accuracy_score(y[te], pred))

    return {
        "task": col, "type": "binary_cv", "n": int(len(y)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "bal_acc_mean": float(np.mean(balaccs)), "bal_acc_std": float(np.std(balaccs))
    }


def main():
    ap = argparse.ArgumentParser("Patient-level linear probe evaluation (holdout or CV)")

    ap.add_argument("--patient_npz", default="", type=str)
    ap.add_argument("--feat_pth", default="", type=str)
    ap.add_argument("--paths_json", default="", type=str)
    ap.add_argument("--save_patient_npz", default="", type=str)

    ap.add_argument("--train_csv", default="", type=str)
    ap.add_argument("--val_csv", default="", type=str)
    ap.add_argument("--csv", default="", type=str)
    ap.add_argument("--id_col", default="ID", type=str)

    ap.add_argument("--binary_cols", default="y12,y24,y36,y60,mi_50,Radiation_Therapy,Chemotherapy,Residual_Tumor", type=str)
    ap.add_argument("--multiclass_cols", default="Histologic_Grade,Subtype", type=str)

    ap.add_argument("--mode", default="holdout", choices=["holdout", "cv"])
    ap.add_argument("--cv_folds", default=5, type=int)

    ap.add_argument("--out_dir", default="./eval_out", type=str)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.patient_npz:
        ids, feats, id2i = load_patient_npz(args.patient_npz)
    else:
        if not (args.feat_pth and args.paths_json):
            raise ValueError("Provide either --patient_npz OR (--feat_pth AND --paths_json).")
        ids, feats, id2i = load_patch_feats_and_aggregate(args.feat_pth, args.paths_json, args.save_patient_npz)

    results = []
    binary_cols = [c.strip() for c in args.binary_cols.split(",") if c.strip()]
    multiclass_cols = [c.strip() for c in args.multiclass_cols.split(",") if c.strip()]

    if args.mode == "holdout":
        if not (args.train_csv and args.val_csv):
            raise ValueError("holdout mode requires --train_csv and --val_csv")

        df_tr = pd.read_csv(args.train_csv)
        df_va = pd.read_csv(args.val_csv)

        X_train, keep_tr = align_X(df_tr, id2i, feats, args.id_col)
        X_val, keep_va = align_X(df_va, id2i, feats, args.id_col)
        df_tr = df_tr[keep_tr].reset_index(drop=True)
        df_va = df_va[keep_va].reset_index(drop=True)

        print(f"[align] train matched={len(df_tr)} | val matched={len(df_va)} | feat_dim={X_train.shape[1]}")

        print("\n=== Multi-class (Train + Val) ===")
        for col in multiclass_cols:
            if col not in df_tr.columns or col not in df_va.columns:
                print(f"{col}: column missing")
                continue
            r = eval_holdout_multiclass(df_tr, df_va, X_train, X_val, col)
            if r is None:
                print(f"{col:<22} | skipped")
            else:
                print(f"{col:<22} | "
                      f"train bal-acc={r['train_bal_acc']:.3f} acc={r['train_acc']:.3f} n={r['train_n']} || "
                      f"val bal-acc={r['val_bal_acc']:.3f} acc={r['val_acc']:.3f} n={r['val_n']} | n_cls={r['n_cls']}")
                results.append(r)

        print("\n=== Binary (Train + Val) ===")
        for col in binary_cols:
            if col not in df_tr.columns or col not in df_va.columns:
                print(f"{col}: column missing")
                continue
            r = eval_holdout_binary(df_tr, df_va, X_train, X_val, col)
            if r is None:
                print(f"{col:<22} | skipped (need >=2 classes & enough samples)")
            else:
                print(f"{col:<22} | "
                      f"train AUC={r['train_auc']:.3f} bal-acc={r['train_bal_acc']:.3f} acc={r['train_acc']:.3f} n={r['train_n']} || "
                      f"val AUC={r['val_auc']:.3f} bal-acc={r['val_bal_acc']:.3f} acc={r['val_acc']:.3f} n={r['val_n']}")
                results.append(r)

        out_csv = os.path.join(args.out_dir, "eval_results_holdout_train_val.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\n[save] {out_csv}")

    else:
        if not args.csv:
            raise ValueError("cv mode requires --csv")

        df = pd.read_csv(args.csv)
        X, keep = align_X(df, id2i, feats, args.id_col)
        df = df[keep].reset_index(drop=True)

        print(f"[align] matched={len(df)} | feat_dim={X.shape[1]}")

        for col in binary_cols:
            if col not in df.columns:
                continue
            r = eval_cv_binary(df, X, col, n_splits=args.cv_folds)
            if r is not None:
                results.append(r)
                print(f"{col:<22} | AUC={r['auc_mean']:.3f}±{r['auc_std']:.3f} | "
                      f"bal-acc={r['bal_acc_mean']:.3f}±{r['bal_acc_std']:.3f} | n={r['n']}")

        out_csv = os.path.join(args.out_dir, "eval_results_cv.csv")
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print(f"\n[save] {out_csv}")


if __name__ == "__main__":
    main()
