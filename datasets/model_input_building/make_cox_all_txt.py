from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"

clinical_csv = DATASETS_DIR / "processed" / "EC_Clinic_PFS_287.csv"
out_dir = DATASETS_DIR / "model_inputs" / "pamt_287" / "cox"
out_dir.mkdir(parents=True, exist_ok=True)

out_txt = out_dir / "all.txt"

df = pd.read_csv(clinical_csv)
df = df[["ID", "PFS.time", "PFS"]].copy()
df.columns = ["ID", "time", "event"]

df["ID"] = df["ID"].astype(str)
df["time"] = pd.to_numeric(df["time"], errors="coerce")
df["event"] = pd.to_numeric(df["event"], errors="coerce")

df = df.dropna(subset=["ID", "time", "event"]).copy()
df["event"] = df["event"].astype(int)

df.to_csv(out_txt, sep="\t", index=False, header=False)

print("OK wrote:", out_txt)
