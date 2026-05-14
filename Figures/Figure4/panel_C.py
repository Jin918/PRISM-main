#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURE_ID = 'Figure4'
PANEL_ID = 'C'
PANEL_DESCRIPTION = 'TCGA spline and forest summary'
ORIGIN_HINT = 'Figure-main/script/figure4_final_spline_forest_core.R'
NOTE = ''
BUILDER_KIND = 'rscript'
BUILDER_SCRIPT = REPO_ROOT / 'Figure-main/script/make_figure4.R'
BUILDER_ARGS = []
EXPECTED_OUTPUTS = [REPO_ROOT / relative_path for relative_path in [
    'Figure-main/Figure4/Figure4C_TCGA_spline_forest.pdf',
    'Figure-main/Figure4/Figure4C_TCGA_spline_forest.png',
]]


def resolve_runner(kind: str) -> str:
    if kind == "python":
        candidates = [
            os.environ.get("FINAL_TIF_PYTHON"),
            os.environ.get("SUPP_FIGURES_PYTHON"),
            os.environ.get("FIGURE5_PYTHON"),
            "/root/anaconda3/envs/NT/bin/python",
            sys.executable,
            "/root/anaconda3/bin/python",
            shutil.which("python3"),
            shutil.which("python"),
        ]
    elif kind == "rscript":
        candidates = [
            os.environ.get("RSCRIPT_BIN"),
            "/root/anaconda3/envs/NT/bin/Rscript",
            shutil.which("Rscript"),
        ]
    else:
        raise ValueError(f"Unsupported builder kind: {kind}")

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise RuntimeError(f"No usable runner found for builder kind: {kind}")


def main() -> None:
    if not BUILDER_SCRIPT.exists():
        raise FileNotFoundError(f"Missing builder script: {BUILDER_SCRIPT}")

    runner = resolve_runner(BUILDER_KIND)
    subprocess.run([runner, str(BUILDER_SCRIPT), *BUILDER_ARGS], check=True)

    missing = [str(path) for path in EXPECTED_OUTPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected panel outputs for "
            f"{FIGURE_ID} panel {PANEL_ID}: {', '.join(missing)}"
        )

    print(f"[OK] {FIGURE_ID} panel {PANEL_ID}: {PANEL_DESCRIPTION}")
    print(f"builder: {BUILDER_SCRIPT}")
    if ORIGIN_HINT:
        print(f"origin: {ORIGIN_HINT}")
    if NOTE:
        print(f"note: {NOTE}")
    for output in EXPECTED_OUTPUTS:
        print(f"output: {output}")


if __name__ == "__main__":
    main()
