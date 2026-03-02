from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_registered_study_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "experiments/run_registered_study.py",
        "--study",
        "stage1_prxq",
        "--phase",
        "stage1",
        "--dry-run",
    ]
    subprocess.run(cmd, check=True, cwd=root)


def test_registered_stage2_study_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "experiments/run_registered_study.py",
        "--study",
        "stage2_prlx",
        "--phase",
        "stage2",
        "--dry-run",
    ]
    subprocess.run(cmd, check=True, cwd=root)


def test_registered_stage3_study_dry_run() -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "experiments/run_registered_study.py",
        "--study",
        "stage3_prx",
        "--phase",
        "stage3",
        "--dry-run",
    ]
    subprocess.run(cmd, check=True, cwd=root)
