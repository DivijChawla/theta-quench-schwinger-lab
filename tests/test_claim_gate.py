from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_claim_gate_script_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "experiments/claim_gate.py",
        "--publishability",
        "report/publishability_status.md",
        "--targets",
        "README.md,report/main.tex",
    ]
    subprocess.run(cmd, check=True, cwd=root)
