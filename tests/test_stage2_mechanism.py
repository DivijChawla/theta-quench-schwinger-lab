from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_stage2_mechanism_regression_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    in_dir = tmp_path / "scan"
    in_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(5)
    rows = []
    archs = ["gru", "made", "independent"]
    for arch_idx, arch in enumerate(archs):
        for cond in range(6):
            cid = f"{arch}_c{cond}"
            for snap_idx in range(8):
                magic = 0.8 + 0.12 * snap_idx + 0.03 * arch_idx
                entropy = 0.2 + 0.01 * snap_idx
                noise = rng.normal(0.0, 0.01)
                val_nll = 0.5 + 0.9 * magic + 0.4 * entropy + noise
                rows.append(
                    {
                        "condition_id": cid,
                        "model_family": "schwinger" if cond % 2 == 0 else "tfim",
                        "architecture": arch,
                        "snapshot_index": snap_idx,
                        "magic_m2": magic,
                        "snapshot_entropy": entropy,
                        "final_val_nll": val_nll,
                    }
                )
    pd.DataFrame(rows).to_csv(in_dir / "corridor_snapshot_all.csv", index=False)

    cmd = [sys.executable, "experiments/stage2_mechanism_regression.py", "--in-dir", str(in_dir)]
    subprocess.run(cmd, check=True, cwd=root)

    summary = json.loads((in_dir / "stage2_mechanism_summary.json").read_text(encoding="utf-8"))
    assert summary["all_arch_beta_positive"] is True
    out = pd.read_csv(in_dir / "stage2_mechanism_arch_slope.csv")
    assert set(out["architecture"]) == set(archs)
    assert np.all(out["ci_low"].to_numpy(dtype=float) > 0.0)
