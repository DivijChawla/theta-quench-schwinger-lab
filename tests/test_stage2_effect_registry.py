from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_stage2_effect_registry_and_claim_gate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]

    summary = {
        "universal_sign_ok": False,
        "universal_beta_ok": True,
        "pinsker_ok": True,
        "universal_beta_claim": True,
        "universal_corridor_claim": False,
        "expressive_sign_ok": True,
        "expressive_beta_ok": True,
        "universal_expressive_beta_claim": True,
        "universal_expressive_claim": True,
    }
    all_rob = {"all_quantiles_pass": False, "all_quantiles_beta_pass": True, "all_quantiles_sign_pass": False}
    exp_rob = {"all_quantiles_pass": True, "all_quantiles_beta_pass": True, "all_quantiles_sign_pass": True}
    mechanism = {"all_arch_beta_positive": True, "records": []}

    summary_path = tmp_path / "summary.json"
    all_path = tmp_path / "all.json"
    exp_path = tmp_path / "exp.json"
    mechanism_path = tmp_path / "mechanism.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    all_path.write_text(json.dumps(all_rob), encoding="utf-8")
    exp_path.write_text(json.dumps(exp_rob), encoding="utf-8")
    mechanism_path.write_text(json.dumps(mechanism), encoding="utf-8")

    pooled = pd.DataFrame(
        [
            {
                "model_family": "schwinger",
                "architecture": "gru",
                "pooled_r_partial": 0.6,
                "ci_low_partial": 0.2,
                "ci_high_partial": 0.8,
                "n_conditions": 10,
            }
        ]
    )
    beta = pd.DataFrame(
        [
            {
                "model_family": "schwinger",
                "architecture": "gru",
                "beta_magic": 0.9,
                "beta_magic_ci_low": 0.4,
                "beta_magic_ci_high": 1.2,
                "positive_lower_bound": True,
                "n_snapshots": 100,
            }
        ]
    )
    pooled_path = tmp_path / "pooled.csv"
    beta_path = tmp_path / "beta.csv"
    pooled.to_csv(pooled_path, index=False)
    beta.to_csv(beta_path, index=False)

    out_registry = tmp_path / "effect_registry.json"
    cmd_registry = [
        sys.executable,
        "experiments/build_stage2_effect_registry.py",
        "--summary",
        str(summary_path),
        "--all-arch-robustness",
        str(all_path),
        "--expressive-robustness",
        str(exp_path),
        "--pooled",
        str(pooled_path),
        "--beta",
        str(beta_path),
        "--mechanism",
        str(mechanism_path),
        "--out",
        str(out_registry),
    ]
    subprocess.run(cmd_registry, check=True, cwd=root)

    payload = json.loads(out_registry.read_text(encoding="utf-8"))
    assert payload["gate_flags"]["all_arch_universal_corridor_claim"] is False
    assert payload["gate_flags"]["expressive_universal_corridor_claim"] is True
    assert payload["gate_flags"]["all_arch_beta_claim"] is True
    assert payload["interpretation"]["all_architecture_universal_beta_law"] == "supported"
    assert payload["interpretation"]["all_architecture_universal_corridor_sign_law"] == "unsupported"
    assert payload["interpretation"]["expressive_universal_beta_law"] == "supported"
    assert payload["interpretation"]["expressive_universal_corridor_sign_law"] == "supported"

    target = tmp_path / "target.md"
    target.write_text("all-architecture universal law", encoding="utf-8")
    cmd_gate = [
        sys.executable,
        "experiments/stage2_claim_gate.py",
        "--registry",
        str(out_registry),
        "--targets",
        str(target),
    ]
    completed = subprocess.run(cmd_gate, cwd=root)
    assert completed.returncode == 2
