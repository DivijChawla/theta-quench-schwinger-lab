from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent audit for stage-2 universal scan artifacts")
    parser.add_argument("--in-dir", type=str, default="outputs/stage2_prlx/universal_scan_v2")
    parser.add_argument("--registry", type=str, default="outputs/stage2_prlx/stage2/effect_registry.json")
    parser.add_argument("--out", type=str, default="outputs/stage2_prlx/stage2/stage2_audit.json")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    summary = json.loads((in_dir / "universal_law_summary.json").read_text(encoding="utf-8"))
    pooled = pd.read_csv(in_dir / "cross_model_pooled_effects.csv")
    beta = pd.read_csv(in_dir / "cross_model_beta_bounds.csv")
    snap = pd.read_csv(in_dir / "corridor_snapshot_all.csv")
    rob_all = json.loads((in_dir / "corridor_robustness_summary_all_arch.json").read_text(encoding="utf-8"))
    rob_expr = json.loads((in_dir / "corridor_robustness_summary_expressive.json").read_text(encoding="utf-8"))
    mech = json.loads((in_dir / "stage2_mechanism_summary.json").read_text(encoding="utf-8"))

    sign_ok = bool(np.all(np.isfinite(pooled["ci_low_partial"].to_numpy(dtype=float))) and np.all(pooled["ci_low_partial"].to_numpy(dtype=float) > 0.0))
    beta_ok = bool(np.all(beta["positive_lower_bound"].to_numpy(dtype=bool)))
    pinsker_ok = bool(np.all(snap["final_kl"].to_numpy(dtype=float) + 1e-12 >= 2.0 * (snap["final_tv"].to_numpy(dtype=float) ** 2)))
    beta_claim = bool(beta_ok and pinsker_ok)
    corridor_claim = bool(sign_ok and beta_ok and pinsker_ok)

    expr_arch = {"gru", "made"}
    pooled_expr = pooled[pooled["architecture"].isin(expr_arch)].copy()
    beta_expr = beta[beta["architecture"].isin(expr_arch)].copy()
    expr_sign_ok = bool(np.all(np.isfinite(pooled_expr["ci_low_partial"].to_numpy(dtype=float))) and np.all(pooled_expr["ci_low_partial"].to_numpy(dtype=float) > 0.0))
    expr_beta_ok = bool(np.all(beta_expr["positive_lower_bound"].to_numpy(dtype=bool)))
    expr_beta_claim = bool(expr_beta_ok and pinsker_ok)
    expr_corridor_claim = bool(expr_sign_ok and expr_beta_ok and pinsker_ok)

    gates = registry.get("gate_flags", {})
    interp = registry.get("interpretation", {})

    checks = [
        ("summary.universal_sign_ok", bool(summary.get("universal_sign_ok")) == sign_ok),
        ("summary.universal_beta_ok", bool(summary.get("universal_beta_ok")) == beta_ok),
        ("summary.pinsker_ok", bool(summary.get("pinsker_ok")) == pinsker_ok),
        ("summary.universal_beta_claim", bool(summary.get("universal_beta_claim")) == beta_claim),
        ("summary.universal_corridor_claim", bool(summary.get("universal_corridor_claim")) == corridor_claim),
        ("summary.expressive_sign_ok", bool(summary.get("expressive_sign_ok")) == expr_sign_ok),
        ("summary.expressive_beta_ok", bool(summary.get("expressive_beta_ok")) == expr_beta_ok),
        ("summary.universal_expressive_beta_claim", bool(summary.get("universal_expressive_beta_claim")) == expr_beta_claim),
        ("summary.universal_expressive_claim", bool(summary.get("universal_expressive_claim")) == expr_corridor_claim),
        ("registry.all_arch_beta_claim", bool(gates.get("all_arch_beta_claim")) == beta_claim),
        ("registry.all_arch_corridor_claim", bool(gates.get("all_arch_universal_corridor_claim")) == corridor_claim),
        ("registry.expressive_beta_claim", bool(gates.get("expressive_beta_claim")) == expr_beta_claim),
        ("registry.expressive_corridor_claim", bool(gates.get("expressive_universal_corridor_claim")) == expr_corridor_claim),
        ("registry.all_arch_beta_quantile_robust", bool(gates.get("all_arch_beta_quantile_robust")) == bool(rob_all.get("all_quantiles_beta_pass"))),
        ("registry.expressive_beta_quantile_robust", bool(gates.get("expressive_beta_quantile_robust")) == bool(rob_expr.get("all_quantiles_beta_pass"))),
        ("registry.all_arch_mechanism_beta_positive", bool(gates.get("all_arch_mechanism_beta_positive")) == bool(mech.get("all_arch_beta_positive"))),
        (
            "interpretation.all_architecture_universal_beta_law",
            interp.get("all_architecture_universal_beta_law") == ("supported" if (beta_claim and bool(rob_all.get("all_quantiles_beta_pass")) and bool(mech.get("all_arch_beta_positive"))) else "unsupported"),
        ),
        (
            "interpretation.all_architecture_universal_corridor_sign_law",
            interp.get("all_architecture_universal_corridor_sign_law") == ("supported" if (corridor_claim and bool(rob_all.get("all_quantiles_pass"))) else "unsupported"),
        ),
    ]

    payload = {
        "in_dir": str(in_dir),
        "registry": args.registry,
        "pass_count": int(sum(1 for _, ok in checks if ok)),
        "total_checks": int(len(checks)),
        "all_pass": bool(all(ok for _, ok in checks)),
        "checks": [{"name": name, "pass": ok} for name, ok in checks],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")

    if not payload["all_pass"]:
        for row in payload["checks"]:
            if not row["pass"]:
                print(f"FAIL: {row['name']}")
        raise SystemExit(2)
    print(f"PASS: {payload['pass_count']}/{payload['total_checks']} checks")


if __name__ == "__main__":
    main()
