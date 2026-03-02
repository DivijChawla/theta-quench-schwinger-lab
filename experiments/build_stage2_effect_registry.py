from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage-2 effect registry from universal scan outputs")
    parser.add_argument("--study-id", type=str, default="stage2_prlx")
    parser.add_argument("--phase", type=str, default="stage2")
    parser.add_argument(
        "--summary",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/universal_law_summary.json",
    )
    parser.add_argument(
        "--all-arch-robustness",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_all_arch.json",
    )
    parser.add_argument(
        "--expressive-robustness",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_expressive.json",
    )
    parser.add_argument(
        "--pooled",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/cross_model_pooled_effects.csv",
    )
    parser.add_argument(
        "--beta",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/cross_model_beta_bounds.csv",
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/stage2_mechanism_summary.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/stage2_prlx/stage2/effect_registry.json",
    )
    args = parser.parse_args()

    summary = _load(Path(args.summary))
    robust_all = _load(Path(args.all_arch_robustness))
    robust_expr = _load(Path(args.expressive_robustness))
    mechanism = _load(Path(args.mechanism))

    import pandas as pd

    pooled_df = pd.read_csv(args.pooled)
    beta_df = pd.read_csv(args.beta)

    records = []
    for _, r in pooled_df.iterrows():
        records.append(
            {
                "model_family": str(r["model_family"]),
                "architecture": str(r["architecture"]),
                "metric": "primary_partial_corr",
                "pooled_r": float(r["pooled_r_partial"]),
                "ci_low": float(r["ci_low_partial"]),
                "ci_high": float(r["ci_high_partial"]),
                "n_conditions": int(r["n_conditions"]),
            }
        )
    for _, r in beta_df.iterrows():
        records.append(
            {
                "model_family": str(r["model_family"]),
                "architecture": str(r["architecture"]),
                "metric": "beta_magic_entropy_controlled",
                "beta": float(r["beta_magic"]),
                "ci_low": float(r["beta_magic_ci_low"]),
                "ci_high": float(r["beta_magic_ci_high"]),
                "positive_lower_bound": bool(r["positive_lower_bound"]),
                "n_snapshots": int(r["n_snapshots"]),
            }
        )

    gates = {
        "all_arch_sign_ok": bool(summary.get("universal_sign_ok", False)),
        "all_arch_beta_ok": bool(summary.get("universal_beta_ok", False)),
        "all_arch_pinsker_ok": bool(summary.get("pinsker_ok", False)),
        "all_arch_beta_claim": bool(summary.get("universal_beta_claim", False)),
        "all_arch_universal_corridor_claim": bool(summary.get("universal_corridor_claim", False)),
        "expressive_sign_ok": bool(summary.get("expressive_sign_ok", False)),
        "expressive_beta_ok": bool(summary.get("expressive_beta_ok", False)),
        "expressive_beta_claim": bool(summary.get("universal_expressive_beta_claim", False)),
        "expressive_universal_corridor_claim": bool(summary.get("universal_expressive_claim", False)),
        "all_arch_quantile_robust": bool(robust_all.get("all_quantiles_pass", False)),
        "all_arch_beta_quantile_robust": bool(robust_all.get("all_quantiles_beta_pass", False)),
        "all_arch_sign_quantile_robust": bool(robust_all.get("all_quantiles_sign_pass", False)),
        "expressive_quantile_robust": bool(robust_expr.get("all_quantiles_pass", False)),
        "expressive_beta_quantile_robust": bool(robust_expr.get("all_quantiles_beta_pass", False)),
        "all_arch_mechanism_beta_positive": bool(mechanism.get("all_arch_beta_positive", False)),
    }

    all_arch_beta_law_supported = bool(
        gates["all_arch_beta_claim"]
        and gates["all_arch_beta_quantile_robust"]
        and gates["all_arch_mechanism_beta_positive"]
    )
    expressive_beta_law_supported = bool(gates["expressive_beta_claim"] and gates["expressive_beta_quantile_robust"])
    all_arch_corridor_supported = bool(gates["all_arch_universal_corridor_claim"] and gates["all_arch_quantile_robust"])
    expressive_corridor_supported = bool(gates["expressive_universal_corridor_claim"] and gates["expressive_quantile_robust"])

    payload = {
        "schema_version": "1.0",
        "study_id": args.study_id,
        "phase": args.phase,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_summary": args.summary,
        "source_robustness_all_arch": args.all_arch_robustness,
        "source_robustness_expressive": args.expressive_robustness,
        "source_mechanism": args.mechanism,
        "records": records,
        "gate_flags": gates,
        "interpretation": {
            "all_architecture_universal_beta_law": "supported"
            if all_arch_beta_law_supported
            else "unsupported",
            "all_architecture_universal_corridor_sign_law": "supported"
            if all_arch_corridor_supported
            else "unsupported",
            "expressive_universal_beta_law": "supported" if expressive_beta_law_supported else "unsupported",
            "expressive_universal_corridor_sign_law": "supported" if expressive_corridor_supported else "unsupported",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
