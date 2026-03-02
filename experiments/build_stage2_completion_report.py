from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stage2 completion markdown")
    parser.add_argument(
        "--registry",
        type=str,
        default="outputs/stage2_prlx/stage2/effect_registry.json",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/universal_law_summary.json",
    )
    parser.add_argument(
        "--all-robustness",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_all_arch.json",
    )
    parser.add_argument(
        "--expressive-robustness",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_expressive.json",
    )
    parser.add_argument("--out", type=str, default="report/stage2_completion.md")
    args = parser.parse_args()

    reg = _load(Path(args.registry))
    summ = _load(Path(args.summary))
    all_r = _load(Path(args.all_robustness))
    exp_r = _load(Path(args.expressive_robustness))
    gates = reg.get("gate_flags", {})

    lines = [
        "# Stage 2 Completion Status",
        "",
        "## Universal-Law Gates",
        f"- all_arch_beta_claim: `{gates.get('all_arch_beta_claim')}`",
        f"- all_arch_beta_quantile_robust: `{gates.get('all_arch_beta_quantile_robust')}`",
        f"- all_arch_mechanism_beta_positive: `{gates.get('all_arch_mechanism_beta_positive')}`",
        f"- all_arch_universal_corridor_claim: `{gates.get('all_arch_universal_corridor_claim')}`",
        f"- all_arch_quantile_robust: `{gates.get('all_arch_quantile_robust')}`",
        f"- expressive_beta_claim: `{gates.get('expressive_beta_claim')}`",
        f"- expressive_beta_quantile_robust: `{gates.get('expressive_beta_quantile_robust')}`",
        f"- expressive_universal_corridor_claim: `{gates.get('expressive_universal_corridor_claim')}`",
        f"- expressive_quantile_robust: `{gates.get('expressive_quantile_robust')}`",
        "",
        "## Final Stage 2 Claim",
        f"- all-architecture universal beta-law: `{reg.get('interpretation', {}).get('all_architecture_universal_beta_law')}`",
        f"- all-architecture universal corridor sign-law: `{reg.get('interpretation', {}).get('all_architecture_universal_corridor_sign_law')}`",
        f"- expressive-model universal beta-law (GRU/MADE): `{reg.get('interpretation', {}).get('expressive_universal_beta_law')}`",
        f"- expressive-model universal corridor sign-law (GRU/MADE): `{reg.get('interpretation', {}).get('expressive_universal_corridor_sign_law')}`",
        "",
        "## Dataset Scale",
        f"- total conditions: `{summ.get('n_conditions_all')}`",
        f"- corridor conditions: `{summ.get('n_conditions_corridor')}`",
        f"- corridor snapshots: `{summ.get('n_snapshots_corridor')}`",
        "",
        "## Robustness Sweeps",
        f"- all architectures quantile sweep pass (sign+beta): `{all_r.get('all_quantiles_pass')}`",
        f"- all architectures quantile sweep pass (beta-only): `{all_r.get('all_quantiles_beta_pass')}`",
        f"- expressive-only quantile sweep pass (sign+beta): `{exp_r.get('all_quantiles_pass')}`",
        f"- expressive-only quantile sweep pass (beta-only): `{exp_r.get('all_quantiles_beta_pass')}`",
        "",
        "## Key Artifacts",
        "- `outputs/stage2_prlx/universal_scan_v2/cross_model_pooled_effects.csv`",
        "- `outputs/stage2_prlx/universal_scan_v2/cross_model_beta_bounds.csv`",
        "- `outputs/stage2_prlx/universal_scan_v2/finite_size_extrapolation.csv`",
        "- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_crossmodel_pooled_r.png`",
        "- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_beta_bounds.png`",
        "- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_finite_size.png`",
        "- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_pinsker_bound.png`",
        "- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_corridor_robustness_all_arch.png`",
        "- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_corridor_robustness_expressive.png`",
        "- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_mechanism_arch_slope.png`",
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
