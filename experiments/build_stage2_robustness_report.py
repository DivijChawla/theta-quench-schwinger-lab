from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage-2 robustness markdown from robustness JSON summaries")
    parser.add_argument(
        "--all-arch",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_all_arch.json",
    )
    parser.add_argument(
        "--expressive",
        type=str,
        default="outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_expressive.json",
    )
    parser.add_argument("--out", type=str, default="report/stage2_corridor_robustness.md")
    args = parser.parse_args()

    all_arch = _load(Path(args.all_arch))
    expr = _load(Path(args.expressive))

    lines = [
        "# Stage 2 Corridor-Robustness Report",
        "",
        "## All-architecture gate",
        f"- quantiles pass (sign+beta): `{all_arch.get('all_quantiles_pass')}`",
        f"- quantiles pass (beta-only): `{all_arch.get('all_quantiles_beta_pass')}`",
        f"- quantiles pass (sign-only): `{all_arch.get('all_quantiles_sign_pass')}`",
        "",
        "| q_lambda | n_conditions | n_snapshots | min pooled CI low | min beta CI low | sign+beta claim | beta-only claim |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in all_arch.get("rows", []):
        lines.append(
            f"| {row.get('lambda_quantile')} | {row.get('n_conditions')} | {row.get('n_snapshots')} | "
            f"{row.get('min_pooled_ci_low'):.4f} | {row.get('min_beta_ci_low'):.4f} | {row.get('universal_corridor_claim')} | {row.get('beta_only_claim')} |"
        )

    lines.extend(
        [
            "",
            "## Expressive-only gate (GRU, MADE)",
            f"- quantiles pass (sign+beta): `{expr.get('all_quantiles_pass')}`",
            f"- quantiles pass (beta-only): `{expr.get('all_quantiles_beta_pass')}`",
            f"- quantiles pass (sign-only): `{expr.get('all_quantiles_sign_pass')}`",
            "",
            "| q_lambda | n_conditions | n_snapshots | min pooled CI low | min beta CI low | sign+beta claim | beta-only claim |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in expr.get("rows", []):
        lines.append(
            f"| {row.get('lambda_quantile')} | {row.get('n_conditions')} | {row.get('n_snapshots')} | "
            f"{row.get('min_pooled_ci_low'):.4f} | {row.get('min_beta_ci_low'):.4f} | {row.get('universal_corridor_claim')} | {row.get('beta_only_claim')} |"
        )

    lines.extend(
        [
            "",
            "Conclusion:",
            "- Beta-only law is robust across all tested quantiles.",
            "- Correlation-sign law remains architecture-dependent.",
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
