from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm


def _slope_and_ci(
    fit,
    arch: str,
    key_main: str = "magic_m2",
    key_prefix: str = "magic_m2:C(architecture)[T.",
) -> tuple[float, float, float, float]:
    slope = float(fit.params.get(key_main, 0.0))
    interaction_key = f"{key_prefix}{arch}]"
    cov = fit.cov_params()

    var = float(cov.loc[key_main, key_main]) if key_main in cov.index else 0.0
    if interaction_key in cov.index:
        slope += float(fit.params.get(interaction_key, 0.0))
        var += float(cov.loc[interaction_key, interaction_key] + 2.0 * cov.loc[key_main, interaction_key])
    se = float(np.sqrt(max(var, 0.0)))
    lo = float(slope - 1.96 * se)
    hi = float(slope + 1.96 * se)
    z = float(slope / se) if se > 0 else float("nan")
    p = float(2.0 * (1.0 - norm.cdf(abs(z)))) if np.isfinite(z) else float("nan")
    return slope, lo, hi, p


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-robust mechanism regression for stage-2 snapshots")
    parser.add_argument("--in-dir", type=str, default="outputs/stage2_prlx/universal_scan_v2")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    _ = args.seed
    in_dir = Path(args.in_dir)
    fig_dir = in_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_dir / "corridor_snapshot_all.csv")
    needed = ["final_val_nll", "magic_m2", "snapshot_entropy", "architecture", "condition_id"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=needed).copy()
    df = df[np.isfinite(df["final_val_nll"]) & np.isfinite(df["magic_m2"]) & np.isfinite(df["snapshot_entropy"])].copy()

    fit = smf.ols(
        "final_val_nll ~ magic_m2 + snapshot_entropy + C(model_family) + C(architecture) + magic_m2:C(architecture)",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["condition_id"]})

    archs = sorted(df["architecture"].unique().tolist())
    rows: list[dict[str, float | str | bool]] = []
    for arch in archs:
        slope, lo, hi, p = _slope_and_ci(fit, arch=arch)
        rows.append(
            {
                "architecture": arch,
                "beta_magic": slope,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": p,
                "positive_lower_bound": bool(np.isfinite(lo) and lo > 0.0),
            }
        )
    out_df = pd.DataFrame(rows).sort_values("architecture").reset_index(drop=True)
    out_df.to_csv(in_dir / "stage2_mechanism_arch_slope.csv", index=False)

    x = np.arange(len(out_df))
    y = out_df["beta_magic"].to_numpy(dtype=float)
    lo = out_df["ci_low"].to_numpy(dtype=float)
    hi = out_df["ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=4, color="#0f5a9c")
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(out_df["architecture"].tolist(), rotation=10)
    ax.set_ylabel("Cluster-robust beta_magic (95% CI)")
    ax.set_title("Stage-2 mechanism regression by architecture")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_stage2_mechanism_arch_slope.png", dpi=220)
    plt.close(fig)

    summary = {
        "in_dir": str(in_dir),
        "n_snapshots": int(len(df)),
        "formula": "final_val_nll ~ magic_m2 + snapshot_entropy + C(model_family) + C(architecture) + magic_m2:C(architecture)",
        "all_arch_beta_positive": bool(np.all(out_df["positive_lower_bound"].to_numpy(dtype=bool))),
        "records": out_df.to_dict(orient="records"),
    }
    with (in_dir / "stage2_mechanism_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {in_dir / 'stage2_mechanism_arch_slope.csv'}")
    print(f"Wrote {in_dir / 'stage2_mechanism_summary.json'}")


if __name__ == "__main__":
    main()
