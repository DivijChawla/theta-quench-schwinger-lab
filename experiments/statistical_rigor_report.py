from __future__ import annotations

import argparse
import json
from math import atanh, ceil, tanh
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def _power_for_corr(n: int, r: float, alpha: float = 0.05) -> float:
    if n <= 3:
        return float("nan")
    z_alpha = norm.ppf(1 - alpha / 2.0)
    delta = abs(atanh(r)) * np.sqrt(n - 3)
    # Two-sided normal approximation in Fisher-z space.
    power = norm.cdf(-z_alpha - delta) + (1 - norm.cdf(z_alpha - delta))
    return float(power)


def _min_n_for_power(r: float, target_power: float, alpha: float = 0.05, n_max: int = 10_000) -> int:
    for n in range(4, n_max + 1):
        if _power_for_corr(n=n, r=r, alpha=alpha) >= target_power:
            return n
    return n_max


def _mixed_effects(snapshot_csv: Path) -> dict[str, float | str]:
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        return {"status": f"unavailable: {exc}"}

    df = pd.read_csv(snapshot_csv)
    needed = {"final_val_nll", "magic_m2", "snapshot_entropy", "condition_id", "architecture"}
    missing = needed - set(df.columns)
    if missing:
        return {"status": f"missing_columns: {sorted(missing)}"}

    df = df.dropna(subset=["final_val_nll", "magic_m2", "snapshot_entropy", "condition_id", "architecture"]).reset_index(
        drop=True
    )
    if df.empty:
        return {"status": "empty_after_dropna"}

    fit = smf.mixedlm(
        "final_val_nll ~ magic_m2 + snapshot_entropy + C(architecture)",
        data=df,
        groups=df["condition_id"],
    ).fit(reml=False)

    out: dict[str, float | str] = {"status": "ok"}
    for key in ["magic_m2", "snapshot_entropy"]:
        out[f"coef_{key}"] = float(fit.params.get(key, np.nan))
        out[f"p_{key}"] = float(fit.pvalues.get(key, np.nan))
    out["aic"] = float(fit.aic)
    out["bic"] = float(fit.bic)
    out["n_obs"] = int(df.shape[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build statistical rigor report")
    parser.add_argument("--snapshot-csv", type=str, default="outputs/novelty/nnqs_snapshot_all.csv")
    parser.add_argument("--condition-csv", type=str, default="outputs/novelty/nnqs_condition_stats.csv")
    parser.add_argument("--out-dir", type=str, default="outputs/statistical_rigor")
    parser.add_argument("--report", type=str, default="report/statistical_rigor.md")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cond = pd.read_csv(args.condition_csv)
    n_snap = int(round(cond["n_snapshots"].mean())) if "n_snapshots" in cond.columns and len(cond) > 0 else 8

    power_rows = []
    for r in [0.25, 0.30, 0.35]:
        power_rows.append(
            {
                "target_r": r,
                "power_at_mean_n_snapshots": _power_for_corr(n=n_snap, r=r, alpha=args.alpha),
                "min_n_for_target_power": _min_n_for_power(r=r, target_power=args.target_power, alpha=args.alpha),
            }
        )
    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(out_dir / "power_analysis.csv", index=False)

    mixed = _mixed_effects(Path(args.snapshot_csv))

    payload = {
        "alpha": args.alpha,
        "target_power": args.target_power,
        "mean_n_snapshots": n_snap,
        "power_analysis": power_df.to_dict(orient="records"),
        "mixed_effects": mixed,
    }
    with (out_dir / "statistical_rigor.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Statistical Rigor Report",
        "",
        f"- alpha: `{args.alpha}`",
        f"- target power: `{args.target_power}`",
        f"- mean snapshots per condition: `{n_snap}`",
        "",
        "## Power analysis",
    ]
    for row in power_df.to_dict(orient="records"):
        lines.append(
            f"- r={row['target_r']:.2f}: power@mean_n={row['power_at_mean_n_snapshots']:.3f}, "
            f"min_n_for_target_power={int(row['min_n_for_target_power'])}"
        )
    lines.extend(["", "## Mixed-effects model", f"- status: `{mixed.get('status')}`"])
    if mixed.get("status") == "ok":
        lines.append(f"- coef(magic_m2)={mixed.get('coef_magic_m2'):.4f}, p={mixed.get('p_magic_m2'):.3g}")
        lines.append(
            f"- coef(snapshot_entropy)={mixed.get('coef_snapshot_entropy'):.4f}, "
            f"p={mixed.get('p_snapshot_entropy'):.3g}"
        )
        lines.append(f"- AIC={mixed.get('aic'):.3f}, BIC={mixed.get('bic'):.3f}, n_obs={mixed.get('n_obs')}")
    lines.extend(
        [
            "",
            "Artifacts:",
            f"- `{out_dir / 'power_analysis.csv'}`",
            f"- `{out_dir / 'statistical_rigor.json'}`",
        ]
    )
    Path(args.report).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'power_analysis.csv'}")
    print(f"Wrote {out_dir / 'statistical_rigor.json'}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
