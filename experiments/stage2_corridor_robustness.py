from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fisher_meta(corr: np.ndarray, n_obs: np.ndarray) -> tuple[float, float, float, int]:
    mask = np.isfinite(corr) & np.isfinite(n_obs) & (n_obs > 3)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan"), 0
    c = np.clip(corr[mask], -0.999999, 0.999999)
    w = n_obs[mask] - 3.0
    z = np.arctanh(c)
    z_bar = float(np.sum(w * z) / np.sum(w))
    se = math.sqrt(1.0 / float(np.sum(w)))
    z_lo = z_bar - 1.96 * se
    z_hi = z_bar + 1.96 * se
    return float(np.tanh(z_bar)), float(np.tanh(z_lo)), float(np.tanh(z_hi)), int(mask.sum())


def _bootstrap_beta_magic(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    x = df["magic_m2"].to_numpy(dtype=float)
    s = df["snapshot_entropy"].to_numpy(dtype=float)
    y = df["final_val_nll"].to_numpy(dtype=float)
    if len(df) < 8:
        return float("nan"), float("nan"), float("nan")
    a = np.column_stack([np.ones(len(df)), x, s])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    point = float(beta[1])
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aa = a[idx]
        yy = y[idx]
        b, *_ = np.linalg.lstsq(aa, yy, rcond=None)
        vals.append(float(b[1]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return point, float(lo), float(hi)


def _corridor_filter(cond_df: pd.DataFrame, lambda_quantile: float) -> pd.DataFrame:
    out = cond_df.copy()
    out = out[np.isfinite(out["pearson_magic_vs_val_nll"])].copy()
    out = out[np.isfinite(out["pearson_magic_vs_val_nll_partial_entropy"])].copy()
    out = out[np.isfinite(out["max_lambda"])].copy()
    out = out[np.abs(out["quench_delta"]) > 1e-12].copy()
    keep_idx: list[int] = []
    for _, sub in out.groupby("model_family"):
        thresh = float(sub["max_lambda"].quantile(lambda_quantile))
        keep_idx.extend(sub.index[sub["max_lambda"] >= thresh].tolist())
    return out.loc[sorted(keep_idx)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness sweep over dynamical-corridor quantile")
    parser.add_argument("--in-dir", type=str, default="outputs/stage2_prlx/universal_scan_quick")
    parser.add_argument("--quantiles", type=str, default="0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--architectures", type=str, default="")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4321)
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    cond = pd.read_csv(in_dir / "all_condition_stats.csv")
    snap = pd.read_csv(in_dir / "all_snapshot_all.csv")
    arch_filter = [x.strip() for x in args.architectures.split(",") if x.strip()]
    if arch_filter:
        cond = cond[cond["architecture"].isin(arch_filter)].copy()
        snap = snap[snap["architecture"].isin(arch_filter)].copy()
    quantiles = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]

    rows: list[dict[str, float | bool]] = []
    for idx, q in enumerate(quantiles):
        corridor = _corridor_filter(cond, lambda_quantile=q)
        ids = set(corridor["condition_id"].tolist())
        snap_q = snap[snap["condition_id"].isin(ids)].copy().reset_index(drop=True)

        pooled_lows: list[float] = []
        for (_, _), sub in corridor.groupby(["model_family", "architecture"]):
            _, lo, _, _ = _fisher_meta(
                sub["pearson_magic_vs_val_nll_partial_entropy"].to_numpy(dtype=float),
                sub["n_snapshots"].to_numpy(dtype=float),
            )
            if np.isfinite(lo):
                pooled_lows.append(float(lo))

        beta_lows: list[float] = []
        for (_, _), sub in snap_q.groupby(["model_family", "architecture"]):
            _, lo, _ = _bootstrap_beta_magic(
                sub,
                n_boot=args.bootstrap,
                seed=args.seed + idx * 13 + len(beta_lows),
            )
            if np.isfinite(lo):
                beta_lows.append(float(lo))

        pinsker_ok = bool(np.all(snap_q["final_kl"].to_numpy(dtype=float) + 1e-12 >= 2.0 * (snap_q["final_tv"].to_numpy(dtype=float) ** 2)))
        sign_ok = bool(len(pooled_lows) > 0 and min(pooled_lows) > 0.0)
        beta_ok = bool(len(beta_lows) > 0 and min(beta_lows) > 0.0)
        rows.append(
            {
                "lambda_quantile": q,
                "n_conditions": int(len(corridor)),
                "n_snapshots": int(len(snap_q)),
                "min_pooled_ci_low": float(min(pooled_lows)) if pooled_lows else float("nan"),
                "min_beta_ci_low": float(min(beta_lows)) if beta_lows else float("nan"),
                "sign_ok": sign_ok,
                "beta_ok": beta_ok,
                "pinsker_ok": pinsker_ok,
                "beta_only_claim": bool(beta_ok and pinsker_ok),
                "universal_corridor_claim": bool(sign_ok and beta_ok and pinsker_ok),
            }
        )

    out_df = pd.DataFrame(rows).sort_values("lambda_quantile").reset_index(drop=True)
    out_df.to_csv(in_dir / "corridor_robustness.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(out_df["lambda_quantile"], out_df["min_pooled_ci_low"], marker="o", label="min pooled CI low")
    ax.plot(out_df["lambda_quantile"], out_df["min_beta_ci_low"], marker="o", label="min beta CI low")
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_xlabel("lambda quantile")
    ax.set_ylabel("Lower-bound value")
    ax.set_title("Corridor-threshold robustness")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(in_dir / "figs" / "fig_stage2_corridor_robustness.png", dpi=220)
    plt.close(fig)

    summary = {
        "in_dir": str(in_dir),
        "architectures": arch_filter if arch_filter else "all",
        "all_quantiles_beta_pass": bool(np.all(out_df["beta_only_claim"].to_numpy(dtype=bool))),
        "all_quantiles_sign_pass": bool(np.all(out_df["sign_ok"].to_numpy(dtype=bool))),
        "all_quantiles_pass": bool(np.all(out_df["universal_corridor_claim"].to_numpy(dtype=bool))),
        "rows": out_df.to_dict(orient="records"),
    }
    with (in_dir / "corridor_robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {in_dir / 'corridor_robustness.csv'}")
    print(f"Wrote {in_dir / 'corridor_robustness_summary.json'}")


if __name__ == "__main__":
    main()
