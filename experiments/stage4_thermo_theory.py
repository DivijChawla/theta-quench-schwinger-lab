from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


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


def _weighted_line_fit(x: np.ndarray, y: np.ndarray, se: np.ndarray) -> tuple[float, float]:
    w = 1.0 / np.clip(se, 1e-9, None) ** 2
    xbar = float(np.sum(w * x) / np.sum(w))
    ybar = float(np.sum(w * y) / np.sum(w))
    denom = float(np.sum(w * (x - xbar) ** 2))
    if denom <= 1e-15:
        return float("nan"), float("nan")
    slope = float(np.sum(w * (x - xbar) * (y - ybar)) / denom)
    intercept = float(ybar - slope * xbar)
    return slope, intercept


def _fit_size_extrapolation(size_df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for (model, arch), sub in size_df.groupby(["model_family", "architecture"]):
        if len(sub) < 2:
            continue
        s = sub.sort_values("n_sites").copy()
        x = 1.0 / s["n_sites"].to_numpy(dtype=float)
        y = s["beta_magic"].to_numpy(dtype=float)
        se = ((s["beta_ci_high"].to_numpy(dtype=float) - s["beta_ci_low"].to_numpy(dtype=float)) / 3.92).astype(float)
        slope, intercept = _weighted_line_fit(x, y, se)
        ints: list[float] = []
        for _ in range(n_boot):
            yb = rng.normal(y, np.clip(se, 1e-9, None))
            _, ib = _weighted_line_fit(x, yb, se)
            if np.isfinite(ib):
                ints.append(float(ib))
        if len(ints) >= 8:
            lo, hi = np.quantile(ints, [0.025, 0.975])
        else:
            lo, hi = float("nan"), float("nan")
        rows.append(
            {
                "model_family": model,
                "architecture": arch,
                "n_sizes": int(len(s)),
                "slope_vs_invN": slope,
                "beta_inf": intercept,
                "beta_inf_ci_low": float(lo),
                "beta_inf_ci_high": float(hi),
                "beta_inf_positive": bool(np.isfinite(lo) and lo > 0.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_family", "architecture"]).reset_index(drop=True)


def _cluster_boot_quantile(
    df: pd.DataFrame,
    tau: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    formula = "final_val_nll ~ magic_m2 + snapshot_entropy + C(model_family) + C(architecture)"
    fit = smf.quantreg(formula, data=df).fit(q=tau, max_iter=4000)
    point = float(fit.params.get("magic_m2", float("nan")))

    cond_ids = np.array(sorted(df["condition_id"].astype(str).unique().tolist()))
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(cond_ids, size=len(cond_ids), replace=True)
        parts = [df[df["condition_id"].astype(str) == cid] for cid in pick]
        boot_df = pd.concat(parts, ignore_index=True)
        try:
            bf = smf.quantreg(formula, data=boot_df).fit(q=tau, max_iter=4000)
            vals.append(float(bf.params.get("magic_m2", float("nan"))))
        except Exception:
            continue
    vals = [v for v in vals if np.isfinite(v)]
    if len(vals) < 10:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return point, float(lo), float(hi)


def _plot_beta_by_size(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    for (model, arch), sub in df.groupby(["model_family", "architecture"]):
        s = sub.sort_values("n_sites")
        ax.plot(s["n_sites"], s["beta_magic"], marker="o", lw=1.4, label=f"{model}-{arch}")
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_xlabel("N")
    ax.set_ylabel("beta_magic")
    ax.set_title("Thermodynamic trend: entropy-controlled beta by size")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_beta_inf_forest(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    labels = [f"{m}\n{a}" for m, a in zip(df["model_family"], df["architecture"], strict=False)]
    y = df["beta_inf"].to_numpy(dtype=float)
    lo = df["beta_inf_ci_low"].to_numpy(dtype=float)
    hi = df["beta_inf_ci_high"].to_numpy(dtype=float)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=4, color="#0f5a9c")
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("beta_inf (N→∞ extrapolation)")
    ax.set_title("Thermodynamic extrapolation of beta-law")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_quantile_bound(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    x = df["tau"].to_numpy(dtype=float)
    y = df["beta_magic"].to_numpy(dtype=float)
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=4, color="#7a1f5c")
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_xlabel("Quantile τ")
    ax.set_ylabel("beta_magic at Quantile τ")
    ax.set_title("Lower-envelope theory check (quantile regression)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-4 thermodynamic scaling and theory-bound checks")
    parser.add_argument("--stage3-dir", type=str, default="outputs/stage3_prx/universal_extension_v3")
    parser.add_argument("--registry", type=str, default="outputs/stage3_prx/stage3/effect_registry.json")
    parser.add_argument("--out-dir", type=str, default="outputs/stage4_prl/thermo_theory_v1")
    parser.add_argument("--bootstrap", type=int, default=220)
    parser.add_argument("--quantiles", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--report-out", type=str, default="report/stage4_thermo_theory.md")
    args = parser.parse_args()

    stage3_dir = Path(args.stage3_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    snap = pd.read_csv(stage3_dir / "corridor_snapshot_all_stage3.csv")
    cond = pd.read_csv(stage3_dir / "corridor_condition_stats_stage3.csv")
    stage3_reg = json.loads(Path(args.registry).read_text(encoding="utf-8"))

    size_rows: list[dict[str, Any]] = []
    idx = 0
    for (model, arch, n_sites), sub in snap.groupby(["model_family", "architecture", "n_sites"]):
        beta, lo, hi = _bootstrap_beta_magic(sub, n_boot=args.bootstrap, seed=args.seed + idx * 11)
        size_rows.append(
            {
                "model_family": model,
                "architecture": arch,
                "n_sites": int(n_sites),
                "n_snapshots": int(len(sub)),
                "n_conditions": int(sub["condition_id"].nunique()),
                "beta_magic": beta,
                "beta_ci_low": lo,
                "beta_ci_high": hi,
                "beta_positive": bool(np.isfinite(lo) and lo > 0.0),
            }
        )
        idx += 1
    size_df = pd.DataFrame(size_rows).sort_values(["model_family", "architecture", "n_sites"]).reset_index(drop=True)
    size_df.to_csv(out_dir / "stage4_beta_by_size.csv", index=False)

    thermo_df = _fit_size_extrapolation(size_df=size_df, n_boot=max(120, args.bootstrap), seed=args.seed + 909)
    thermo_df.to_csv(out_dir / "stage4_beta_thermo_extrapolation.csv", index=False)

    taus = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]
    q_rows: list[dict[str, Any]] = []
    for i, tau in enumerate(taus):
        b, lo, hi = _cluster_boot_quantile(
            df=snap,
            tau=tau,
            n_boot=max(120, args.bootstrap),
            seed=args.seed + 2000 + i * 19,
        )
        q_rows.append(
            {
                "tau": tau,
                "beta_magic": b,
                "ci_low": lo,
                "ci_high": hi,
                "positive_lower_bound": bool(np.isfinite(lo) and lo > 0.0),
            }
        )
    q_df = pd.DataFrame(q_rows).sort_values("tau").reset_index(drop=True)
    q_df.to_csv(out_dir / "stage4_quantile_bound.csv", index=False)

    _plot_beta_by_size(size_df, fig_dir / "fig_stage4_beta_by_size.png")
    _plot_beta_inf_forest(thermo_df, fig_dir / "fig_stage4_beta_inf_forest.png")
    _plot_quantile_bound(q_df, fig_dir / "fig_stage4_quantile_bound.png")

    thermo_gate = bool(len(thermo_df) > 0 and np.all(thermo_df["beta_inf_positive"].to_numpy(dtype=bool)))
    max_n = int(size_df["n_sites"].max()) if len(size_df) else -1
    size_max = size_df[size_df["n_sites"] == max_n].copy() if max_n > 0 else pd.DataFrame()
    finite_size_gate = bool(len(size_max) > 0 and np.all(size_max["beta_positive"].to_numpy(dtype=bool)))
    quantile_gate = bool(len(q_df) > 0 and np.all(q_df["positive_lower_bound"].to_numpy(dtype=bool)))
    stage3_beta_gate = bool(stage3_reg.get("gate_flags", {}).get("universal_beta_law", False))
    combined_gate = bool(stage3_beta_gate and thermo_gate and quantile_gate)

    expressive = {"gru", "made"}
    thermo_expr = thermo_df[thermo_df["architecture"].isin(expressive)].copy()
    thermo_gate_expressive = bool(
        len(thermo_expr) > 0 and np.all(thermo_expr["beta_inf_positive"].to_numpy(dtype=bool))
    )
    size_max_expr = size_max[size_max["architecture"].isin(expressive)].copy() if len(size_max) else pd.DataFrame()
    finite_size_gate_expressive = bool(
        len(size_max_expr) > 0 and np.all(size_max_expr["beta_positive"].to_numpy(dtype=bool))
    )
    stage3_records = stage3_reg.get("records", [])
    stage3_expr_ok = bool(
        len(stage3_records) > 0
        and all(
            (
                (str(r.get("architecture")) not in expressive)
                or (
                    bool(r.get("beta_ci_positive", False))
                    and bool(r.get("perm_q_pass", False))
                )
            )
            for r in stage3_records
        )
    )
    combined_gate_expressive = bool(stage3_expr_ok and thermo_gate_expressive and quantile_gate)
    combined_gate_finite_size = bool(stage3_beta_gate and finite_size_gate and quantile_gate)
    combined_gate_finite_size_expressive = bool(stage3_expr_ok and finite_size_gate_expressive and quantile_gate)

    summary = {
        "stage3_dir": str(stage3_dir),
        "registry": args.registry,
        "n_size_cells": int(len(size_df)),
        "n_thermo_cells": int(len(thermo_df)),
        "finite_size_gate_beta_positive_at_maxN": finite_size_gate,
        "finite_size_gate_beta_positive_at_maxN_expressive": finite_size_gate_expressive,
        "thermo_gate_beta_inf_positive": thermo_gate,
        "thermo_gate_beta_inf_positive_expressive": thermo_gate_expressive,
        "quantile_lower_envelope_gate": quantile_gate,
        "stage3_beta_gate": stage3_beta_gate,
        "stage3_beta_gate_expressive": stage3_expr_ok,
        "combined_universal_beta_law_with_thermo_theory": combined_gate,
        "combined_universal_beta_law_with_thermo_theory_expressive": combined_gate_expressive,
        "combined_universal_beta_law_with_finite_size_gate": combined_gate_finite_size,
        "combined_universal_beta_law_with_finite_size_gate_expressive": combined_gate_finite_size_expressive,
        "artifacts": {
            "beta_by_size_csv": str(out_dir / "stage4_beta_by_size.csv"),
            "thermo_csv": str(out_dir / "stage4_beta_thermo_extrapolation.csv"),
            "quantile_csv": str(out_dir / "stage4_quantile_bound.csv"),
            "fig_beta_size": str(fig_dir / "fig_stage4_beta_by_size.png"),
            "fig_beta_inf": str(fig_dir / "fig_stage4_beta_inf_forest.png"),
            "fig_quantile": str(fig_dir / "fig_stage4_quantile_bound.png"),
        },
    }
    (out_dir / "stage4_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Stage 4 Thermodynamic + Theory Checks",
        "",
        "This stage attempts to promote the beta-law into a stronger physics-law statement by adding:",
        "- finite-size extrapolation (`N -> infinity`) on beta-law cells",
        "- lower-envelope (quantile) regression checks as an empirical bound proxy",
        "",
        f"- finite-size gate at max N={max_n} (`beta_ci_low > 0` all cells): `{finite_size_gate}`",
        f"- finite-size gate expressive-only at max N={max_n} (`gru,made`): `{finite_size_gate_expressive}`",
        f"- thermo gate (`beta_inf_ci_low > 0` all cells): `{thermo_gate}`",
        f"- thermo gate expressive-only (`gru,made`): `{thermo_gate_expressive}`",
        f"- quantile lower-envelope gate (`beta_tau_ci_low > 0` for all taus): `{quantile_gate}`",
        f"- inherited stage3 beta gate: `{stage3_beta_gate}`",
        f"- inherited stage3 beta gate expressive-only: `{stage3_expr_ok}`",
        f"- **combined gate (thermo+theory+stage3)**: `{combined_gate}`",
        f"- **combined expressive gate (thermo+theory+stage3)**: `{combined_gate_expressive}`",
        f"- **combined gate (finite-size+theory+stage3)**: `{combined_gate_finite_size}`",
        f"- **combined expressive gate (finite-size+theory+stage3)**: `{combined_gate_finite_size_expressive}`",
        "",
        "## Files",
        f"- `{out_dir / 'stage4_beta_by_size.csv'}`",
        f"- `{out_dir / 'stage4_beta_thermo_extrapolation.csv'}`",
        f"- `{out_dir / 'stage4_quantile_bound.csv'}`",
        f"- `{fig_dir / 'fig_stage4_beta_by_size.png'}`",
        f"- `{fig_dir / 'fig_stage4_beta_inf_forest.png'}`",
        f"- `{fig_dir / 'fig_stage4_quantile_bound.png'}`",
        f"- `{out_dir / 'stage4_summary.json'}`",
    ]
    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_dir / 'stage4_summary.json'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
