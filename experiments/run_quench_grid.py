from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tqm.config import load_config
from tqm.pipeline import run_single_quench, save_quench_run
from tqm.plotting import plot_heatmaps, plot_loschmidt_family, plot_magic_lambda_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run theta1 quench sweep")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.output.out_dir)
    fig_dir = Path(cfg.output.fig_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for theta1 in cfg.model.theta1_values:
        run = run_single_quench(cfg, theta1)
        runs.append(run)
        save_quench_run(run, out_dir / "runs" / f"quench_theta1_{theta1:.3f}.npz")

    plot_loschmidt_family(runs, fig_dir / "fig1_loschmidt_family.png")
    plot_magic_lambda_overlay(runs[-1], fig_dir / "fig2_magic_lambda_overlay.png", alpha=2.0)

    theta = np.array([r.theta1 for r in runs])
    times = runs[0].times
    lambda_grid = np.stack([r.loschmidt_rate for r in runs], axis=0)
    magic_grid = np.stack([r.magic.get(2.0, np.zeros_like(r.loschmidt_rate)) for r in runs], axis=0)
    plot_heatmaps(times, theta, lambda_grid, magic_grid, fig_dir / "fig3_heatmaps_lambda_magic.png")

    pd.DataFrame(
        {
            "theta1": theta,
            "peak_lambda": [float(np.max(r.loschmidt_rate)) for r in runs],
            "peak_m2": [float(np.max(r.magic.get(2.0, np.zeros_like(r.loschmidt_rate)))) for r in runs],
            "norm_drift": [r.norm_drift for r in runs],
            "dense_krylov_error": [r.dense_krylov_error for r in runs],
        }
    ).to_csv(out_dir / "quench_summary.csv", index=False)


if __name__ == "__main__":
    main()
