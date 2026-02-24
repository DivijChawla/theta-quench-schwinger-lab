from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tqm.config import load_config
from tqm.magic import magic_over_time
from tqm.nnqs.train import run_snapshot_study
from tqm.pipeline import run_single_quench
from tqm.plotting import plot_nnqs_scatter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NNQS learnability study on quench snapshots")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--theta1", type=float, default=None, help="Override theta1 for snapshot source")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg.output.out_dir)
    fig_dir = Path(cfg.output.fig_dir)
    (out_dir / "nnqs").mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    theta1 = float(args.theta1) if args.theta1 is not None else float(max(cfg.model.theta1_values))
    run = run_single_quench(cfg, theta1=theta1)

    if 2.0 in run.magic:
        m2 = run.magic[2.0]
    else:
        m2 = magic_over_time(
            states=run.state_trajectory,
            n_sites=cfg.model.n_sites,
            alphas=[2.0],
            z_batch_size=cfg.magic.z_batch_size,
        )[2.0]

    df, _ = run_snapshot_study(
        states=run.state_trajectory,
        times=run.times,
        n_sites=cfg.model.n_sites,
        magic_m2=m2,
        cfg=cfg.nnqs,
    )
    df.to_csv(out_dir / "nnqs" / "nnqs_snapshot_metrics.csv", index=False)
    plot_nnqs_scatter(df, fig_dir / "fig4_nnqs_loss_vs_magic.png")

    if len(df) > 1:
        corr_kl = np.corrcoef(df["magic_m2"], df["final_kl"])[0, 1]
        corr_nll = np.corrcoef(df["magic_m2"], df["final_val_nll"])[0, 1]
    else:
        corr_kl = np.nan
        corr_nll = np.nan
    print(f"Correlation(M2, final_val_nll) = {corr_nll:.4f}")
    print(f"Correlation(M2, final_KL) = {corr_kl:.4f}")


if __name__ == "__main__":
    main()
