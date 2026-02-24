from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig, load_config
from .magic import magic_over_time
from .pipeline import run_magic_sanity, run_single_quench, save_quench_run
from .plotting import (
    plot_entropy_vs_magic,
    plot_heatmaps,
    plot_loschmidt_family,
    plot_magic_lambda_overlay,
    plot_nnqs_scatter,
)


def _ensure_output_dirs(cfg: ExperimentConfig) -> tuple[Path, Path]:
    out_dir = Path(cfg.output.out_dir)
    fig_dir = Path(cfg.output.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "nnqs").mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir


def _run_sweep(cfg: ExperimentConfig) -> list:
    out_dir, fig_dir = _ensure_output_dirs(cfg)
    runs = []
    for idx, theta1 in enumerate(cfg.model.theta1_values):
        run = run_single_quench(
            cfg,
            theta1=theta1,
            compute_dense_krylov=(idx == 0),
        )
        runs.append(run)
        label = str(theta1).replace(".", "p")
        save_quench_run(run, out_dir / "runs" / f"quench_theta1_{label}.npz")

    plot_loschmidt_family(runs, fig_dir / "fig1_loschmidt_family.png")

    best_overlay = runs[-1]
    plot_magic_lambda_overlay(best_overlay, fig_dir / "fig2_magic_lambda_overlay.png", alpha=2.0)

    times = runs[0].times
    theta = np.array([r.theta1 for r in runs], dtype=float)
    lambda_grid = np.stack([r.loschmidt_rate for r in runs], axis=0)

    if all(2.0 in r.magic for r in runs):
        magic_grid = np.stack([r.magic[2.0] for r in runs], axis=0)
    else:
        magic_grid = np.zeros_like(lambda_grid)
    plot_heatmaps(times, theta, lambda_grid, magic_grid, fig_dir / "fig3_heatmaps_lambda_magic.png")

    if 2.0 in best_overlay.magic:
        plot_entropy_vs_magic(
            times=best_overlay.times,
            entropy=best_overlay.entanglement_entropy,
            magic=best_overlay.magic[2.0],
            save_path=fig_dir / "fig5_entropy_vs_magic.png",
        )

    summary_rows = []
    for r in runs:
        row = {
            "theta1": r.theta1,
            "max_lambda": float(np.max(r.loschmidt_rate)),
            "max_entropy": float(np.max(r.entanglement_entropy)),
            "norm_drift": r.norm_drift,
            "dense_krylov_error": r.dense_krylov_error,
            "hermitian_h0": r.hermitian_h0,
            "hermitian_h1": r.hermitian_h1,
        }
        if 2.0 in r.magic:
            row["max_magic_m2"] = float(np.max(r.magic[2.0]))
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(out_dir / "quench_summary.csv", index=False)
    np.savez(
        out_dir / "sweep_grid.npz",
        theta1=theta,
        times=times,
        lambda_grid=lambda_grid,
        magic_grid=magic_grid,
    )

    return runs


def _plot_nnqs_histories(histories: dict[int, dict[str, list[float]]], save_path: Path) -> None:
    if not histories:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for idx, hist in histories.items():
        ax.plot(hist["val_nll"], alpha=0.65, lw=1.5, label=f"idx={idx}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation NLL")
    ax.set_title("NNQS convergence across snapshots")
    ax.grid(alpha=0.2)
    if len(histories) <= 8:
        ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _run_nnqs(cfg: ExperimentConfig, runs: list | None = None) -> pd.DataFrame:
    if importlib.util.find_spec("torch") is None:
        raise ModuleNotFoundError(
            "NNQS requires torch, but torch is not installed in the current environment."
        )
    from .nnqs.train import run_snapshot_study

    out_dir, fig_dir = _ensure_output_dirs(cfg)
    if runs is None or len(runs) == 0:
        runs = [run_single_quench(cfg, theta1=max(cfg.model.theta1_values))]

    target = runs[-1]
    if 2.0 not in target.magic:
        magic = magic_over_time(
            states=target.state_trajectory,
            n_sites=cfg.model.n_sites,
            alphas=[2.0],
            z_batch_size=cfg.magic.z_batch_size,
        )
        m2 = magic[2.0]
    else:
        m2 = target.magic[2.0]

    df, histories = run_snapshot_study(
        states=target.state_trajectory,
        times=target.times,
        n_sites=cfg.model.n_sites,
        magic_m2=m2,
        cfg=cfg.nnqs,
    )
    df.to_csv(out_dir / "nnqs" / "nnqs_snapshot_metrics.csv", index=False)

    with (out_dir / "nnqs" / "nnqs_histories.json").open("w", encoding="utf-8") as f:
        json.dump(histories, f)

    plot_nnqs_scatter(df, fig_dir / "fig4_nnqs_loss_vs_magic.png")
    _plot_nnqs_histories(histories, fig_dir / "fig4b_nnqs_val_curves.png")
    return df


def _run_validation(cfg: ExperimentConfig) -> dict[str, float | bool]:
    out_dir, _ = _ensure_output_dirs(cfg)
    val_cfg = copy.deepcopy(cfg)
    val_cfg.model.n_sites = min(val_cfg.model.n_sites, 6)
    val_cfg.evolution.n_steps = min(val_cfg.evolution.n_steps, 31)
    val_cfg.evolution.t_max = min(val_cfg.evolution.t_max, 2.0)
    val_cfg.magic.alphas = [2.0]

    run = run_single_quench(
        val_cfg,
        theta1=val_cfg.model.theta1_values[0],
        compute_dense_krylov=True,
    )
    sanity = run_magic_sanity(val_cfg)

    payload: dict[str, float | bool] = {
        "hermitian_h0": run.hermitian_h0,
        "hermitian_h1": run.hermitian_h1,
        "norm_drift": run.norm_drift,
        "dense_krylov_error": run.dense_krylov_error,
    }
    for key, val in sanity.items():
        payload[f"magic_{key}"] = float(val)

    with (out_dir / "validation.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def _sync_report_figs(cfg: ExperimentConfig) -> None:
    fig_dir = Path(cfg.output.fig_dir)
    report_fig_dir = Path("report/figs")
    report_fig_dir.mkdir(parents=True, exist_ok=True)
    for png in fig_dir.glob("*.png"):
        shutil.copy2(png, report_fig_dir / png.name)


def run_all(cfg: ExperimentConfig) -> None:
    runs = _run_sweep(cfg)
    if cfg.nnqs.enabled:
        _run_nnqs(cfg, runs=runs)
    _run_validation(cfg)
    _sync_report_figs(cfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Theta-Quench Schwinger Lab CLI")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config")

    group = p.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run full pipeline")
    group.add_argument("--sweep", action="store_true", help="Run theta1 sweep")
    group.add_argument("--nnqs", action="store_true", help="Run NNQS on snapshot states")
    group.add_argument("--validate", action="store_true", help="Run validation checks")
    p.add_argument("--theta1", type=float, default=None, help="Single-quench theta1 override")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    if args.theta1 is not None:
        out_dir, _ = _ensure_output_dirs(cfg)
        run = run_single_quench(cfg, theta1=float(args.theta1))
        save_quench_run(run, out_dir / "runs" / f"single_theta1_{args.theta1:.3f}.npz")
        return

    if args.sweep:
        _run_sweep(cfg)
        return
    if args.nnqs:
        _run_nnqs(cfg)
        return
    if args.validate:
        _run_validation(cfg)
        return

    run_all(cfg)


if __name__ == "__main__":
    main()
