from __future__ import annotations

import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .pipeline import QuenchRun


def _prepare_path(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_loschmidt_family(runs: list[QuenchRun], save_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for run in runs:
        ax.plot(run.times, run.loschmidt_rate, lw=2, label=fr"$\theta_1={run.theta1:.2f}$")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Rate function $\lambda(t)$")
    ax.set_title("Loschmidt rate function under $\\theta$ quenches")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_prepare_path(save_path), dpi=220)
    plt.close(fig)


def plot_magic_lambda_overlay(run: QuenchRun, save_path: str | Path, alpha: float = 2.0) -> None:
    if alpha not in run.magic:
        return

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(run.times, run.loschmidt_rate, color="#125D98", lw=2.2, label=r"$\lambda(t)$")
    ax2.plot(run.times, run.magic[alpha], color="#DD6E42", lw=2.2, label=fr"$M_{{{alpha:g}}}(t)$")

    ax1.set_xlabel("Time")
    ax1.set_ylabel(r"$\lambda(t)$", color="#125D98")
    ax2.set_ylabel(fr"$M_{{{alpha:g}}}(t)$", color="#DD6E42")
    ax1.tick_params(axis="y", colors="#125D98")
    ax2.tick_params(axis="y", colors="#DD6E42")
    ax1.set_title(fr"Rate function and magic, $\theta_1={run.theta1:.2f}$")
    ax1.grid(alpha=0.2)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(_prepare_path(save_path), dpi=220)
    plt.close(fig)


def plot_heatmaps(
    times: np.ndarray,
    theta1_values: np.ndarray,
    lambda_grid: np.ndarray,
    magic_grid: np.ndarray,
    save_path: str | Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharex=True, sharey=True)

    im0 = axes[0].imshow(
        lambda_grid,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], theta1_values[0], theta1_values[-1]],
        cmap="viridis",
    )
    axes[0].set_title(r"$\lambda(t,\theta_1)$")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$\theta_1$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        magic_grid,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], theta1_values[0], theta1_values[-1]],
        cmap="magma",
    )
    axes[1].set_title(r"$M_2(t,\theta_1)$")
    axes[1].set_xlabel("Time")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(_prepare_path(save_path), dpi=240)
    plt.close(fig)


def plot_nnqs_scatter(
    df: pd.DataFrame,
    save_path: str | Path,
    y_col: str = "final_val_nll",
) -> None:
    if df.empty:
        return

    x = df["magic_m2"].to_numpy()
    if y_col not in df.columns:
        y_col = "final_kl"
    y = df[y_col].to_numpy()

    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    ax.scatter(x, y, color="#1E88A8", s=50, alpha=0.85)

    if len(df) >= 2:
        coeff = np.polyfit(x, y, deg=1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        y_line = coeff[0] * x_line + coeff[1]
        ax.plot(x_line, y_line, color="#BF3F34", lw=2.0, label=f"slope={coeff[0]:.3f}")
        ax.legend(frameon=False)

    ax.set_xlabel(r"Snapshot magic $M_2$")
    if y_col == "final_val_nll":
        ax.set_ylabel("Final validation NLL")
        ax.set_title("NNQS learnability vs magic (loss-based)")
    else:
        ax.set_ylabel(r"Final KL$(p_{\mathrm{true}}\|p_\theta)$")
        ax.set_title("NNQS learnability vs magic")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(_prepare_path(save_path), dpi=220)
    plt.close(fig)


def plot_entropy_vs_magic(
    times: np.ndarray,
    entropy: np.ndarray,
    magic: np.ndarray,
    save_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    sc = ax.scatter(entropy, magic, c=times, cmap="plasma", s=48)
    ax.set_xlabel(r"Bipartite entropy $S_A$")
    ax.set_ylabel(r"Magic $M_2$")
    ax.set_title("Entanglement vs magic trajectory")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Time")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(_prepare_path(save_path), dpi=220)
    plt.close(fig)
