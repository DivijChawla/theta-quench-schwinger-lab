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
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _fmt_bool(v: bool) -> str:
    return "PASS" if bool(v) else "FAIL"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one capsule animation summarizing the entire project story.")
    parser.add_argument("--grid", type=str, default="outputs/sweep_grid.npz")
    parser.add_argument("--quench", type=str, default="outputs/quench_summary.csv")
    parser.add_argument("--nnqs", type=str, default="outputs/nnqs/nnqs_snapshot_metrics.csv")
    parser.add_argument("--validation", type=str, default="outputs/validation.json")
    parser.add_argument("--stage3", type=str, default="outputs/stage3_prx/universal_extension_v8/stage3_summary.json")
    parser.add_argument("--stage4", type=str, default="outputs/stage4_prl/thermo_theory_v5/stage4_summary.json")
    parser.add_argument("--stage5", type=str, default="outputs/stage5_prx/external_regime_v1/stage5_summary.json")
    parser.add_argument("--out-gif", type=str, default="outputs/figs/project_capsule.gif")
    parser.add_argument("--out-mp4", type=str, default="outputs/figs/project_capsule.mp4")
    parser.add_argument("--fps", type=int, default=14)
    args = parser.parse_args()

    grid_path = Path(args.grid)
    quench_path = Path(args.quench)
    nnqs_path = Path(args.nnqs)
    val_path = Path(args.validation)
    s3_path = Path(args.stage3)
    s4_path = Path(args.stage4)
    s5_path = Path(args.stage5)

    if not grid_path.exists():
        raise FileNotFoundError(f"Missing grid: {grid_path}")
    if not quench_path.exists():
        raise FileNotFoundError(f"Missing quench summary: {quench_path}")
    if not nnqs_path.exists():
        raise FileNotFoundError(f"Missing NNQS snapshot metrics: {nnqs_path}")

    grid = np.load(grid_path)
    theta1 = grid["theta1"]
    times = grid["times"]
    lambda_grid = grid["lambda_grid"]
    magic_grid = grid["magic_grid"]

    q = pd.read_csv(quench_path).sort_values("theta1").reset_index(drop=True)
    n = pd.read_csv(nnqs_path).sort_values("time").reset_index(drop=True)

    val = json.loads(val_path.read_text(encoding="utf-8")) if val_path.exists() else {}
    s3 = json.loads(s3_path.read_text(encoding="utf-8")) if s3_path.exists() else {}
    s4 = json.loads(s4_path.read_text(encoding="utf-8")) if s4_path.exists() else {}
    s5 = json.loads(s5_path.read_text(encoding="utf-8")) if s5_path.exists() else {}

    corr_nll = _safe_corr(n["magic_m2"].to_numpy(float), n["final_val_nll"].to_numpy(float))
    corr_kl = _safe_corr(n["magic_m2"].to_numpy(float), n["final_kl"].to_numpy(float))
    corr_theta_l = _safe_corr(q["theta1"].to_numpy(float), q["max_lambda"].to_numpy(float))
    corr_theta_m = _safe_corr(q["theta1"].to_numpy(float), q["max_magic_m2"].to_numpy(float))

    # Map NNQS snapshots onto evolution frame indices.
    snap_frame_idx = np.searchsorted(times, n["time"].to_numpy(float), side="right") - 1
    snap_frame_idx = np.clip(snap_frame_idx, 0, len(times) - 1)

    out_gif = Path(args.out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(args.out_mp4)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    ax_l = axes[0, 0]
    ax_m = axes[0, 1]
    ax_s = axes[1, 0]
    ax_t = axes[1, 1]

    # Panel A: Loschmidt dynamics.
    lmin, lmax = float(np.min(lambda_grid)), float(np.max(lambda_grid))
    l_lines = []
    for th in theta1:
        ln, = ax_l.plot([], [], lw=2.0, alpha=0.9, label=fr"$\theta_1={th:.1f}$")
        l_lines.append(ln)
    ax_l.set_xlim(float(times[0]), float(times[-1]))
    ax_l.set_ylim(lmin - 0.05 * (lmax - lmin + 1e-12), lmax + 0.12 * (lmax - lmin + 1e-12))
    ax_l.set_title("A) Dynamical-transition diagnostic: Loschmidt rate")
    ax_l.set_xlabel("time")
    ax_l.set_ylabel(r"$\lambda(t)$")
    ax_l.grid(alpha=0.25)
    ax_l.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    # Panel B: magic dynamics.
    mmin, mmax = float(np.min(magic_grid)), float(np.max(magic_grid))
    m_lines = []
    for th in theta1:
        ln, = ax_m.plot([], [], lw=2.0, alpha=0.9, label=fr"$\theta_1={th:.1f}$")
        m_lines.append(ln)
    ax_m.set_xlim(float(times[0]), float(times[-1]))
    ax_m.set_ylim(mmin - 0.05 * (mmax - mmin + 1e-12), mmax + 0.12 * (mmax - mmin + 1e-12))
    ax_m.set_title("B) Stabilizer complexity growth")
    ax_m.set_xlabel("time")
    ax_m.set_ylabel(r"$M_2(t)$")
    ax_m.grid(alpha=0.25)

    # Panel C: NNQS learnability vs magic (incremental points as time advances).
    x_all = n["magic_m2"].to_numpy(float)
    y_all = n["final_val_nll"].to_numpy(float)
    xr = np.linspace(float(np.min(x_all)), float(np.max(x_all)), 100)
    if len(x_all) >= 2 and np.std(x_all) > 1e-14:
        slope, intercept = np.polyfit(x_all, y_all, 1)
        yr = slope * xr + intercept
        ax_s.plot(xr, yr, color="#0f5a9c", lw=2.0, alpha=0.9, label=f"fit slope={slope:.3f}")
    scat = ax_s.scatter([], [], s=55, c=[], cmap="viridis", vmin=float(np.min(n["time"])), vmax=float(np.max(n["time"])))
    ax_s.set_title("C) NNQS learnability vs magic (snapshot trajectory)")
    ax_s.set_xlabel(r"$M_2$ snapshot")
    ax_s.set_ylabel("final validation NLL")
    ax_s.grid(alpha=0.25)
    ax_s.legend(frameon=False, fontsize=8, loc="upper left")

    # Panel D: global summary + gate board.
    ax_t.axis("off")

    gate_items = [
        ("Hermitian + norm conservation", bool(val.get("hermitian_h0", False) and val.get("hermitian_h1", False))),
        ("Stage-3 beta law", bool(s3.get("universal_beta_law", False))),
        ("Stage-3 sign law", bool(s3.get("universal_sign_law", False))),
        ("Stage-4 thermo extrapolation", bool(s4.get("thermo_gate_beta_inf_positive", False))),
        ("Stage-4 quantile envelope", bool(s4.get("quantile_lower_envelope_gate", False))),
        ("Stage-5 beta cell gate", bool(s5.get("beta_cell_gate", False))),
        ("Stage-5 regime minimax gate", bool(s5.get("minimax_regime_gate", False))),
    ]

    global_title = fig.suptitle("", fontsize=14, fontweight="bold")

    def _draw_text_panel(frame: int) -> None:
        ax_t.clear()
        ax_t.axis("off")

        t_now = float(times[frame])
        completed = snap_frame_idx <= frame
        n_seen = int(np.sum(completed))
        x_seen = x_all[completed]
        y_seen = y_all[completed]
        c_seen = n.loc[completed, "time"].to_numpy(float)
        scat.set_offsets(np.c_[x_seen, y_seen] if n_seen > 0 else np.empty((0, 2)))
        scat.set_array(c_seen if n_seen > 0 else np.array([]))

        mono_l = bool(np.all(np.diff(q["max_lambda"].to_numpy(float)) >= -1e-12))
        mono_m = bool(np.all(np.diff(q["max_magic_m2"].to_numpy(float)) >= -1e-12))

        txt = [
            "D) Project capsule (single-view summary)",
            "",
            f"Current time slice: t = {t_now:.2f}",
            f"NNQS snapshots integrated: {n_seen}/{len(n)}",
            "",
            f"corr(theta1, max lambda) = {corr_theta_l:.3f}",
            f"corr(theta1, max M2) = {corr_theta_m:.3f}",
            f"corr(M2, val NLL) = {corr_nll:.3f}",
            f"corr(M2, KL) = {corr_kl:.3f}",
            "",
            f"Monotone quench trend (lambda): {_fmt_bool(mono_l)}",
            f"Monotone quench trend (magic): {_fmt_bool(mono_m)}",
            "",
            "Gate board:",
        ]
        y0 = 0.98
        for i, line in enumerate(txt):
            ax_t.text(0.02, y0 - 0.055 * i, line, va="top", fontsize=10.5)

        y_gate = 0.22
        for idx, (name, ok) in enumerate(gate_items):
            color = "#1a7f37" if ok else "#b42318"
            ax_t.add_patch(plt.Rectangle((0.02, y_gate - 0.09 * idx), 0.025, 0.03, color=color, transform=ax_t.transAxes))
            ax_t.text(0.055, y_gate - 0.09 * idx + 0.015, f"{name}: {_fmt_bool(ok)}", va="center", fontsize=10.5)

    def init():
        for ln in l_lines:
            ln.set_data([], [])
        for ln in m_lines:
            ln.set_data([], [])
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array(np.array([]))
        _draw_text_panel(0)
        global_title.set_text("Theta-Quench Schwinger Lab | dynamics + magic + NNQS + claim gates")
        return [*l_lines, *m_lines, scat, global_title]

    def update(frame: int):
        x = times[: frame + 1]
        for i, ln in enumerate(l_lines):
            ln.set_data(x, lambda_grid[i, : frame + 1])
        for i, ln in enumerate(m_lines):
            ln.set_data(x, magic_grid[i, : frame + 1])
        _draw_text_panel(frame)
        global_title.set_text(
            f"Theta-Quench Schwinger Lab | dynamics + magic + NNQS + claim gates | t={times[frame]:.2f}"
        )
        return [*l_lines, *m_lines, scat, global_title]

    ani = FuncAnimation(fig, update, init_func=init, frames=len(times), interval=1000 / args.fps, blit=False)
    ani.save(out_gif, writer=PillowWriter(fps=args.fps))

    try:
        ani.save(out_mp4, writer="ffmpeg", fps=args.fps)
        print(f"Saved GIF + MP4: {out_gif}, {out_mp4}")
    except Exception:
        print(f"Saved GIF only: {out_gif} (ffmpeg unavailable for MP4)")

    plt.close(fig)


if __name__ == "__main__":
    main()
