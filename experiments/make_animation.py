from __future__ import annotations

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create quench dynamics animation from sweep_grid.npz")
    parser.add_argument("--grid", type=str, default="outputs/sweep_grid.npz")
    parser.add_argument("--out-gif", type=str, default="outputs/figs/quench_dynamics.gif")
    parser.add_argument("--out-mp4", type=str, default="outputs/figs/quench_dynamics.mp4")
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()

    grid_path = Path(args.grid)
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing grid file: {grid_path}")

    data = np.load(grid_path)
    theta = data["theta1"]
    times = data["times"]
    lambda_grid = data["lambda_grid"]
    magic_grid = data["magic_grid"]

    out_gif = Path(args.out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    out_mp4 = Path(args.out_mp4)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax_l, ax_m = axes

    lmin, lmax = float(np.min(lambda_grid)), float(np.max(lambda_grid))
    mmin, mmax = float(np.min(magic_grid)), float(np.max(magic_grid))

    line_l = []
    line_m = []
    for th in theta:
        ll, = ax_l.plot([], [], lw=2.0, alpha=0.85, label=fr"$\theta_1={th:.1f}$")
        mm, = ax_m.plot([], [], lw=2.0, alpha=0.85, label=fr"$\theta_1={th:.1f}$")
        line_l.append(ll)
        line_m.append(mm)

    ax_l.set_xlim(float(times[0]), float(times[-1]))
    ax_l.set_ylim(lmin - 0.05 * (lmax - lmin + 1e-12), lmax + 0.1 * (lmax - lmin + 1e-12))
    ax_l.set_xlabel("Time")
    ax_l.set_ylabel(r"$\lambda(t)$")
    ax_l.set_title("Loschmidt rate accumulation")
    ax_l.grid(alpha=0.25)

    ax_m.set_xlim(float(times[0]), float(times[-1]))
    ax_m.set_ylim(mmin - 0.05 * (mmax - mmin + 1e-12), mmax + 0.1 * (mmax - mmin + 1e-12))
    ax_m.set_xlabel("Time")
    ax_m.set_ylabel(r"$M_2(t)$")
    ax_m.set_title("Magic accumulation")
    ax_m.grid(alpha=0.25)

    time_txt = fig.text(0.5, 0.98, "", ha="center", va="top", fontsize=11)

    def init():
        for ll, mm in zip(line_l, line_m):
            ll.set_data([], [])
            mm.set_data([], [])
        time_txt.set_text("")
        return [*line_l, *line_m, time_txt]

    def update(frame: int):
        x = times[: frame + 1]
        for i, (ll, mm) in enumerate(zip(line_l, line_m)):
            ll.set_data(x, lambda_grid[i, : frame + 1])
            mm.set_data(x, magic_grid[i, : frame + 1])
        time_txt.set_text(f"t = {times[frame]:.2f}")
        return [*line_l, *line_m, time_txt]

    ani = FuncAnimation(fig, update, init_func=init, frames=len(times), interval=1000 / args.fps, blit=True)

    ani.save(out_gif, writer=PillowWriter(fps=args.fps))

    try:
        ani.save(out_mp4, writer="ffmpeg", fps=args.fps)
        print(f"Saved GIF + MP4: {out_gif}, {out_mp4}")
    except Exception:
        print(f"Saved GIF only: {out_gif} (ffmpeg unavailable for MP4)")

    plt.close(fig)


if __name__ == "__main__":
    main()
