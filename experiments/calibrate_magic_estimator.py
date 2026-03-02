from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((ROOT / ".mplconfig").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqm.config import load_config
from tqm.magic import magic_over_time
from tqm.pipeline import run_single_quench

from experiments.novelty_robustness import approximate_magic_m2_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate approximate M2 estimator against exact M2")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-sites", type=int, default=8)
    parser.add_argument("--theta1", type=str, default="0.0,1.2,2.4")
    parser.add_argument("--sample-grid", type=str, default="1000,2500,5000,10000,20000")
    parser.add_argument("--n-times", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="outputs/magic_calibration")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.model.n_sites = int(args.n_sites)
    cfg.magic.alphas = [2.0]
    cfg.magic.max_sites_exact = max(cfg.magic.max_sites_exact, int(args.n_sites))

    theta_values = [float(x.strip()) for x in args.theta1.split(",") if x.strip()]
    sample_grid = [int(x.strip()) for x in args.sample_grid.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int]] = []
    for theta in theta_values:
        run = run_single_quench(cfg, theta1=theta, compute_dense_krylov=False)
        idx = np.unique(np.linspace(0, len(run.times) - 1, args.n_times, dtype=int))
        states = run.state_trajectory[idx]
        exact = magic_over_time(states=states, n_sites=cfg.model.n_sites, alphas=[2.0], z_batch_size=cfg.magic.z_batch_size)[2.0]

        for n_samples in sample_grid:
            for j, psi in enumerate(states):
                approx = approximate_magic_m2_state(
                    psi=psi,
                    n_sites=cfg.model.n_sites,
                    n_pauli_samples=n_samples,
                    seed=10_000 + 100 * j + n_samples + int(round(theta * 1000)),
                )
                err = float(abs(approx - exact[j]))
                rows.append(
                    {
                        "theta1": float(theta),
                        "time_index": int(idx[j]),
                        "time": float(run.times[idx[j]]),
                        "n_pauli_samples": int(n_samples),
                        "m2_exact": float(exact[j]),
                        "m2_approx": float(approx),
                        "abs_error": err,
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "magic_calibration_rows.csv", index=False)

    summary = (
        df.groupby("n_pauli_samples", as_index=False)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            max_abs_error=("abs_error", "max"),
        )
        .reset_index(drop=True)
    )
    summary.to_csv(out_dir / "magic_calibration_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(summary["n_pauli_samples"], summary["mean_abs_error"], marker="o", label="mean abs error")
    ax.plot(summary["n_pauli_samples"], summary["median_abs_error"], marker="s", label="median abs error")
    ax.plot(summary["n_pauli_samples"], summary["max_abs_error"], marker="^", label="max abs error")
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo Pauli samples")
    ax.set_ylabel("|M2_approx - M2_exact|")
    ax.set_title("Approximate magic estimator calibration")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "magic_calibration_error_vs_samples.png", dpi=220)
    plt.close(fig)

    payload = {
        "n_sites": cfg.model.n_sites,
        "theta1_values": theta_values,
        "sample_grid": sample_grid,
        "rows": int(len(df)),
        "summary": summary.to_dict(orient="records"),
    }
    with (out_dir / "magic_calibration_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    report_md = Path("report/magic_estimator_validation.md")
    lines = [
        "# Approximate Magic Estimator Validation",
        "",
        f"- n_sites: `{cfg.model.n_sites}`",
        f"- theta1 values: `{theta_values}`",
        f"- sample grid: `{sample_grid}`",
        "",
        "## Error envelope",
    ]
    for r in summary.to_dict(orient="records"):
        lines.append(
            f"- samples={int(r['n_pauli_samples'])}: mean={r['mean_abs_error']:.4e}, "
            f"median={r['median_abs_error']:.4e}, max={r['max_abs_error']:.4e}"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            f"- `{out_dir / 'magic_calibration_rows.csv'}`",
            f"- `{out_dir / 'magic_calibration_summary.csv'}`",
            f"- `{fig_dir / 'magic_calibration_error_vs_samples.png'}`",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_dir / 'magic_calibration_rows.csv'}")
    print(f"Wrote {out_dir / 'magic_calibration_summary.csv'}")
    print(f"Wrote {out_dir / 'magic_calibration_summary.json'}")
    print(f"Wrote {fig_dir / 'magic_calibration_error_vs_samples.png'}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
