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
from scipy.stats import pearsonr

from tqm.config import NNQSConfig
from tqm.magic import magic_over_time
from tqm.nnqs.data import sample_bitstrings_from_state, train_val_split
from tqm.nnqs.train import train_nnqs_on_state


def _basis_state(n_sites: int, index: int = 0) -> np.ndarray:
    psi = np.zeros(2**n_sites, dtype=np.complex128)
    psi[index] = 1.0
    return psi


def _ghz_state(n_sites: int) -> np.ndarray:
    psi = np.zeros(2**n_sites, dtype=np.complex128)
    psi[0] = 1.0 / np.sqrt(2.0)
    psi[-1] = 1.0 / np.sqrt(2.0)
    return psi


def _haar_state(n_sites: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = 2**n_sites
    re = rng.normal(size=dim)
    im = rng.normal(size=dim)
    psi = re + 1j * im
    psi = psi / np.linalg.norm(psi)
    return psi.astype(np.complex128)


def _mix_states(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    psi = (1.0 - alpha) * a + alpha * b
    psi = psi / np.linalg.norm(psi)
    return psi


def _bernoulli_nll(bits: np.ndarray, probs: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(probs, eps, 1.0 - eps)
    ll = bits * np.log(p[None, :]) + (1.0 - bits) * np.log(1.0 - p[None, :])
    return float(-np.mean(np.sum(ll, axis=1)))


def _independent_val_nll(
    psi: np.ndarray,
    n_sites: int,
    cfg: NNQSConfig,
    seed: int,
) -> float:
    samples = sample_bitstrings_from_state(
        psi=psi,
        n_sites=n_sites,
        num_samples=cfg.measurement_samples,
        seed=seed,
    ).astype(float)
    train_np, val_np = train_val_split(samples, val_fraction=cfg.val_fraction, seed=seed)
    p_hat = np.clip(np.mean(train_np, axis=0), 1e-6, 1.0 - 1e-6)
    return _bernoulli_nll(val_np, p_hat)


def _state_family(n_sites: int, seeds: list[int], alphas: list[float]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = [
        {"family": "basis", "alpha": 0.0, "seed": -1, "psi": _basis_state(n_sites, index=0)},
        {"family": "ghz", "alpha": 0.0, "seed": -1, "psi": _ghz_state(n_sites)},
    ]
    base = _basis_state(n_sites, index=0)
    for seed in seeds:
        haar = _haar_state(n_sites, seed=seed)
        for alpha in alphas:
            out.append(
                {
                    "family": "mix_basis_haar",
                    "alpha": float(alpha),
                    "seed": int(seed),
                    "psi": _mix_states(base, haar, alpha=float(alpha)),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Positive-control synthetic magic family experiment")
    parser.add_argument("--n-sites", type=int, default=6)
    parser.add_argument("--architectures", type=str, default="gru,made,independent")
    parser.add_argument("--seeds", type=str, default="11,17,23,29")
    parser.add_argument("--alphas", type=str, default="0.1,0.2,0.35,0.5,0.7,0.9")
    parser.add_argument("--measurement-samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--out-dir", type=str, default="outputs/positive_control")
    args = parser.parse_args()

    archs = [a.strip().lower() for a in args.architectures.split(",") if a.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = NNQSConfig(
        enabled=True,
        snapshot_count=1,
        measurement_samples=int(args.measurement_samples),
        val_fraction=0.2,
        hidden_size=48,
        epochs=int(args.epochs),
        lr=1e-3,
        batch_size=256,
        threshold_nll=2.8,
        seed=11,
    )

    rows: list[dict[str, float | str | int]] = []
    states = _state_family(args.n_sites, seeds=seeds, alphas=alphas)

    for idx, item in enumerate(states):
        psi = item["psi"]
        m2 = float(
            magic_over_time(
                states=np.asarray([psi]),
                n_sites=args.n_sites,
                alphas=[2.0],
                z_batch_size=128,
            )[2.0][0]
        )

        for arch in archs:
            if arch == "independent":
                val_nll = _independent_val_nll(psi=psi, n_sites=args.n_sites, cfg=cfg, seed=cfg.seed + idx)
            else:
                res = train_nnqs_on_state(
                    psi=psi,
                    n_sites=args.n_sites,
                    cfg=cfg,
                    seed=cfg.seed + idx,
                    model_type=arch,
                )
                val_nll = float(res.final_val_nll)

            rows.append(
                {
                    "architecture": arch,
                    "family": str(item["family"]),
                    "alpha": float(item["alpha"]),
                    "seed": int(item["seed"]),
                    "magic_m2": m2,
                    "final_val_nll": val_nll,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "positive_control_rows.csv", index=False)

    summary_rows: list[dict[str, float | str]] = []
    for arch in sorted(df["architecture"].unique()):
        sub = df[df["architecture"] == arch]
        r, p = pearsonr(sub["magic_m2"].to_numpy(dtype=float), sub["final_val_nll"].to_numpy(dtype=float))
        summary_rows.append(
            {
                "architecture": arch,
                "pearson_magic_vs_val_nll": float(r),
                "pearson_p": float(p),
                "positive_control_pass": bool(r > 0.2),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "positive_control_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for arch in sorted(df["architecture"].unique()):
        sub = df[df["architecture"] == arch]
        ax.scatter(
            sub["magic_m2"],
            sub["final_val_nll"],
            s=24,
            alpha=0.7,
            label=arch,
        )
    ax.set_xlabel("Magic M2")
    ax.set_ylabel("Validation NLL")
    ax.set_title("Positive control: synthetic state family")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "positive_control_scatter.png", dpi=220)
    plt.close(fig)

    payload = {
        "n_sites": int(args.n_sites),
        "architectures": archs,
        "seeds": seeds,
        "alphas": alphas,
        "summary": summary_df.to_dict(orient="records"),
    }
    with (out_dir / "positive_control_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_dir / 'positive_control_rows.csv'}")
    print(f"Wrote {out_dir / 'positive_control_summary.csv'}")
    print(f"Wrote {out_dir / 'positive_control_summary.json'}")
    print(f"Wrote {fig_dir / 'positive_control_scatter.png'}")


if __name__ == "__main__":
    main()
