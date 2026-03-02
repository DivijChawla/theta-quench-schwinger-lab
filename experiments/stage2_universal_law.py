from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import yaml
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqm.evolve import evolve_state
from tqm.loschmidt import rate_function
from tqm.magic import magic_over_time
from tqm.nnqs.train import run_snapshot_study
from tqm.observables import bipartite_entropy_time
from tqm.config import NNQSConfig


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan"), float("nan")
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    if len(y) != len(z):
        raise ValueError("y and z must have same length")
    if len(y) < 3:
        return np.full_like(y, np.nan, dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(z).all():
        return np.full_like(y, np.nan, dtype=float)
    a = np.column_stack([np.ones(len(z), dtype=float), z.astype(float)])
    beta, *_ = np.linalg.lstsq(a, y.astype(float), rcond=None)
    return y.astype(float) - (a @ beta)


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    if len(x) < 4:
        return float("nan"), float("nan")
    xr = _residualize(x, z)
    yr = _residualize(y, z)
    return _safe_corr(xr, yr)


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


def _bootstrap_beta_magic(
    df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
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


def _independent_snapshot_study(
    states: np.ndarray,
    times: np.ndarray,
    n_sites: int,
    magic_m2: np.ndarray,
    cfg: NNQSConfig,
) -> pd.DataFrame:
    from tqm.nnqs.data import (
        all_bitstrings,
        sample_bitstrings_from_state,
        state_probabilities,
        total_variation_distance,
        train_val_split,
    )

    def _bernoulli_nll(bits: np.ndarray, probs: np.ndarray, eps: float = 1e-7) -> float:
        p = np.clip(probs, eps, 1.0 - eps)
        ll = bits * np.log(p[None, :]) + (1.0 - bits) * np.log(1.0 - p[None, :])
        return float(-np.mean(np.sum(ll, axis=1)))

    def _product_distribution(all_bits: np.ndarray, probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        p = np.clip(probs, eps, 1.0 - eps)
        logp = all_bits * np.log(p[None, :]) + (1.0 - all_bits) * np.log(1.0 - p[None, :])
        out = np.exp(np.sum(logp, axis=1))
        z = out.sum()
        return out / z if z > 0 else np.full_like(out, 1.0 / len(out))

    n_times = states.shape[0]
    snapshot_idx = np.unique(np.linspace(0, n_times - 1, cfg.snapshot_count, dtype=int))
    bits_all = all_bitstrings(n_sites).astype(float)

    rows: list[dict[str, float]] = []
    for rank, idx in enumerate(snapshot_idx):
        psi = states[idx]
        seed = int(cfg.seed + rank)
        samples = sample_bitstrings_from_state(
            psi=psi,
            n_sites=n_sites,
            num_samples=cfg.measurement_samples,
            seed=seed,
        ).astype(float)
        train_np, val_np = train_val_split(samples, val_fraction=cfg.val_fraction, seed=seed)
        p_hat = np.clip(np.mean(train_np, axis=0), 1e-6, 1.0 - 1e-6)
        train_nll = _bernoulli_nll(train_np, p_hat)
        val_nll = _bernoulli_nll(val_np, p_hat)
        p_model = _product_distribution(bits_all, p_hat)
        p_true = state_probabilities(psi)
        eps = 1e-15
        kl = float(np.sum(p_true * (np.log(p_true + eps) - np.log(p_model + eps))))
        tv = total_variation_distance(p_true, p_model)
        rows.append(
            {
                "snapshot_index": int(idx),
                "time": float(times[idx]),
                "magic_m2": float(magic_m2[idx]),
                "final_train_nll": train_nll,
                "final_val_nll": val_nll,
                "final_kl": kl,
                "final_tv": tv,
                "steps_to_threshold": float(1 if val_nll <= cfg.threshold_nll else -1),
            }
        )
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def _sigma(label: str) -> sp.csr_matrix:
    if label == "I":
        return sp.csr_matrix(np.eye(2, dtype=np.complex128))
    if label == "X":
        return sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=np.complex128))
    if label == "Y":
        return sp.csr_matrix(np.array([[0, -1j], [1j, 0]], dtype=np.complex128))
    if label == "Z":
        return sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=np.complex128))
    raise ValueError(f"Unknown Pauli label: {label}")


@lru_cache(maxsize=None)
def _local_pauli(n_sites: int, site: int, label: str) -> sp.csr_matrix:
    ops: list[sp.csr_matrix] = []
    for i in range(n_sites):
        ops.append(_sigma(label if i == site else "I"))
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csr")
    return out


@lru_cache(maxsize=None)
def _two_site_pauli(n_sites: int, i: int, j: int, label_i: str, label_j: str) -> sp.csr_matrix:
    ops: list[sp.csr_matrix] = []
    for site in range(n_sites):
        if site == i:
            ops.append(_sigma(label_i))
        elif site == j:
            ops.append(_sigma(label_j))
        else:
            ops.append(_sigma("I"))
    out = ops[0]
    for op in ops[1:]:
        out = sp.kron(out, op, format="csr")
    return out


def build_tfim_hamiltonian(n_sites: int, j: float, h: float, hz: float = 0.0) -> sp.csr_matrix:
    dim = 2**n_sites
    h_tot = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for i in range(n_sites - 1):
        h_tot = h_tot - j * _two_site_pauli(n_sites, i, i + 1, "Z", "Z")
    for i in range(n_sites):
        h_tot = h_tot - h * _local_pauli(n_sites, i, "X")
    if abs(hz) > 1e-15:
        for i in range(n_sites):
            h_tot = h_tot - hz * _local_pauli(n_sites, i, "Z")
    return h_tot.tocsr()


def _ground_state(h: sp.csr_matrix) -> np.ndarray:
    vals, vecs = spla.eigsh(h, k=1, which="SA")
    psi0 = vecs[:, 0]
    phase = np.exp(-1j * np.angle(psi0[np.argmax(np.abs(psi0))]))
    psi0 = psi0 * phase
    psi0 = psi0 / np.linalg.norm(psi0)
    _ = vals
    return psi0


@dataclass
class UniversalConfig:
    n_sites: list[int]
    h0: float
    h1_values: list[float]
    j: float
    hz: float
    t_max: float
    n_steps: int
    architectures: list[str]
    seeds: list[int]
    hidden_size: int
    epochs: int
    measurement_samples: int
    snapshot_count: int
    val_fraction: float
    lr: float
    batch_size: int
    threshold_nll: float
    bootstrap: int
    lambda_quantile: float


def _load_cfg(path: Path) -> UniversalConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return UniversalConfig(
        n_sites=[int(x) for x in payload.get("n_sites", [6, 8, 10])],
        h0=float(payload.get("h0", 0.2)),
        h1_values=[float(x) for x in payload.get("h1_values", [0.6, 1.0, 1.4])],
        j=float(payload.get("j", 1.0)),
        hz=float(payload.get("hz", 0.0)),
        t_max=float(payload.get("t_max", 2.0)),
        n_steps=int(payload.get("n_steps", 11)),
        architectures=[str(x) for x in payload.get("architectures", ["gru", "made", "independent"])],
        seeds=[int(x) for x in payload.get("seeds", [11, 17, 23, 29])],
        hidden_size=int(payload.get("hidden_size", 48)),
        epochs=int(payload.get("epochs", 35)),
        measurement_samples=int(payload.get("measurement_samples", 3200)),
        snapshot_count=int(payload.get("snapshot_count", 6)),
        val_fraction=float(payload.get("val_fraction", 0.2)),
        lr=float(payload.get("lr", 1e-3)),
        batch_size=int(payload.get("batch_size", 256)),
        threshold_nll=float(payload.get("threshold_nll", 4.0)),
        bootstrap=int(payload.get("bootstrap", 300)),
        lambda_quantile=float(payload.get("lambda_quantile", 0.4)),
    )


def _nnqs_cfg(cfg: UniversalConfig, seed: int) -> NNQSConfig:
    return NNQSConfig(
        enabled=True,
        snapshot_count=cfg.snapshot_count,
        measurement_samples=cfg.measurement_samples,
        val_fraction=cfg.val_fraction,
        hidden_size=cfg.hidden_size,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=cfg.batch_size,
        threshold_nll=cfg.threshold_nll,
        seed=seed,
    )


def _run_tfim_suite(cfg: UniversalConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    times = np.linspace(0.0, cfg.t_max, cfg.n_steps)
    cond_rows: list[dict[str, Any]] = []
    snap_rows: list[pd.DataFrame] = []
    quench_rows: list[dict[str, float]] = []

    cond_path = out_dir / "tfim_condition_stats.csv"
    snap_path = out_dir / "tfim_snapshot_all.csv"
    quench_path = out_dir / "tfim_quench_features.csv"

    done_ids: set[str] = set()
    if cond_path.exists():
        prev_cond = pd.read_csv(cond_path)
        if not prev_cond.empty:
            cond_rows.extend(prev_cond.to_dict(orient="records"))
            done_ids = set(prev_cond["condition_id"].astype(str).tolist())
    if snap_path.exists():
        prev_snap = pd.read_csv(snap_path)
        if not prev_snap.empty:
            snap_rows.append(prev_snap)
    if quench_path.exists():
        prev_quench = pd.read_csv(quench_path)
        if not prev_quench.empty:
            quench_rows.extend(prev_quench.to_dict(orient="records"))

    run_id = int(len(cond_rows))
    for n_sites in cfg.n_sites:
        for h1 in cfg.h1_values:
            cond_ids_this_quench = {
                f"tfim_N{n_sites}_h{h1:.3f}_{arch}_s{seed}"
                for arch in cfg.architectures
                for seed in cfg.seeds
            }
            if cond_ids_this_quench.issubset(done_ids):
                continue
            h0_mat = build_tfim_hamiltonian(n_sites=n_sites, j=cfg.j, h=cfg.h0, hz=cfg.hz)
            h1_mat = build_tfim_hamiltonian(n_sites=n_sites, j=cfg.j, h=h1, hz=cfg.hz)
            psi0 = _ground_state(h0_mat)
            states = evolve_state(h1_mat, psi0, times, method="krylov")
            lam = rate_function(psi0, states, n_sites=n_sites)
            entropy = bipartite_entropy_time(states, n_sites=n_sites)
            m2 = magic_over_time(states=states, n_sites=n_sites, alphas=[2.0])[2.0]
            max_lambda = float(np.max(lam))
            quench_key = (int(n_sites), float(h1))
            existing_quench = {
                (int(r["n_sites"]), float(r["quench_param"])) for r in quench_rows
            }
            if quench_key not in existing_quench:
                quench_rows.append(
                    {
                        "model_family": "tfim",
                        "n_sites": int(n_sites),
                        "quench_param": float(h1),
                        "quench_delta": float(h1 - cfg.h0),
                        "max_lambda": max_lambda,
                        "max_magic_m2": float(np.max(m2)),
                    }
                )

            for arch in cfg.architectures:
                for seed in cfg.seeds:
                    cond_id = f"tfim_N{n_sites}_h{h1:.3f}_{arch}_s{seed}"
                    if cond_id in done_ids:
                        continue
                    nn_cfg = _nnqs_cfg(cfg, seed=seed)
                    if arch in {"gru", "made"}:
                        df_snap, _ = run_snapshot_study(
                            states=states,
                            times=times,
                            n_sites=n_sites,
                            magic_m2=m2,
                            cfg=nn_cfg,
                            device="cpu",
                            model_type=arch,
                        )
                    elif arch == "independent":
                        df_snap = _independent_snapshot_study(
                            states=states,
                            times=times,
                            n_sites=n_sites,
                            magic_m2=m2,
                            cfg=nn_cfg,
                        )
                    else:
                        raise ValueError(f"Unsupported architecture in stage2 universal scan: {arch}")

                    df_snap = df_snap.copy()
                    df_snap["snapshot_entropy"] = entropy[df_snap["snapshot_index"].to_numpy(dtype=int)]
                    df_snap["condition_id"] = cond_id
                    df_snap["model_family"] = "tfim"
                    df_snap["architecture"] = arch
                    df_snap["n_sites"] = int(n_sites)
                    df_snap["quench_param"] = float(h1)
                    df_snap["quench_delta"] = float(h1 - cfg.h0)
                    df_snap["seed"] = int(seed)
                    df_snap["max_lambda"] = max_lambda
                    snap_rows.append(df_snap)

                    x = df_snap["magic_m2"].to_numpy(dtype=float)
                    y = df_snap["final_val_nll"].to_numpy(dtype=float)
                    z = df_snap["snapshot_entropy"].to_numpy(dtype=float)
                    r, p = _safe_corr(x, y)
                    r_partial, p_partial = _partial_corr(x, y, z)
                    cond_rows.append(
                        {
                            "run_id": int(run_id),
                            "condition_id": cond_id,
                            "model_family": "tfim",
                            "architecture": arch,
                            "n_sites": int(n_sites),
                            "quench_param": float(h1),
                            "quench_delta": float(h1 - cfg.h0),
                            "seed": int(seed),
                            "n_snapshots": int(len(df_snap)),
                            "max_lambda": max_lambda,
                            "pearson_magic_vs_val_nll": r,
                            "pearson_p_magic_vs_val_nll": p,
                            "pearson_magic_vs_val_nll_partial_entropy": r_partial,
                            "pearson_p_magic_vs_val_nll_partial_entropy": p_partial,
                        }
                    )
                    done_ids.add(cond_id)
                    run_id += 1
                    if run_id % 6 == 0:
                        cond_partial = pd.DataFrame(cond_rows)
                        snap_partial = pd.concat(snap_rows, axis=0, ignore_index=True) if snap_rows else pd.DataFrame()
                        quench_partial = pd.DataFrame(quench_rows)
                        cond_partial.to_csv(cond_path, index=False)
                        snap_partial.to_csv(snap_path, index=False)
                        quench_partial.to_csv(quench_path, index=False)
                        print(
                            f"[tfim checkpoint] conditions={len(cond_partial)} snapshots={len(snap_partial)}",
                            flush=True,
                        )

    cond_df = pd.DataFrame(cond_rows)
    snap_df = pd.concat(snap_rows, axis=0, ignore_index=True) if snap_rows else pd.DataFrame()
    quench_df = pd.DataFrame(quench_rows)

    cond_df.to_csv(cond_path, index=False)
    snap_df.to_csv(snap_path, index=False)
    quench_df.to_csv(quench_path, index=False)
    return cond_df, snap_df, quench_df


def _load_schwinger(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_specs = [
        ("baseline", ROOT / "outputs/stage1_prxq/baseline_n6n8_multiarch"),
        ("baseline_n10", ROOT / "outputs/stage1_prxq/n10_power_exact"),
        ("alt_baseline", ROOT / "outputs/stage1_prxq/alt_regime_baseline"),
        ("alt_heavy_mass", ROOT / "outputs/stage1_prxq/alt_regime_heavy_mass"),
        ("alt_strong_coupling", ROOT / "outputs/stage1_prxq/alt_regime_strong_coupling"),
    ]
    cond_list: list[pd.DataFrame] = []
    snap_list: list[pd.DataFrame] = []
    for regime, run_dir in run_specs:
        cond = pd.read_csv(run_dir / "nnqs_condition_stats.csv")
        snap = pd.read_csv(run_dir / "nnqs_snapshot_all.csv")
        q = pd.read_csv(run_dir / "quench_trend_features.csv")
        q = q.rename(columns={"theta1": "quench_param"})
        cond = cond.rename(columns={"theta1": "quench_param"})
        snap = snap.rename(columns={"theta1": "quench_param"})
        merge_cols = ["n_sites", "quench_param"]
        cond = cond.merge(q[merge_cols + ["max_lambda"]], on=merge_cols, how="left")
        snap = snap.merge(cond[["condition_id", "max_lambda"]], on="condition_id", how="left")
        cond["model_family"] = "schwinger"
        cond["regime"] = regime
        cond["quench_delta"] = cond["quench_param"]  # theta0 = 0
        snap["model_family"] = "schwinger"
        snap["regime"] = regime
        snap["quench_delta"] = snap["quench_param"]  # theta0 = 0
        cond_list.append(cond)
        snap_list.append(snap)
    cond_all = pd.concat(cond_list, ignore_index=True)
    snap_all = pd.concat(snap_list, ignore_index=True)
    cond_all.to_csv(out_dir / "schwinger_condition_stats.csv", index=False)
    snap_all.to_csv(out_dir / "schwinger_snapshot_all.csv", index=False)
    return cond_all, snap_all


def _corridor_filter(cond_df: pd.DataFrame, lambda_quantile: float) -> pd.DataFrame:
    out = cond_df.copy()
    out = out[np.isfinite(out["pearson_magic_vs_val_nll"])].copy()
    out = out[np.isfinite(out["pearson_magic_vs_val_nll_partial_entropy"])].copy()
    out = out[np.isfinite(out["max_lambda"])].copy()
    out = out[np.abs(out["quench_delta"]) > 1e-12].copy()
    keep_idx: list[int] = []
    for model, sub in out.groupby("model_family"):
        thresh = float(sub["max_lambda"].quantile(lambda_quantile))
        keep_idx.extend(sub.index[sub["max_lambda"] >= thresh].tolist())
        _ = model
    return out.loc[sorted(keep_idx)].reset_index(drop=True)


def _plot_pooled_effects(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    labels = [f"{m}\n{a}" for m, a in zip(df["model_family"], df["architecture"], strict=False)]
    y = df["pooled_r_partial"].to_numpy(dtype=float)
    lo = df["ci_low_partial"].to_numpy(dtype=float)
    hi = df["ci_high_partial"].to_numpy(dtype=float)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=4, color="#1259a7")
    ax.axhline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Pooled r (magic vs val NLL | entropy)")
    ax.set_title("Cross-model pooled primary effect (corridor-filtered)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_beta_bounds(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    labels = [f"{m}\n{a}" for m, a in zip(df["model_family"], df["architecture"], strict=False)]
    y = df["beta_magic"].to_numpy(dtype=float)
    lo = df["beta_magic_ci_low"].to_numpy(dtype=float)
    hi = df["beta_magic_ci_high"].to_numpy(dtype=float)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=4, color="#7a1f5c")
    ax.axhline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("beta_magic in valNLL ~ 1 + magic + entropy")
    ax.set_title("Statistical lower bounds on magic effect")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_finite_size(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for (model, arch), sub in df.groupby(["model_family", "architecture"]):
        if len(sub) < 2:
            continue
        x = 1.0 / sub["n_sites"].to_numpy(dtype=float)
        y = sub["pooled_r"].to_numpy(dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        coeff = np.polyfit(x, y, deg=1)
        xx = np.linspace(0.0, max(x), 80)
        yy = coeff[0] * xx + coeff[1]
        ax.plot(xx, yy, lw=1.2, alpha=0.8)
        ax.scatter(x, y, s=34, label=f"{model}-{arch}")
    ax.axhline(0.0, color="k", lw=1, alpha=0.5)
    ax.set_xlabel("1 / N")
    ax.set_ylabel("Pooled r")
    ax.set_title("Finite-size trend in corridor")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_pinsker(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    x = (2.0 * (df["final_tv"].to_numpy(dtype=float) ** 2)).astype(float)
    y = df["final_kl"].to_numpy(dtype=float).astype(float)
    lim = float(max(np.max(x), np.max(y), 1e-6))
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.scatter(x, y, s=8, alpha=0.35, color="#1f6f3b")
    ax.plot([0, lim], [0, lim], "k--", lw=1.0, alpha=0.7)
    ax.set_xlabel(r"$2\,TV^2$")
    ax.set_ylabel("KL")
    ax.set_title("Pinsker consistency check")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 universal-law scan with cross-model evidence and bounds")
    parser.add_argument("--config", type=str, default="configs/stage2_universal.yaml")
    parser.add_argument("--out-dir", type=str, default="outputs/stage2_prlx/universal_scan")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sch_cond, sch_snap = _load_schwinger(out_dir=out_dir)
    tf_cond, tf_snap, tf_quench = _run_tfim_suite(cfg=cfg, out_dir=out_dir)
    _ = tf_quench

    all_cond = pd.concat([sch_cond, tf_cond], ignore_index=True)
    all_snap = pd.concat([sch_snap, tf_snap], ignore_index=True)
    all_cond.to_csv(out_dir / "all_condition_stats.csv", index=False)
    all_snap.to_csv(out_dir / "all_snapshot_all.csv", index=False)

    corridor = _corridor_filter(all_cond, lambda_quantile=cfg.lambda_quantile)
    corridor_ids = set(corridor["condition_id"].tolist())
    snap_corridor = all_snap[all_snap["condition_id"].isin(corridor_ids)].copy().reset_index(drop=True)
    corridor.to_csv(out_dir / "corridor_condition_stats.csv", index=False)
    snap_corridor.to_csv(out_dir / "corridor_snapshot_all.csv", index=False)

    pooled_rows: list[dict[str, Any]] = []
    for (model, arch), sub in corridor.groupby(["model_family", "architecture"]):
        r_raw, lo_raw, hi_raw, n_raw = _fisher_meta(
            sub["pearson_magic_vs_val_nll"].to_numpy(dtype=float),
            sub["n_snapshots"].to_numpy(dtype=float),
        )
        r_partial, lo_partial, hi_partial, n_partial = _fisher_meta(
            sub["pearson_magic_vs_val_nll_partial_entropy"].to_numpy(dtype=float),
            sub["n_snapshots"].to_numpy(dtype=float),
        )
        pooled_rows.append(
            {
                "model_family": model,
                "architecture": arch,
                "pooled_r_raw": r_raw,
                "ci_low_raw": lo_raw,
                "ci_high_raw": hi_raw,
                "n_studies_raw": int(n_raw),
                "pooled_r_partial": r_partial,
                "ci_low_partial": lo_partial,
                "ci_high_partial": hi_partial,
                "n_conditions": int(len(sub)),
                "n_studies_partial": int(n_partial),
            }
        )
    pooled_df = pd.DataFrame(pooled_rows).sort_values(["model_family", "architecture"]).reset_index(drop=True)
    pooled_df.to_csv(out_dir / "cross_model_pooled_effects.csv", index=False)

    beta_rows: list[dict[str, Any]] = []
    for (model, arch), sub in snap_corridor.groupby(["model_family", "architecture"]):
        point, lo, hi = _bootstrap_beta_magic(sub, n_boot=cfg.bootstrap, seed=args.seed + len(beta_rows) * 13)
        beta_rows.append(
            {
                "model_family": model,
                "architecture": arch,
                "beta_magic": point,
                "beta_magic_ci_low": lo,
                "beta_magic_ci_high": hi,
                "positive_lower_bound": bool(np.isfinite(lo) and lo > 0.0),
                "n_snapshots": int(len(sub)),
            }
        )
    beta_df = pd.DataFrame(beta_rows).sort_values(["model_family", "architecture"]).reset_index(drop=True)
    beta_df.to_csv(out_dir / "cross_model_beta_bounds.csv", index=False)

    fs_rows: list[dict[str, Any]] = []
    by_size = corridor.groupby(["model_family", "architecture", "n_sites"], as_index=False).agg(
        pooled_r=("pearson_magic_vs_val_nll_partial_entropy", "mean"),
        n_conditions=("condition_id", "count"),
    )
    for (model, arch), sub in by_size.groupby(["model_family", "architecture"]):
        if len(sub) < 2:
            continue
        x = 1.0 / sub["n_sites"].to_numpy(dtype=float)
        y = sub["pooled_r"].to_numpy(dtype=float)
        coeff = np.polyfit(x, y, deg=1)
        fs_rows.append(
            {
                "model_family": model,
                "architecture": arch,
                "slope_vs_invN": float(coeff[0]),
                "r_inf_extrapolated": float(coeff[1]),
                "n_points": int(len(sub)),
            }
        )
    fs_df = pd.DataFrame(fs_rows).sort_values(["model_family", "architecture"]).reset_index(drop=True)
    fs_df.to_csv(out_dir / "finite_size_extrapolation.csv", index=False)

    pinsker_lhs = snap_corridor["final_kl"].to_numpy(dtype=float)
    pinsker_rhs = 2.0 * (snap_corridor["final_tv"].to_numpy(dtype=float) ** 2)
    pinsker_ok = bool(np.all(pinsker_lhs + 1e-12 >= pinsker_rhs))

    _plot_pooled_effects(pooled_df, fig_dir / "fig_stage2_crossmodel_pooled_r.png")
    _plot_beta_bounds(beta_df, fig_dir / "fig_stage2_beta_bounds.png")
    _plot_finite_size(by_size, fig_dir / "fig_stage2_finite_size.png")
    _plot_pinsker(snap_corridor, fig_dir / "fig_stage2_pinsker_bound.png")

    universal_sign_ok = bool(
        (len(pooled_df) > 0)
        and np.all(np.isfinite(pooled_df["ci_low_partial"].to_numpy(dtype=float)))
        and np.all(pooled_df["ci_low_partial"].to_numpy(dtype=float) > 0.0)
    )
    universal_beta_ok = bool(
        (len(beta_df) > 0)
        and np.all(beta_df["positive_lower_bound"].to_numpy(dtype=bool))
    )
    universal_beta_claim = bool(universal_beta_ok and pinsker_ok)
    universal_corridor_claim = bool(universal_sign_ok and universal_beta_ok and pinsker_ok)

    expressive_arch = {"gru", "made"}
    pooled_expr = pooled_df[pooled_df["architecture"].isin(expressive_arch)].copy()
    beta_expr = beta_df[beta_df["architecture"].isin(expressive_arch)].copy()
    expressive_sign_ok = bool(
        (len(pooled_expr) > 0)
        and np.all(np.isfinite(pooled_expr["ci_low_partial"].to_numpy(dtype=float)))
        and np.all(pooled_expr["ci_low_partial"].to_numpy(dtype=float) > 0.0)
    )
    expressive_beta_ok = bool(
        (len(beta_expr) > 0)
        and np.all(beta_expr["positive_lower_bound"].to_numpy(dtype=bool))
    )
    universal_expressive_beta_claim = bool(expressive_beta_ok and pinsker_ok)
    universal_expressive_claim = bool(expressive_sign_ok and expressive_beta_ok and pinsker_ok)

    summary = {
        "config": args.config,
        "out_dir": str(out_dir),
        "n_conditions_all": int(len(all_cond)),
        "n_conditions_corridor": int(len(corridor)),
        "n_snapshots_corridor": int(len(snap_corridor)),
        "lambda_quantile": float(cfg.lambda_quantile),
        "universal_sign_ok": universal_sign_ok,
        "universal_beta_ok": universal_beta_ok,
        "pinsker_ok": pinsker_ok,
        "universal_beta_claim": universal_beta_claim,
        "universal_corridor_claim": universal_corridor_claim,
        "expressive_architectures": sorted(expressive_arch),
        "expressive_sign_ok": expressive_sign_ok,
        "expressive_beta_ok": expressive_beta_ok,
        "universal_expressive_beta_claim": universal_expressive_beta_claim,
        "universal_expressive_claim": universal_expressive_claim,
        "artifacts": {
            "pooled_effects_csv": str(out_dir / "cross_model_pooled_effects.csv"),
            "beta_bounds_csv": str(out_dir / "cross_model_beta_bounds.csv"),
            "finite_size_csv": str(out_dir / "finite_size_extrapolation.csv"),
            "fig_pooled": str(fig_dir / "fig_stage2_crossmodel_pooled_r.png"),
            "fig_beta": str(fig_dir / "fig_stage2_beta_bounds.png"),
            "fig_finite_size": str(fig_dir / "fig_stage2_finite_size.png"),
            "fig_pinsker": str(fig_dir / "fig_stage2_pinsker_bound.png"),
        },
    }
    with (out_dir / "universal_law_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Stage 2 Universal-Law Scan",
        "",
        "This report evaluates a cross-model corridor claim using Schwinger + TFIM data.",
        "",
        f"- total conditions analyzed: `{len(all_cond)}`",
        f"- corridor-filtered conditions: `{len(corridor)}` (lambda quantile `{cfg.lambda_quantile}`)",
        f"- corridor snapshots: `{len(snap_corridor)}`",
        "",
        "## Corridor Claim Gate",
        f"- sign-consistency gate (CI low > 0 for pooled primary r): `{universal_sign_ok}`",
        f"- beta-lower-bound gate (95% CI low > 0): `{universal_beta_ok}`",
        f"- Pinsker sanity gate (KL >= 2 TV^2): `{pinsker_ok}`",
        f"- **universal beta-law claim**: `{universal_beta_claim}`",
        f"- **universal corridor claim**: `{universal_corridor_claim}`",
        "",
        "## Expressive-Model Gate (GRU + MADE)",
        f"- expressive sign gate: `{expressive_sign_ok}`",
        f"- expressive beta gate: `{expressive_beta_ok}`",
        f"- universal expressive beta-law claim: `{universal_expressive_beta_claim}`",
        f"- **universal expressive-model corridor claim**: `{universal_expressive_claim}`",
        "",
        "Interpretation:",
        "- The beta-law can hold while pooled-correlation sign fails due architecture/model heterogeneity.",
        "- If all-architecture correlation gate fails but expressive gate passes, the strongest sign-stable claim is an expressive-NNQS corridor law.",
        "",
        "## Files",
        f"- `{out_dir / 'cross_model_pooled_effects.csv'}`",
        f"- `{out_dir / 'cross_model_beta_bounds.csv'}`",
        f"- `{out_dir / 'finite_size_extrapolation.csv'}`",
        f"- `{fig_dir / 'fig_stage2_crossmodel_pooled_r.png'}`",
        f"- `{fig_dir / 'fig_stage2_beta_bounds.png'}`",
        f"- `{fig_dir / 'fig_stage2_finite_size.png'}`",
        f"- `{fig_dir / 'fig_stage2_pinsker_bound.png'}`",
    ]
    (ROOT / "report" / "stage2_universal_law.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'universal_law_summary.json'}")
    print(f"Wrote {ROOT / 'report' / 'stage2_universal_law.md'}")


if __name__ == "__main__":
    main()
