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

from tqm.config import NNQSConfig
from tqm.evolve import evolve_state
from tqm.loschmidt import rate_function
from tqm.magic import magic_over_time
from tqm.observables import bipartite_entropy_time
from tqm.pauli import pauli_expectation_batch_for_x

# Reuse the independent baseline and stage2 helpers.
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


def _approximate_magic_m2_state(psi: np.ndarray, n_sites: int, n_pauli_samples: int, seed: int) -> float:
    if n_pauli_samples <= 0:
        raise ValueError("n_pauli_samples must be > 0")
    n_masks = 2**n_sites
    rng = np.random.default_rng(seed)
    x_samples = rng.integers(0, n_masks, size=n_pauli_samples, dtype=np.uint32)
    z_samples = rng.integers(0, n_masks, size=n_pauli_samples, dtype=np.uint32)
    c4_vals = np.empty(n_pauli_samples, dtype=np.float64)

    order = np.argsort(x_samples)
    x_sorted = x_samples[order]
    z_sorted = z_samples[order]
    start = 0
    while start < n_pauli_samples:
        x = int(x_sorted[start])
        end = start + 1
        while end < n_pauli_samples and int(x_sorted[end]) == x:
            end += 1
        z_batch = z_sorted[start:end]
        exps = pauli_expectation_batch_for_x(psi=psi, x_mask=x, z_masks=z_batch)
        c4_vals[order[start:end]] = np.abs(exps) ** 4
        start = end

    mean_c4 = max(float(np.mean(c4_vals)), 1e-15)
    d = float(2**n_sites)
    return float(-np.log(mean_c4) - np.log(d))


def _approximate_magic_m2_over_time(
    states: np.ndarray,
    n_sites: int,
    n_pauli_samples: int,
    seed_base: int,
) -> np.ndarray:
    out = np.empty(states.shape[0], dtype=np.float64)
    for t_idx, psi in enumerate(states):
        out[t_idx] = _approximate_magic_m2_state(
            psi=psi,
            n_sites=n_sites,
            n_pauli_samples=n_pauli_samples,
            seed=seed_base + t_idx,
        )
    return out


def _perm_p_beta_magic(df: pd.DataFrame, n_perm: int, seed: int) -> float:
    x = df["magic_m2"].to_numpy(dtype=float)
    s = df["snapshot_entropy"].to_numpy(dtype=float)
    y = df["final_val_nll"].to_numpy(dtype=float)
    cond = df["condition_id"].astype(str).to_numpy()
    if len(df) < 10 or n_perm <= 0:
        return float("nan")

    a = np.column_stack([np.ones(len(df)), x, s])
    beta_obs, *_ = np.linalg.lstsq(a, y, rcond=None)
    b_obs = float(beta_obs[1])

    groups: dict[str, np.ndarray] = {}
    for cid in np.unique(cond):
        groups[cid] = np.where(cond == cid)[0]

    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        xp = x.copy()
        for idx in groups.values():
            xp[idx] = xp[rng.permutation(idx)]
        ap = np.column_stack([np.ones(len(df)), xp, s])
        beta_p, *_ = np.linalg.lstsq(ap, y, rcond=None)
        if float(beta_p[1]) >= b_obs:
            ge += 1
    return float((ge + 1) / (n_perm + 1))


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    q = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if not np.any(mask):
        return q
    p = pvals[mask]
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q_ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        k = i + 1
        val = min(prev, ranked[i] * n / k)
        q_ranked[i] = val
        prev = val
    q_masked = np.empty(n, dtype=float)
    q_masked[order] = q_ranked
    q[mask] = q_masked
    return q


def _meta_i2(effect: np.ndarray, ci_low: np.ndarray, ci_high: np.ndarray) -> float:
    mask = np.isfinite(effect) & np.isfinite(ci_low) & np.isfinite(ci_high) & (ci_high > ci_low)
    if np.sum(mask) < 2:
        return float("nan")
    e = effect[mask]
    se = (ci_high[mask] - ci_low[mask]) / 3.92
    se = np.clip(se, 1e-9, None)
    w = 1.0 / (se**2)
    e_bar = float(np.sum(w * e) / np.sum(w))
    q = float(np.sum(w * (e - e_bar) ** 2))
    df = len(e) - 1
    if q <= 0.0:
        return 0.0
    return float(max(0.0, (q - df) / q))


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


def build_xxz_hamiltonian(n_sites: int, jxy: float, delta: float, hx: float, hz: float = 0.0) -> sp.csr_matrix:
    dim = 2**n_sites
    h_tot = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for i in range(n_sites - 1):
        h_tot = h_tot + jxy * _two_site_pauli(n_sites, i, i + 1, "X", "X")
        h_tot = h_tot + jxy * _two_site_pauli(n_sites, i, i + 1, "Y", "Y")
        h_tot = h_tot + (jxy * delta) * _two_site_pauli(n_sites, i, i + 1, "Z", "Z")
    for i in range(n_sites):
        h_tot = h_tot + hx * _local_pauli(n_sites, i, "X")
    if abs(hz) > 1e-15:
        for i in range(n_sites):
            h_tot = h_tot + hz * _local_pauli(n_sites, i, "Z")
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
class Stage3Config:
    stage2_base_dir: str
    out_dir: str
    n_sites: list[int]
    delta_values: list[float]
    h0: float
    h1_values: list[float]
    jxy: float
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
    permutations: int
    lambda_quantile: float
    fdr_alpha: float
    exact_magic_max_sites: int
    approx_magic_samples: int
    base_architectures: list[str]


def _load_cfg(path: Path, out_override: str | None) -> Stage3Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out_dir = str(out_override) if out_override else str(payload.get("out_dir", "outputs/stage3_prx/universal_extension_v1"))
    return Stage3Config(
        stage2_base_dir=str(payload.get("stage2_base_dir", "outputs/stage2_prlx/universal_scan_v2")),
        out_dir=out_dir,
        n_sites=[int(x) for x in payload.get("n_sites", [6, 8])],
        delta_values=[float(x) for x in payload.get("delta_values", [0.6, 1.0, 1.4])],
        h0=float(payload.get("h0", 0.2)),
        h1_values=[float(x) for x in payload.get("h1_values", [0.6, 1.0, 1.4])],
        jxy=float(payload.get("jxy", 1.0)),
        hz=float(payload.get("hz", 0.0)),
        t_max=float(payload.get("t_max", 2.0)),
        n_steps=int(payload.get("n_steps", 11)),
        architectures=[str(x) for x in payload.get("architectures", ["gru", "made", "independent"])],
        seeds=[int(x) for x in payload.get("seeds", [11, 17, 23])],
        hidden_size=int(payload.get("hidden_size", 48)),
        epochs=int(payload.get("epochs", 30)),
        measurement_samples=int(payload.get("measurement_samples", 2800)),
        snapshot_count=int(payload.get("snapshot_count", 6)),
        val_fraction=float(payload.get("val_fraction", 0.2)),
        lr=float(payload.get("lr", 1e-3)),
        batch_size=int(payload.get("batch_size", 256)),
        threshold_nll=float(payload.get("threshold_nll", 4.0)),
        bootstrap=int(payload.get("bootstrap", 250)),
        permutations=int(payload.get("permutations", 250)),
        lambda_quantile=float(payload.get("lambda_quantile", 0.4)),
        fdr_alpha=float(payload.get("fdr_alpha", 0.05)),
        exact_magic_max_sites=int(payload.get("exact_magic_max_sites", 10)),
        approx_magic_samples=int(payload.get("approx_magic_samples", 0)),
        base_architectures=[str(x) for x in payload.get("base_architectures", [])],
    )


def _nnqs_cfg(cfg: Stage3Config, seed: int) -> NNQSConfig:
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


def _run_xxz_suite(cfg: Stage3Config, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tqm.nnqs.train import run_snapshot_study

    times = np.linspace(0.0, cfg.t_max, cfg.n_steps)
    cond_rows: list[dict[str, Any]] = []
    snap_rows: list[pd.DataFrame] = []

    cond_path = out_dir / "xxz_condition_stats.csv"
    snap_path = out_dir / "xxz_snapshot_all.csv"
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

    run_id = int(len(cond_rows))
    for n_sites in cfg.n_sites:
        for delta in cfg.delta_values:
            h0_mat = build_xxz_hamiltonian(n_sites=n_sites, jxy=cfg.jxy, delta=delta, hx=cfg.h0, hz=cfg.hz)
            psi0 = _ground_state(h0_mat)
            for h1 in cfg.h1_values:
                h1_mat = build_xxz_hamiltonian(n_sites=n_sites, jxy=cfg.jxy, delta=delta, hx=h1, hz=cfg.hz)
                states = evolve_state(h1_mat, psi0, times, method="krylov")
                lam = rate_function(psi0, states, n_sites=n_sites)
                entropy = bipartite_entropy_time(states, n_sites=n_sites)
                if n_sites <= cfg.exact_magic_max_sites:
                    m2 = magic_over_time(states=states, n_sites=n_sites, alphas=[2.0])[2.0]
                elif cfg.approx_magic_samples > 0:
                    seed_base = int(1000 * n_sites + 100 * round(10 * delta) + round(100 * h1))
                    m2 = _approximate_magic_m2_over_time(
                        states=states,
                        n_sites=n_sites,
                        n_pauli_samples=cfg.approx_magic_samples,
                        seed_base=seed_base,
                    )
                else:
                    raise ValueError(
                        f"No magic path for n_sites={n_sites}. Set exact_magic_max_sites >= {n_sites} or approx_magic_samples > 0."
                    )
                max_lambda = float(np.max(lam))
                for arch in cfg.architectures:
                    for seed in cfg.seeds:
                        cond_id = f"xxz_N{n_sites}_d{delta:.3f}_h{h1:.3f}_{arch}_s{seed}"
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
                                device=args.device,
                                model_type=arch,
                            )
                        elif arch == "independent":
                            df_snap = _independent_snapshot_study(states=states, times=times, n_sites=n_sites, magic_m2=m2, cfg=nn_cfg)
                        else:
                            raise ValueError(f"Unsupported architecture for stage3 extension: {arch}")
                        df_snap = df_snap.copy()
                        df_snap["snapshot_entropy"] = entropy[df_snap["snapshot_index"].to_numpy(dtype=int)]
                        df_snap["condition_id"] = cond_id
                        df_snap["model_family"] = "xxz"
                        df_snap["regime"] = "stage3_xxz"
                        df_snap["architecture"] = arch
                        df_snap["n_sites"] = int(n_sites)
                        df_snap["quench_param"] = float(h1)
                        df_snap["quench_delta"] = float(h1 - cfg.h0)
                        df_snap["seed"] = int(seed)
                        df_snap["hidden_size"] = float(cfg.hidden_size)
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
                                "architecture": arch,
                                "n_sites": int(n_sites),
                                "quench_param": float(h1),
                                "hidden_size": float(cfg.hidden_size),
                                "seed": int(seed),
                                "n_snapshots": int(len(df_snap)),
                                "pearson_magic_vs_val_nll": r,
                                "pearson_p_magic_vs_val_nll": p,
                                "pearson_magic_vs_val_nll_partial_entropy": r_partial,
                                "pearson_p_magic_vs_val_nll_partial_entropy": p_partial,
                                "max_lambda": max_lambda,
                                "model_family": "xxz",
                                "regime": "stage3_xxz",
                                "quench_delta": float(h1 - cfg.h0),
                            }
                        )
                        done_ids.add(cond_id)
                        run_id += 1
                        if run_id % 6 == 0:
                            pd.DataFrame(cond_rows).to_csv(cond_path, index=False)
                            pd.concat(snap_rows, axis=0, ignore_index=True).to_csv(snap_path, index=False)
                            print(f"[xxz checkpoint] conditions={len(cond_rows)}", flush=True)

    cond_df = pd.DataFrame(cond_rows)
    snap_df = pd.concat(snap_rows, axis=0, ignore_index=True) if snap_rows else pd.DataFrame()
    cond_df.to_csv(cond_path, index=False)
    snap_df.to_csv(snap_path, index=False)
    return cond_df, snap_df


def _corridor_filter(cond_df: pd.DataFrame, lambda_quantile: float) -> pd.DataFrame:
    out = cond_df.copy()
    out = out[np.isfinite(out["pearson_magic_vs_val_nll_partial_entropy"])].copy()
    out = out[np.isfinite(out["max_lambda"])].copy()
    out = out[np.abs(out["quench_delta"]) > 1e-12].copy()
    keep_idx: list[int] = []
    for _, sub in out.groupby("model_family"):
        thresh = float(sub["max_lambda"].quantile(lambda_quantile))
        keep_idx.extend(sub.index[sub["max_lambda"] >= thresh].tolist())
    return out.loc[sorted(keep_idx)].reset_index(drop=True)


def _plot_beta_forest(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    labels = [f"{m}\n{a}" for m, a in zip(df["model_family"], df["architecture"], strict=False)]
    x = np.arange(len(df))
    y = df["beta_magic"].to_numpy(dtype=float)
    lo = df["beta_magic_ci_low"].to_numpy(dtype=float)
    hi = df["beta_magic_ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.4, 5.1))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=4, color="#114e8a")
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("beta_magic (95% CI)")
    ax.set_title("Stage-3 beta-law stress across model families")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_qvalues(df: pd.DataFrame, path: Path, alpha: float) -> None:
    if df.empty:
        return
    labels = [f"{m}\n{a}" for m, a in zip(df["model_family"], df["architecture"], strict=False)]
    x = np.arange(len(df))
    y = df["perm_q_beta_magic"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.4, 5.1))
    ax.bar(x, y, color="#7a1f5c", alpha=0.9)
    ax.axhline(alpha, color="k", ls="--", lw=1.2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("BH-FDR q-value")
    ax.set_title("Stage-3 null-disproof q-values for beta_magic")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_partial_heatmap(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    fam = sorted(df["model_family"].unique().tolist())
    arch = sorted(df["architecture"].unique().tolist())
    mat = np.full((len(fam), len(arch)), np.nan, dtype=float)
    for i, f in enumerate(fam):
        for j, a in enumerate(arch):
            sub = df[(df["model_family"] == f) & (df["architecture"] == a)]
            if len(sub) == 0:
                continue
            mat[i, j] = float(sub["ci_low_partial"].iloc[0])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(arch)))
    ax.set_xticklabels(arch)
    ax.set_yticks(np.arange(len(fam)))
    ax.set_yticklabels(fam)
    ax.set_title("Lower CI of partial-correlation endpoint")
    for i in range(len(fam)):
        for j in range(len(arch)):
            v = mat[i, j]
            txt = "NA" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.86)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 cross-model extension with adversarial null-disproof tests")
    parser.add_argument("--config", type=str, default="configs/stage3_universal.yaml")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--registry-out", type=str, default="outputs/stage3_prx/stage3/effect_registry.json")
    parser.add_argument("--report-out", type=str, default="report/stage3_extension.md")
    parser.add_argument("--device", type=str, default="auto", help="NNQS device: auto, cpu, cuda, cuda:0, mps")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config), out_override=args.out_dir)
    out_dir = Path(cfg.out_dir)
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_dir = ROOT / cfg.stage2_base_dir
    base_cond = pd.read_csv(base_dir / "all_condition_stats.csv")
    base_snap = pd.read_csv(base_dir / "all_snapshot_all.csv")
    if cfg.base_architectures:
        base_cond = base_cond[base_cond["architecture"].astype(str).isin(cfg.base_architectures)].copy()
        base_snap = base_snap[base_snap["architecture"].astype(str).isin(cfg.base_architectures)].copy()

    xxz_cond, xxz_snap = _run_xxz_suite(cfg=cfg, out_dir=out_dir)
    all_cond = pd.concat([base_cond, xxz_cond], ignore_index=True, sort=False)
    all_snap = pd.concat([base_snap, xxz_snap], ignore_index=True, sort=False)
    all_cond.to_csv(out_dir / "all_condition_stats_stage3.csv", index=False)
    all_snap.to_csv(out_dir / "all_snapshot_all_stage3.csv", index=False)

    corridor = _corridor_filter(all_cond, lambda_quantile=cfg.lambda_quantile)
    corridor_ids = set(corridor["condition_id"].astype(str).tolist())
    snap_corridor = all_snap[all_snap["condition_id"].astype(str).isin(corridor_ids)].copy().reset_index(drop=True)
    corridor.to_csv(out_dir / "corridor_condition_stats_stage3.csv", index=False)
    snap_corridor.to_csv(out_dir / "corridor_snapshot_all_stage3.csv", index=False)

    rows: list[dict[str, Any]] = []
    for idx, ((model, arch), sub) in enumerate(corridor.groupby(["model_family", "architecture"])):
        r_partial, lo_partial, hi_partial, n_studies = _fisher_meta(
            sub["pearson_magic_vs_val_nll_partial_entropy"].to_numpy(dtype=float),
            sub["n_snapshots"].to_numpy(dtype=float),
        )
        sub_snap = snap_corridor[(snap_corridor["model_family"] == model) & (snap_corridor["architecture"] == arch)].copy()
        beta, beta_lo, beta_hi = _bootstrap_beta_magic(sub_snap, n_boot=cfg.bootstrap, seed=args.seed + idx * 17)
        p_perm = _perm_p_beta_magic(sub_snap, n_perm=cfg.permutations, seed=args.seed + idx * 23)
        rows.append(
            {
                "model_family": model,
                "architecture": arch,
                "n_conditions": int(len(sub)),
                "n_snapshots": int(len(sub_snap)),
                "pooled_r_partial": r_partial,
                "ci_low_partial": lo_partial,
                "ci_high_partial": hi_partial,
                "n_studies_partial": int(n_studies),
                "beta_magic": beta,
                "beta_magic_ci_low": beta_lo,
                "beta_magic_ci_high": beta_hi,
                "beta_ci_positive": bool(np.isfinite(beta_lo) and beta_lo > 0.0),
                "perm_p_beta_magic": p_perm,
            }
        )
    cell_df = pd.DataFrame(rows).sort_values(["model_family", "architecture"]).reset_index(drop=True)
    cell_df["perm_q_beta_magic"] = _bh_fdr(cell_df["perm_p_beta_magic"].to_numpy(dtype=float))
    cell_df["perm_q_pass"] = cell_df["perm_q_beta_magic"].to_numpy(dtype=float) < cfg.fdr_alpha
    cell_df.to_csv(out_dir / "stage3_cell_metrics.csv", index=False)

    _plot_beta_forest(cell_df, fig_dir / "fig_stage3_beta_forest.png")
    _plot_qvalues(cell_df, fig_dir / "fig_stage3_perm_qvalues.png", alpha=cfg.fdr_alpha)
    _plot_partial_heatmap(cell_df, fig_dir / "fig_stage3_partial_sign_heatmap.png")

    pinsker_ok = bool(
        np.all(
            snap_corridor["final_kl"].to_numpy(dtype=float) + 1e-12
            >= 2.0 * (snap_corridor["final_tv"].to_numpy(dtype=float) ** 2)
        )
    )
    beta_ci_ok = bool(np.all(cell_df["beta_ci_positive"].to_numpy(dtype=bool)))
    beta_fdr_ok = bool(np.all(cell_df["perm_q_pass"].to_numpy(dtype=bool)))
    sign_ok = bool(
        np.all(np.isfinite(cell_df["ci_low_partial"].to_numpy(dtype=float)))
        and np.all(cell_df["ci_low_partial"].to_numpy(dtype=float) > 0.0)
    )
    beta_i2 = _meta_i2(
        effect=cell_df["beta_magic"].to_numpy(dtype=float),
        ci_low=cell_df["beta_magic_ci_low"].to_numpy(dtype=float),
        ci_high=cell_df["beta_magic_ci_high"].to_numpy(dtype=float),
    )
    partial_i2 = _meta_i2(
        effect=cell_df["pooled_r_partial"].to_numpy(dtype=float),
        ci_low=cell_df["ci_low_partial"].to_numpy(dtype=float),
        ci_high=cell_df["ci_high_partial"].to_numpy(dtype=float),
    )

    universal_beta_law = bool(beta_ci_ok and beta_fdr_ok and pinsker_ok)
    universal_sign_law = bool(sign_ok and beta_ci_ok and pinsker_ok)

    summary = {
        "config": args.config,
        "out_dir": str(out_dir),
        "n_conditions_all_stage3": int(len(all_cond)),
        "n_conditions_corridor_stage3": int(len(corridor)),
        "n_snapshots_corridor_stage3": int(len(snap_corridor)),
        "lambda_quantile": float(cfg.lambda_quantile),
        "fdr_alpha": float(cfg.fdr_alpha),
        "pinsker_ok": pinsker_ok,
        "beta_ci_ok": beta_ci_ok,
        "beta_fdr_ok": beta_fdr_ok,
        "sign_ok": sign_ok,
        "beta_i2": beta_i2,
        "partial_i2": partial_i2,
        "universal_beta_law": universal_beta_law,
        "universal_sign_law": universal_sign_law,
    }
    (out_dir / "stage3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    registry = {
        "schema_version": "1.0",
        "study_id": "stage3_prx",
        "phase": "stage3",
        "source_stage2_base_dir": str(base_dir),
        "source_out_dir": str(out_dir),
        "gate_flags": {
            "pinsker_ok": pinsker_ok,
            "beta_ci_ok": beta_ci_ok,
            "beta_fdr_ok": beta_fdr_ok,
            "sign_ok": sign_ok,
            "universal_beta_law": universal_beta_law,
            "universal_sign_law": universal_sign_law,
        },
        "heterogeneity": {
            "beta_i2": beta_i2,
            "partial_i2": partial_i2,
        },
        "records": cell_df.to_dict(orient="records"),
        "interpretation": {
            "extended_universal_beta_law": "supported" if universal_beta_law else "unsupported",
            "extended_universal_sign_law": "supported" if universal_sign_law else "unsupported",
        },
    }
    reg_path = ROOT / args.registry_out
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    lines = [
        "# Stage 3 Universal-Law Extension",
        "",
        "Cross-model extension adds XXZ quenches on top of Stage 2 (Schwinger + TFIM).",
        "",
        f"- all conditions: `{len(all_cond)}`",
        f"- corridor conditions: `{len(corridor)}`",
        f"- corridor snapshots: `{len(snap_corridor)}`",
        f"- lambda quantile: `{cfg.lambda_quantile}`",
        "",
        "## Null-disproof gates",
        f"- beta CI gate (`beta_ci_low > 0` in all cells): `{beta_ci_ok}`",
        f"- permutation FDR gate (`q < {cfg.fdr_alpha}` in all cells): `{beta_fdr_ok}`",
        f"- Pinsker gate: `{pinsker_ok}`",
        f"- **extended universal beta-law**: `{universal_beta_law}`",
        "",
        "## Sign-law gate",
        f"- pooled partial-correlation sign gate (`ci_low_partial > 0` all cells): `{sign_ok}`",
        f"- **extended universal sign-law**: `{universal_sign_law}`",
        "",
        "## Heterogeneity",
        f"- beta-law I2: `{beta_i2:.4f}`",
        f"- partial-correlation I2: `{partial_i2:.4f}`",
        "",
        "## Artifacts",
        f"- `{out_dir / 'stage3_cell_metrics.csv'}`",
        f"- `{out_dir / 'stage3_summary.json'}`",
        f"- `{fig_dir / 'fig_stage3_beta_forest.png'}`",
        f"- `{fig_dir / 'fig_stage3_perm_qvalues.png'}`",
        f"- `{fig_dir / 'fig_stage3_partial_sign_heatmap.png'}`",
        f"- `{reg_path}`",
    ]
    report_path = ROOT / args.report_out
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_dir / 'stage3_summary.json'}")
    print(f"Wrote {out_dir / 'stage3_cell_metrics.csv'}")
    print(f"Wrote {reg_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
