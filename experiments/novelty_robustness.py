from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = str((Path.cwd() / ".mplconfig").resolve())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, pearsonr, spearmanr
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqm.config import load_config
from tqm.nnqs.data import (
    all_bitstrings,
    sample_bitstrings_from_state,
    state_probabilities,
    total_variation_distance,
    train_val_split,
)
from tqm.nnqs.train import run_snapshot_study
from tqm.pauli import pauli_expectation_batch_for_x
from tqm.pipeline import QuenchRun, run_single_quench


@dataclass
class MetaResult:
    metric: str
    pooled_r: float
    ci_low: float
    ci_high: float
    i2: float
    n_studies: int
    positive_fraction: float
    significant_fraction: float
    combined_p: float
    combined_q: float
    verdict: str


def _parse_float_list(raw: str | None, default: list[float]) -> list[float]:
    if raw is None or raw.strip() == "":
        return [float(x) for x in default]
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str | None, default: list[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return [int(x) for x in default]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _safe_corr(x: np.ndarray, y: np.ndarray, fn: Callable[[np.ndarray, np.ndarray], tuple[float, float]]) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan"), float("nan")
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return float("nan"), float("nan")
    stat, p = fn(x, y)
    return float(stat), float(p)


def _bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    corr_kind: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(x)
    vals: list[float] = []

    if corr_kind == "pearson":
        corr_fn = lambda a, b: _safe_corr(a, b, pearsonr)[0]
    elif corr_kind == "spearman":
        corr_fn = lambda a, b: _safe_corr(a, b, spearmanr)[0]
    else:
        raise ValueError(f"Unsupported corr_kind: {corr_kind}")

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = corr_fn(x[idx], y[idx])
        if np.isfinite(c):
            vals.append(float(c))

    if not vals:
        return float("nan"), float("nan")

    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def _permutation_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    corr_kind: str,
    n_perm: int,
    seed: int,
) -> float:
    if len(x) < 3:
        return float("nan")
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return float("nan")

    if corr_kind == "pearson":
        corr_fn = lambda a, b: _safe_corr(a, b, pearsonr)[0]
    elif corr_kind == "spearman":
        corr_fn = lambda a, b: _safe_corr(a, b, spearmanr)[0]
    else:
        raise ValueError(f"Unsupported corr_kind: {corr_kind}")

    obs = corr_fn(x, y)
    if not np.isfinite(obs):
        return float("nan")

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        yp = np.array(y, copy=True)
        rng.shuffle(yp)
        cp = corr_fn(x, yp)
        if np.isfinite(cp) and abs(cp) >= abs(obs):
            exceed += 1

    return float((exceed + 1) / (n_perm + 1))


def _null_baseline_corrs(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_repeats: int = 64,
) -> tuple[float, float]:
    """Return average random-label and shuffled-time baseline correlations."""
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    random_label_vals: list[float] = []
    shuffled_time_vals: list[float] = []

    base_idx = np.arange(len(x))
    for _ in range(n_repeats):
        idx_y = np.array(base_idx, copy=True)
        idx_x = np.array(base_idx, copy=True)
        rng.shuffle(idx_y)
        rng.shuffle(idx_x)
        c1, _ = _safe_corr(x, y[idx_y], pearsonr)
        c2, _ = _safe_corr(x[idx_x], y, pearsonr)
        if np.isfinite(c1):
            random_label_vals.append(float(c1))
        if np.isfinite(c2):
            shuffled_time_vals.append(float(c2))

    random_label_mean = float(np.mean(random_label_vals)) if random_label_vals else float("nan")
    shuffled_time_mean = float(np.mean(shuffled_time_vals)) if shuffled_time_vals else float("nan")
    return random_label_mean, shuffled_time_mean


def _residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return residuals of y after linear regression on [1, z]."""
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
    """Partial corr(x,y|z) via residualization with linear model in z."""
    if len(x) < 4 or len(y) < 4 or len(z) < 4:
        return float("nan"), float("nan")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        return float("nan"), float("nan")

    xr = _residualize(x, z)
    yr = _residualize(y, z)
    return _safe_corr(xr, yr, pearsonr)


def _bootstrap_partial_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if len(x) < 4:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(x)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c, _ = _partial_corr(x[idx], y[idx], z[idx])
        if np.isfinite(c):
            vals.append(float(c))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def _permutation_partial_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n_perm: int,
    seed: int,
) -> float:
    obs, _ = _partial_corr(x, y, z)
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        yp = np.array(y, copy=True)
        rng.shuffle(yp)
        cp, _ = _partial_corr(x, yp, z)
        if np.isfinite(cp) and abs(cp) >= abs(obs):
            exceed += 1
    return float((exceed + 1) / (n_perm + 1))


def _bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values."""
    n = len(pvals)
    qvals = [float("nan")] * n
    finite_idx = [i for i, p in enumerate(pvals) if np.isfinite(p)]
    if not finite_idx:
        return qvals

    ranked = sorted(((pvals[i], i) for i in finite_idx), key=lambda t: t[0])
    m = len(ranked)
    adjusted = [0.0] * m
    for rank, (p, _) in enumerate(ranked, start=1):
        adjusted[rank - 1] = min(1.0, (p * m) / rank)
    # enforce monotonicity from tail to head
    for k in range(m - 2, -1, -1):
        adjusted[k] = min(adjusted[k], adjusted[k + 1])
    for (adj, (_, idx)) in zip(adjusted, ranked):
        qvals[idx] = float(adj)
    return qvals


def _bernoulli_nll(bits: np.ndarray, probs: np.ndarray, eps: float = 1e-7) -> float:
    p = np.clip(probs, eps, 1.0 - eps)
    ll = bits * np.log(p[None, :]) + (1.0 - bits) * np.log(1.0 - p[None, :])
    return float(-np.mean(np.sum(ll, axis=1)))


def _product_distribution(all_bits: np.ndarray, probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(probs, eps, 1.0 - eps)
    logp = all_bits * np.log(p[None, :]) + (1.0 - all_bits) * np.log(1.0 - p[None, :])
    out = np.exp(np.sum(logp, axis=1))
    z = out.sum()
    if z <= 0:
        return np.full_like(out, 1.0 / len(out))
    return out / z


def run_independent_snapshot_study(
    states: np.ndarray,
    times: np.ndarray,
    n_sites: int,
    magic_m2: np.ndarray,
    cfg,
) -> pd.DataFrame:
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


class ExactRBM(torch.nn.Module):
    """RBM with exact visible normalization over all 2^n states (small-n regime)."""

    def __init__(self, n_sites: int, n_hidden: int) -> None:
        super().__init__()
        self.n_sites = int(n_sites)
        self.n_hidden = int(n_hidden)
        self.visible_bias = torch.nn.Parameter(torch.zeros(self.n_sites))
        self.hidden_bias = torch.nn.Parameter(torch.zeros(self.n_hidden))
        self.weights = torch.nn.Parameter(0.01 * torch.randn(self.n_sites, self.n_hidden))

    def unnormalized_log_prob(self, bits: torch.Tensor) -> torch.Tensor:
        bits = bits.float()
        linear = bits @ self.visible_bias
        hidden_term = torch.nn.functional.softplus(bits @ self.weights + self.hidden_bias).sum(dim=1)
        return linear + hidden_term

    def log_prob(self, bits: torch.Tensor, all_bits: torch.Tensor) -> torch.Tensor:
        scores = self.unnormalized_log_prob(bits)
        all_scores = self.unnormalized_log_prob(all_bits)
        logz = torch.logsumexp(all_scores, dim=0)
        return scores - logz

    def nll(self, bits: torch.Tensor, all_bits: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(bits, all_bits)


def train_rbm_on_state(
    psi: np.ndarray,
    n_sites: int,
    cfg,
    hidden_size: int,
    seed: int,
    device: str = "cpu",
) -> dict[str, float | dict[str, list[float]]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    samples = sample_bitstrings_from_state(
        psi=psi,
        n_sites=n_sites,
        num_samples=cfg.measurement_samples,
        seed=seed,
    )
    train_np, val_np = train_val_split(samples, val_fraction=cfg.val_fraction, seed=seed)

    train_t = torch.tensor(train_np, dtype=torch.float32, device=dev)
    val_t = torch.tensor(val_np, dtype=torch.float32, device=dev)
    all_bits_np = all_bitstrings(n_sites).astype(np.float32)
    all_bits_t = torch.tensor(all_bits_np, dtype=torch.float32, device=dev)

    train_loader = DataLoader(TensorDataset(train_t), batch_size=cfg.batch_size, shuffle=True)
    model = ExactRBM(n_sites=n_sites, n_hidden=int(hidden_size)).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    hist_train: list[float] = []
    hist_val: list[float] = []
    steps_to_threshold = -1

    for epoch in range(int(cfg.epochs)):
        model.train()
        running = 0.0
        count = 0
        for (batch,) in train_loader:
            optim.zero_grad(set_to_none=True)
            loss = model.nll(batch, all_bits_t).mean()
            loss.backward()
            optim.step()
            running += float(loss.item()) * batch.shape[0]
            count += int(batch.shape[0])

        train_nll = running / max(count, 1)
        model.eval()
        with torch.no_grad():
            val_nll = float(model.nll(val_t, all_bits_t).mean().item())
        hist_train.append(train_nll)
        hist_val.append(val_nll)
        if steps_to_threshold < 0 and val_nll <= cfg.threshold_nll:
            steps_to_threshold = epoch + 1

    with torch.no_grad():
        logp_all = model.log_prob(all_bits_t, all_bits_t).cpu().numpy()
    p_model = np.exp(logp_all)
    p_model = p_model / max(np.sum(p_model), 1e-15)

    p_true = state_probabilities(psi)
    eps = 1e-15
    kl = float(np.sum(p_true * (np.log(p_true + eps) - np.log(p_model + eps))))
    tv = total_variation_distance(p_true, p_model)

    return {
        "final_train_nll": float(hist_train[-1]),
        "final_val_nll": float(hist_val[-1]),
        "final_kl": float(kl),
        "final_tv": float(tv),
        "steps_to_threshold": float(steps_to_threshold),
        "history": {"train_nll": hist_train, "val_nll": hist_val},
    }


def run_rbm_snapshot_study(
    states: np.ndarray,
    times: np.ndarray,
    n_sites: int,
    magic_m2: np.ndarray,
    cfg,
    hidden_size: int,
    device: str = "cpu",
) -> pd.DataFrame:
    n_times = states.shape[0]
    snapshot_idx = np.unique(np.linspace(0, n_times - 1, cfg.snapshot_count, dtype=int))
    rows: list[dict[str, float]] = []
    for rank, idx in enumerate(snapshot_idx):
        seed = int(cfg.seed + rank)
        res = train_rbm_on_state(
            psi=states[idx],
            n_sites=n_sites,
            cfg=cfg,
            hidden_size=int(hidden_size),
            seed=seed,
            device=device,
        )
        rows.append(
            {
                "snapshot_index": int(idx),
                "time": float(times[idx]),
                "magic_m2": float(magic_m2[idx]),
                "final_train_nll": float(res["final_train_nll"]),
                "final_val_nll": float(res["final_val_nll"]),
                "final_kl": float(res["final_kl"]),
                "final_tv": float(res["final_tv"]),
                "steps_to_threshold": float(res["steps_to_threshold"]),
            }
        )
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def approximate_magic_m2_state(
    psi: np.ndarray,
    n_sites: int,
    n_pauli_samples: int,
    seed: int,
) -> float:
    """Monte Carlo estimate of M2 using random Pauli strings."""
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

    mean_c4 = float(np.mean(c4_vals))
    mean_c4 = max(mean_c4, 1e-15)
    d = float(2**n_sites)
    return float(-np.log(mean_c4) - np.log(d))


def approximate_magic_m2_over_time(
    states: np.ndarray,
    n_sites: int,
    n_pauli_samples: int,
    seed_base: int,
) -> np.ndarray:
    out = np.empty(states.shape[0], dtype=np.float64)
    for t_idx, psi in enumerate(states):
        out[t_idx] = approximate_magic_m2_state(
            psi=psi,
            n_sites=n_sites,
            n_pauli_samples=n_pauli_samples,
            seed=seed_base + t_idx,
        )
    return out


def _fisher_meta(r: np.ndarray, n: np.ndarray) -> tuple[float, float, float, int]:
    mask = np.isfinite(r) & np.isfinite(n) & (n > 3)
    if np.count_nonzero(mask) == 0:
        return float("nan"), float("nan"), float("nan"), 0

    rr = np.clip(r[mask], -0.999999, 0.999999)
    nn = n[mask]
    z = np.arctanh(rr)
    w = nn - 3.0
    z_bar = float(np.sum(w * z) / np.sum(w))
    se = float(1.0 / math.sqrt(np.sum(w)))

    pooled = math.tanh(z_bar)
    lo = math.tanh(z_bar - 1.96 * se)
    hi = math.tanh(z_bar + 1.96 * se)
    return float(pooled), float(lo), float(hi), int(np.count_nonzero(mask))


def _heterogeneity_i2(r: np.ndarray, n: np.ndarray) -> float:
    mask = np.isfinite(r) & np.isfinite(n) & (n > 3)
    if np.count_nonzero(mask) <= 1:
        return float("nan")
    rr = np.clip(r[mask], -0.999999, 0.999999)
    nn = n[mask]
    z = np.arctanh(rr)
    w = nn - 3.0
    z_bar = float(np.sum(w * z) / np.sum(w))
    q = float(np.sum(w * (z - z_bar) ** 2))
    df = float(len(z) - 1)
    if q <= 0.0:
        return 0.0
    return float(max(0.0, (q - df) / q))


def _classify_effect(
    pooled_r: float,
    ci_low: float,
    ci_high: float,
    i2: float,
    positive_fraction: float,
    significant_fraction: float,
    combined_p: float,
    combined_q: float,
) -> str:
    if not np.isfinite(pooled_r):
        return "insufficient"
    if (
        ci_low > 0.0
        and pooled_r >= 0.35
        and (not np.isfinite(i2) or i2 <= 0.6)
        and positive_fraction >= 0.7
        and np.isfinite(combined_p)
        and combined_p < 0.05
        and (not np.isfinite(combined_q) or combined_q < 0.1)
    ):
        return "supported"
    if pooled_r > 0.15 and ci_high > 0.0 and positive_fraction >= 0.6:
        return "mixed"
    return "unsupported"


def _find_peak_time(times: np.ndarray, series: np.ndarray) -> float:
    if len(times) == 0 or len(series) == 0 or not np.isfinite(series).any():
        return float("nan")
    idx = int(np.nanargmax(series))
    return float(times[idx])


def _quench_feature_row(n_sites: int, theta1: float, run: QuenchRun) -> dict[str, float]:
    lam = run.loschmidt_rate
    row: dict[str, float] = {
        "n_sites": float(n_sites),
        "theta1": float(theta1),
        "max_lambda": float(np.max(lam)),
        "max_entropy": float(np.max(run.entanglement_entropy)),
        "peak_time_lambda": _find_peak_time(run.times, lam),
        "norm_drift": float(run.norm_drift),
    }

    if 2.0 in run.magic:
        m2 = run.magic[2.0]
        row.update(
            {
                "max_magic_m2": float(np.max(m2)),
                "peak_time_m2": _find_peak_time(run.times, m2),
                "peak_lag_m2_minus_lambda": _find_peak_time(run.times, m2) - _find_peak_time(run.times, lam),
                "max_abs_dlambda_dt": float(np.max(np.abs(np.gradient(lam, run.times)))),
                "max_abs_dm2_dt": float(np.max(np.abs(np.gradient(m2, run.times)))),
            }
        )
    else:
        row.update(
            {
                "max_magic_m2": float("nan"),
                "peak_time_m2": float("nan"),
                "peak_lag_m2_minus_lambda": float("nan"),
                "max_abs_dlambda_dt": float(np.max(np.abs(np.gradient(lam, run.times)))),
                "max_abs_dm2_dt": float("nan"),
            }
        )

    return row


def _quench_stats_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    for n_sites in sorted(df["n_sites"].unique()):
        sub = df[df["n_sites"] == n_sites].sort_values("theta1")
        theta = sub["theta1"].to_numpy()

        rho_lambda, p_lambda = _safe_corr(theta, sub["max_lambda"].to_numpy(), spearmanr)
        rho_m2, p_m2 = _safe_corr(theta, sub["max_magic_m2"].to_numpy(), spearmanr)
        r_slope, p_slope = _safe_corr(
            sub["max_abs_dlambda_dt"].to_numpy(),
            sub["max_abs_dm2_dt"].to_numpy(),
            pearsonr,
        )

        rows.append(
            {
                "n_sites": int(n_sites),
                "spearman_theta_vs_max_lambda": rho_lambda,
                "spearman_p_theta_vs_max_lambda": p_lambda,
                "spearman_theta_vs_max_magic": rho_m2,
                "spearman_p_theta_vs_max_magic": p_m2,
                "pearson_peak_slope_lambda_vs_m2": r_slope,
                "pearson_p_peak_slope_lambda_vs_m2": p_slope,
                "monotonic_theta_vs_max_lambda": bool(np.all(np.diff(sub["max_lambda"].to_numpy()) >= -1e-12)),
                "monotonic_theta_vs_max_magic": bool(np.all(np.diff(sub["max_magic_m2"].to_numpy()) >= -1e-12)),
            }
        )
    return pd.DataFrame(rows)


def _plot_corr_forest(df_stats: pd.DataFrame, save_path: Path) -> None:
    if df_stats.empty:
        return

    sort_cols = [c for c in ["architecture", "n_sites", "theta1", "hidden_size", "seed"] if c in df_stats.columns]
    df = df_stats.sort_values(sort_cols).reset_index(drop=True)
    y = np.arange(len(df))
    labels = [
        f"{getattr(r, 'architecture', 'na')}: N={int(r.n_sites)}, th={r.theta1:.1f}, h={int(r.hidden_size)}, s={int(r.seed)}"
        for r in df.itertuples(index=False)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5.0, 0.25 * len(df))), sharey=True)

    axes[0].axvline(0.0, color="#888", lw=1.0)
    x0 = df["pearson_magic_vs_val_nll"].to_numpy(dtype=float)
    x0_low = np.maximum(
        (df["pearson_magic_vs_val_nll"] - df["pearson_magic_vs_val_nll_ci_low"]).to_numpy(dtype=float),
        0.0,
    )
    x0_high = np.maximum(
        (df["pearson_magic_vs_val_nll_ci_high"] - df["pearson_magic_vs_val_nll"]).to_numpy(dtype=float),
        0.0,
    )
    axes[0].errorbar(
        x0,
        y,
        xerr=np.vstack([x0_low, x0_high]),
        fmt="o",
        ms=4,
        color="#1b6ca8",
        ecolor="#6aa3cf",
        elinewidth=1.2,
        capsize=2,
    )
    axes[0].set_title("Magic vs validation NLL")
    axes[0].set_xlabel("Pearson r (95% bootstrap CI)")
    axes[0].grid(alpha=0.2)

    axes[1].axvline(0.0, color="#888", lw=1.0)
    x1 = df["pearson_magic_vs_kl"].to_numpy(dtype=float)
    x1_low = np.maximum(
        (df["pearson_magic_vs_kl"] - df["pearson_magic_vs_kl_ci_low"]).to_numpy(dtype=float),
        0.0,
    )
    x1_high = np.maximum(
        (df["pearson_magic_vs_kl_ci_high"] - df["pearson_magic_vs_kl"]).to_numpy(dtype=float),
        0.0,
    )
    axes[1].errorbar(
        x1,
        y,
        xerr=np.vstack([x1_low, x1_high]),
        fmt="o",
        ms=4,
        color="#b84747",
        ecolor="#d28a8a",
        elinewidth=1.2,
        capsize=2,
    )
    axes[1].set_title("Magic vs KL")
    axes[1].set_xlabel("Pearson r (95% bootstrap CI)")
    axes[1].grid(alpha=0.2)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_corr_distributions(df_stats: pd.DataFrame, save_path: Path) -> None:
    if df_stats.empty:
        return

    specs = [
        ("pearson_magic_vs_val_nll", "M2->valNLL", "#1b6ca8"),
        ("pearson_entropy_vs_val_nll", "S->valNLL", "#8f5fbf"),
        ("pearson_magic_vs_val_nll_partial_entropy", "M2->valNLL|S", "#1f7a4f"),
        ("pearson_magic_vs_kl", "M2->KL", "#b84747"),
        ("pearson_magic_vs_tv", "M2->TV", "#6b8f3f"),
    ]
    available = [(col, lab, c) for (col, lab, c) in specs if col in df_stats.columns]
    data = [df_stats[col].dropna().to_numpy(dtype=float) for (col, _, _) in available]
    labels = [lab for (_, lab, _) in available]
    colors = [c for (_, _, c) in available]
    if not data:
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_alpha(0.35)

    for i, arr in enumerate(data, start=1):
        if len(arr) > 0:
            x = np.full(len(arr), i, dtype=float)
            jitter = np.linspace(-0.12, 0.12, len(arr))
            ax.scatter(x + jitter, arr, s=22, alpha=0.75, color=colors[i - 1], edgecolors="none")

    ax.axhline(0.0, color="#888", lw=1.0)
    ax.set_ylabel("Pearson r with magic M2")
    ax.set_title("NNQS metric sensitivity to magic across robustness runs")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _plot_global_scatter(df_snapshots: pd.DataFrame, save_path: Path) -> None:
    if df_snapshots.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for ax, y_col, title, color in [
        (axes[0], "final_val_nll", "Magic vs validation NLL", "#1b6ca8"),
        (axes[1], "final_kl", "Magic vs KL", "#b84747"),
    ]:
        x = df_snapshots["magic_m2"].to_numpy()
        y = df_snapshots[y_col].to_numpy()
        ax.scatter(x, y, s=20, alpha=0.55, color=color, edgecolors="none")

        r, _ = _safe_corr(x, y, pearsonr)
        if np.isfinite(r) and len(x) >= 2:
            c0, c1 = np.polyfit(x, y, deg=1)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 120)
            ax.plot(xs, c0 * xs + c1, color="#222", lw=1.5, label=f"r={r:.3f}")
            ax.legend(frameon=False)

        ax.set_xlabel("Magic M2")
        ax.set_ylabel(y_col)
        ax.set_title(title)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _build_meta(df_stats: pd.DataFrame, metric_col: str, p_col: str) -> MetaResult:
    r = df_stats[metric_col].to_numpy(dtype=float)
    n = df_stats["n_snapshots"].to_numpy(dtype=float)
    pooled, lo, hi, n_studies = _fisher_meta(r, n)
    i2 = _heterogeneity_i2(r, n)

    finite = np.isfinite(r)
    positive_fraction = float(np.mean(r[finite] > 0.0)) if np.any(finite) else float("nan")
    pvals = df_stats[p_col].to_numpy(dtype=float)
    finite_p = np.isfinite(pvals)
    significant_fraction = float(np.mean(pvals[finite_p] < 0.05)) if np.any(finite_p) else float("nan")
    if np.any(finite_p):
        p_clip = np.clip(pvals[finite_p], 1e-12, 1.0)
        fisher_stat = float(-2.0 * np.sum(np.log(p_clip)))
        combined_p = float(chi2.sf(fisher_stat, 2 * len(p_clip)))
        if combined_p == 0.0:
            combined_p = float(np.nextafter(0.0, 1.0))
    else:
        combined_p = float("nan")
    combined_q = float("nan")
    verdict = _classify_effect(pooled, lo, hi, i2, positive_fraction, significant_fraction, combined_p, combined_q)

    return MetaResult(
        metric=metric_col,
        pooled_r=pooled,
        ci_low=lo,
        ci_high=hi,
        i2=i2,
        n_studies=n_studies,
        positive_fraction=positive_fraction,
        significant_fraction=significant_fraction,
        combined_p=combined_p,
        combined_q=combined_q,
        verdict=verdict,
    )


def _write_claims_md(path: Path, metas: list[MetaResult], quench_stats: pd.DataFrame) -> None:
    lines: list[str] = []
    lines.append("# Novelty Claim Assessment")
    lines.append("")
    lines.append("This report converts robustness statistics into claim verdicts with explicit thresholds.")
    lines.append("")

    lines.append("## NNQS Learnability Claims")
    label_map = {
        "pearson_magic_vs_val_nll": "magic -> validation NLL",
        "pearson_entropy_vs_val_nll": "entropy -> validation NLL",
        "pearson_magic_vs_val_nll_partial_entropy": "magic -> validation NLL | entropy",
        "pearson_magic_vs_kl": "magic -> KL",
        "pearson_magic_vs_tv": "magic -> TV",
    }
    for m in metas:
        if ":" in m.metric:
            arch, raw_metric = m.metric.split(":", 1)
            metric_label = f"{arch} | {label_map.get(raw_metric, raw_metric)}"
        else:
            metric_label = label_map.get(m.metric, m.metric)
        lines.append(
            f"- `{metric_label}`: **{m.verdict}** | pooled r={m.pooled_r:.3f} "
            f"[95% CI {m.ci_low:.3f}, {m.ci_high:.3f}], studies={m.n_studies}, "
            f"I2={m.i2:.2f}, positive frac={m.positive_fraction:.2f}, significant frac={m.significant_fraction:.2f}, "
            f"combined p={m.combined_p:.3g}, FDR q={m.combined_q:.3g}"
        )

    lines.append("")
    lines.append("## Quench Dynamics Claims")
    if quench_stats.empty:
        lines.append("- No quench stats available.")
    else:
        for r in quench_stats.itertuples(index=False):
            lines.append(
                f"- `N={int(r.n_sites)}`: Spearman(theta,max lambda)={r.spearman_theta_vs_max_lambda:.3f} "
                f"(p={r.spearman_p_theta_vs_max_lambda:.3g}), "
                f"Spearman(theta,max M2)={r.spearman_theta_vs_max_magic:.3f} "
                f"(p={r.spearman_p_theta_vs_max_magic:.3g}), "
                f"peak-slope coupling r={r.pearson_peak_slope_lambda_vs_m2:.3f} "
                f"(p={r.pearson_p_peak_slope_lambda_vs_m2:.3g})"
            )

    lines.append("")
    lines.append("## Safe Wording")
    lines.append("- Use: 'robust small-N evidence' when verdict is `supported`.")
    lines.append("- Use: 'suggestive but mixed' when verdict is `mixed`.")
    lines.append("- Avoid universal claims when verdict is `unsupported`.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t_all0 = time.perf_counter()
    parser = argparse.ArgumentParser(description="Robustness suite for novelty-grade claims")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-sites", type=str, default="6,8")
    parser.add_argument("--theta1", type=str, default="")
    parser.add_argument("--seeds", type=str, default="11,17,23")
    parser.add_argument("--hidden-sizes", type=str, default="48")
    parser.add_argument("--architectures", type=str, default="gru,made,rbm,independent")
    parser.add_argument("--out-dir", type=str, default="outputs/novelty")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=300)
    parser.add_argument("--approx-magic-samples", type=int, default=0, help="If >0, use Monte Carlo M2 estimate when exact magic is unavailable")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--measurement-samples", type=int, default=None)
    parser.add_argument("--snapshot-count", type=int, default=None)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write partial snapshot/stat CSV checkpoints every K completed conditions (0 disables).",
    )
    parser.add_argument("--quick", action="store_true", help="fast local pass with fewer settings")
    args = parser.parse_args()

    cfg = load_config(args.config)

    n_sites_list = _parse_int_list(args.n_sites, [cfg.model.n_sites])
    theta_values = _parse_float_list(args.theta1, [float(t) for t in cfg.model.theta1_values])
    seeds = _parse_int_list(args.seeds, [cfg.nnqs.seed])
    hidden_sizes = _parse_int_list(args.hidden_sizes, [cfg.nnqs.hidden_size])
    architectures = [a.strip().lower() for a in args.architectures.split(",") if a.strip()]
    supported_arch = {"gru", "made", "rbm", "independent"}
    unknown_arch = [a for a in architectures if a not in supported_arch]
    if unknown_arch:
        raise ValueError(f"Unsupported architectures: {unknown_arch}. Supported: {sorted(supported_arch)}")

    if args.quick:
        theta_values = theta_values[: min(3, len(theta_values))]
        seeds = seeds[: min(2, len(seeds))]
        hidden_sizes = hidden_sizes[:1]
        if args.bootstrap > 120:
            args.bootstrap = 120
        if args.permutations > 200:
            args.permutations = 200

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if 2.0 not in [float(a) for a in cfg.magic.alphas]:
        cfg.magic.alphas = [2.0] + [float(a) for a in cfg.magic.alphas]

    quench_rows: list[dict[str, float]] = []
    quench_cache: dict[tuple[int, float], QuenchRun] = {}

    for n_sites in n_sites_list:
        for theta1 in theta_values:
            c = copy.deepcopy(cfg)
            c.model.n_sites = int(n_sites)
            run = run_single_quench(c, theta1=float(theta1), compute_dense_krylov=False)
            quench_rows.append(_quench_feature_row(n_sites=int(n_sites), theta1=float(theta1), run=run))
            quench_cache[(int(n_sites), float(theta1))] = run

    quench_df = pd.DataFrame(quench_rows).sort_values(["n_sites", "theta1"]).reset_index(drop=True)
    quench_df.to_csv(out_dir / "quench_trend_features.csv", index=False)

    quench_stats = _quench_stats_table(quench_df)
    quench_stats.to_csv(out_dir / "quench_trend_stats.csv", index=False)

    snapshot_rows: list[pd.DataFrame] = []
    condition_rows: list[dict[str, float | int | str]] = []

    run_id = 0
    for n_sites in n_sites_list:
        for theta1 in theta_values:
            run = quench_cache[(int(n_sites), float(theta1))]
            if 2.0 in run.magic:
                m2 = run.magic[2.0]
            elif args.approx_magic_samples > 0:
                m2 = approximate_magic_m2_over_time(
                    states=run.state_trajectory,
                    n_sites=int(n_sites),
                    n_pauli_samples=int(args.approx_magic_samples),
                    seed_base=int(cfg.nnqs.seed + 10000 * int(n_sites) + int(round(theta1 * 1000))),
                )
            else:
                continue

            for architecture in architectures:
                hidden_candidates = hidden_sizes if architecture in {"gru", "made", "rbm"} else [0]
                for hidden in hidden_candidates:
                    for seed in seeds:
                        t_cond0 = time.perf_counter()
                        c = copy.deepcopy(cfg)
                        c.model.n_sites = int(n_sites)
                        c.nnqs.hidden_size = int(hidden)
                        c.nnqs.seed = int(seed)
                        if args.epochs is not None:
                            c.nnqs.epochs = int(args.epochs)
                        if args.measurement_samples is not None:
                            c.nnqs.measurement_samples = int(args.measurement_samples)
                        if args.snapshot_count is not None:
                            c.nnqs.snapshot_count = int(args.snapshot_count)

                        if architecture in {"gru", "made"}:
                            df_snap, _ = run_snapshot_study(
                                states=run.state_trajectory,
                                times=run.times,
                                n_sites=c.model.n_sites,
                                magic_m2=m2,
                                cfg=c.nnqs,
                                device=args.device,
                                model_type=architecture,
                            )
                        elif architecture == "rbm":
                            df_snap = run_rbm_snapshot_study(
                                states=run.state_trajectory,
                                times=run.times,
                                n_sites=c.model.n_sites,
                                magic_m2=m2,
                                cfg=c.nnqs,
                                hidden_size=int(hidden),
                                device=args.device,
                            )
                        else:
                            df_snap = run_independent_snapshot_study(
                                states=run.state_trajectory,
                                times=run.times,
                                n_sites=c.model.n_sites,
                                magic_m2=m2,
                                cfg=c.nnqs,
                            )

                        cond = f"N{int(n_sites)}_th{float(theta1):.2f}_arch{architecture}_h{int(hidden)}_s{int(seed)}"
                        df_snap = df_snap.copy()
                        df_snap["condition_id"] = cond
                        df_snap["architecture"] = architecture
                        df_snap["n_sites"] = int(n_sites)
                        df_snap["theta1"] = float(theta1)
                        df_snap["hidden_size"] = int(hidden)
                        df_snap["seed"] = int(seed)
                        snap_idx = df_snap["snapshot_index"].to_numpy(dtype=int)
                        df_snap["snapshot_entropy"] = run.entanglement_entropy[snap_idx]
                        snapshot_rows.append(df_snap)

                        x = df_snap["magic_m2"].to_numpy(dtype=float)
                        z_entropy = df_snap["snapshot_entropy"].to_numpy(dtype=float)
                        y_nll = df_snap["final_val_nll"].to_numpy(dtype=float)
                        y_kl = df_snap["final_kl"].to_numpy(dtype=float)
                        y_tv = df_snap["final_tv"].to_numpy(dtype=float)

                        r_nll, p_nll = _safe_corr(x, y_nll, pearsonr)
                        rs_nll, ps_nll = _safe_corr(x, y_nll, spearmanr)
                        r_entropy_nll, p_entropy_nll = _safe_corr(z_entropy, y_nll, pearsonr)
                        r_partial, p_partial = _partial_corr(x, y_nll, z_entropy)
                        r_kl, p_kl = _safe_corr(x, y_kl, pearsonr)
                        r_tv, p_tv = _safe_corr(x, y_tv, pearsonr)

                        ci_nll_lo, ci_nll_hi = _bootstrap_corr_ci(x, y_nll, "pearson", args.bootstrap, seed + 1000)
                        ci_entropy_lo, ci_entropy_hi = _bootstrap_corr_ci(
                            z_entropy, y_nll, "pearson", args.bootstrap, seed + 1500
                        )
                        ci_partial_lo, ci_partial_hi = _bootstrap_partial_corr_ci(
                            x, y_nll, z_entropy, args.bootstrap, seed + 1750
                        )
                        ci_kl_lo, ci_kl_hi = _bootstrap_corr_ci(x, y_kl, "pearson", args.bootstrap, seed + 2000)
                        ci_tv_lo, ci_tv_hi = _bootstrap_corr_ci(x, y_tv, "pearson", args.bootstrap, seed + 3000)

                        perm_nll = _permutation_pvalue(x, y_nll, "pearson", args.permutations, seed + 4000)
                        perm_entropy_nll = _permutation_pvalue(
                            z_entropy, y_nll, "pearson", args.permutations, seed + 4500
                        )
                        perm_partial = _permutation_partial_pvalue(
                            x, y_nll, z_entropy, args.permutations, seed + 4750
                        )
                        perm_kl = _permutation_pvalue(x, y_kl, "pearson", args.permutations, seed + 5000)
                        perm_tv = _permutation_pvalue(x, y_tv, "pearson", args.permutations, seed + 6000)
                        null_random_label_r, null_shuffled_time_r = _null_baseline_corrs(
                            x, y_nll, seed=seed + 7000
                        )
                        cond_runtime_s = float(time.perf_counter() - t_cond0)

                        condition_rows.append(
                            {
                                "run_id": run_id,
                                "condition_id": cond,
                                "architecture": architecture,
                                "n_sites": int(n_sites),
                                "theta1": float(theta1),
                                "hidden_size": int(hidden),
                                "seed": int(seed),
                                "n_snapshots": int(len(df_snap)),
                                "pearson_magic_vs_val_nll": r_nll,
                                "pearson_p_magic_vs_val_nll": p_nll,
                                "spearman_magic_vs_val_nll": rs_nll,
                                "spearman_p_magic_vs_val_nll": ps_nll,
                                "pearson_entropy_vs_val_nll": r_entropy_nll,
                                "pearson_p_entropy_vs_val_nll": p_entropy_nll,
                                "pearson_magic_vs_val_nll_partial_entropy": r_partial,
                                "pearson_p_magic_vs_val_nll_partial_entropy": p_partial,
                                "pearson_magic_vs_kl": r_kl,
                                "pearson_p_magic_vs_kl": p_kl,
                                "pearson_magic_vs_tv": r_tv,
                                "pearson_p_magic_vs_tv": p_tv,
                                "pearson_magic_vs_val_nll_ci_low": ci_nll_lo,
                                "pearson_magic_vs_val_nll_ci_high": ci_nll_hi,
                                "pearson_entropy_vs_val_nll_ci_low": ci_entropy_lo,
                                "pearson_entropy_vs_val_nll_ci_high": ci_entropy_hi,
                                "pearson_magic_vs_val_nll_partial_entropy_ci_low": ci_partial_lo,
                                "pearson_magic_vs_val_nll_partial_entropy_ci_high": ci_partial_hi,
                                "pearson_magic_vs_kl_ci_low": ci_kl_lo,
                                "pearson_magic_vs_kl_ci_high": ci_kl_hi,
                                "pearson_magic_vs_tv_ci_low": ci_tv_lo,
                                "pearson_magic_vs_tv_ci_high": ci_tv_hi,
                                "perm_p_magic_vs_val_nll": perm_nll,
                                "perm_p_entropy_vs_val_nll": perm_entropy_nll,
                                "perm_p_magic_vs_val_nll_partial_entropy": perm_partial,
                                "perm_p_magic_vs_kl": perm_kl,
                                "perm_p_magic_vs_tv": perm_tv,
                                "null_random_label_r_magic_vs_val_nll": null_random_label_r,
                                "null_shuffled_time_r_magic_vs_val_nll": null_shuffled_time_r,
                                "condition_runtime_s": cond_runtime_s,
                            }
                        )
                        run_id += 1

                        if args.checkpoint_every > 0 and (run_id % int(args.checkpoint_every) == 0):
                            snap_partial = (
                                pd.concat(snapshot_rows, axis=0, ignore_index=True)
                                if snapshot_rows
                                else pd.DataFrame()
                            )
                            stats_partial = pd.DataFrame(condition_rows)
                            snap_partial.to_csv(out_dir / "nnqs_snapshot_all.partial.csv", index=False)
                            stats_partial.to_csv(out_dir / "nnqs_condition_stats.partial.csv", index=False)
                            print(
                                f"[checkpoint] completed_conditions={run_id} "
                                f"snapshots={len(snap_partial)} out_dir={out_dir}",
                                flush=True,
                            )

    snap_df = pd.concat(snapshot_rows, axis=0, ignore_index=True) if snapshot_rows else pd.DataFrame()
    snap_df.to_csv(out_dir / "nnqs_snapshot_all.csv", index=False)

    stats_df = pd.DataFrame(condition_rows)
    stats_df.to_csv(out_dir / "nnqs_condition_stats.csv", index=False)

    _plot_corr_forest(stats_df, fig_dir / "novelty_corr_forest.png")
    _plot_corr_distributions(stats_df, fig_dir / "novelty_metric_distributions.png")
    _plot_global_scatter(snap_df, fig_dir / "novelty_global_scatter.png")

    metric_specs = [
        ("pearson_magic_vs_val_nll", "perm_p_magic_vs_val_nll"),
        ("pearson_entropy_vs_val_nll", "perm_p_entropy_vs_val_nll"),
        ("pearson_magic_vs_val_nll_partial_entropy", "perm_p_magic_vs_val_nll_partial_entropy"),
        ("pearson_magic_vs_kl", "perm_p_magic_vs_kl"),
        ("pearson_magic_vs_tv", "perm_p_magic_vs_tv"),
    ]

    metas: list[MetaResult] = []
    if stats_df.empty:
        for metric_col, _ in metric_specs:
            metas.append(
                MetaResult(
                    metric_col,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    0,
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    float("nan"),
                    "insufficient",
                )
            )
    else:
        arch_values = sorted(stats_df["architecture"].dropna().unique()) if "architecture" in stats_df.columns else ["all"]
        for arch in arch_values:
            sub = stats_df[stats_df["architecture"] == arch] if "architecture" in stats_df.columns else stats_df
            for metric_col, p_col in metric_specs:
                m = _build_meta(sub, metric_col, p_col)
                m.metric = f"{arch}:{metric_col}"
                metas.append(m)
    qvals = _bh_fdr([m.combined_p for m in metas])
    for m, q in zip(metas, qvals):
        m.combined_q = q
        m.verdict = _classify_effect(
            m.pooled_r,
            m.ci_low,
            m.ci_high,
            m.i2,
            m.positive_fraction,
            m.significant_fraction,
            m.combined_p,
            m.combined_q,
        )

    summary = {
        "config": str(args.config),
        "n_sites": [int(x) for x in n_sites_list],
        "theta1_values": [float(x) for x in theta_values],
        "seeds": [int(x) for x in seeds],
        "hidden_sizes": [int(x) for x in hidden_sizes],
        "architectures": architectures,
        "bootstrap": int(args.bootstrap),
        "permutations": int(args.permutations),
        "approx_magic_samples": int(args.approx_magic_samples),
        "runtime_seconds": float(time.perf_counter() - t_all0),
        "quench_rows": int(len(quench_df)),
        "nnqs_conditions": int(len(stats_df)),
        "nnqs_snapshots": int(len(snap_df)),
        "negative_controls": {
            "mean_null_random_label_r": float(
                stats_df["null_random_label_r_magic_vs_val_nll"].mean()
            )
            if not stats_df.empty and "null_random_label_r_magic_vs_val_nll" in stats_df.columns
            else float("nan"),
            "mean_null_shuffled_time_r": float(
                stats_df["null_shuffled_time_r_magic_vs_val_nll"].mean()
            )
            if not stats_df.empty and "null_shuffled_time_r_magic_vs_val_nll" in stats_df.columns
            else float("nan"),
        },
        "quench_stats": quench_stats.to_dict(orient="records"),
        "meta": {
            m.metric: {
                "pooled_r": m.pooled_r,
                "ci_low": m.ci_low,
                "ci_high": m.ci_high,
                "i2": m.i2,
                "n_studies": m.n_studies,
                "positive_fraction": m.positive_fraction,
                "significant_fraction": m.significant_fraction,
                "combined_p": m.combined_p,
                "combined_q": m.combined_q,
                "verdict": m.verdict,
            }
            for m in metas
        },
    }

    with (out_dir / "novelty_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _write_claims_md(out_dir / "novelty_claims.md", metas, quench_stats)

    print(f"Wrote {out_dir / 'quench_trend_features.csv'}")
    print(f"Wrote {out_dir / 'quench_trend_stats.csv'}")
    print(f"Wrote {out_dir / 'nnqs_snapshot_all.csv'}")
    print(f"Wrote {out_dir / 'nnqs_condition_stats.csv'}")
    print(f"Wrote {out_dir / 'novelty_summary.json'}")
    print(f"Wrote {out_dir / 'novelty_claims.md'}")
    print(f"Wrote figures under {fig_dir}")


if __name__ == "__main__":
    main()
