from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
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
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import IterationLimitWarning
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqm.config import NNQSConfig
from tqm.evolve import evolve_state
from tqm.loschmidt import rate_function
from tqm.magic import magic_over_time
from tqm.nnqs.train import run_snapshot_study
from tqm.observables import bipartite_entropy_time
from tqm.pauli import pauli_expectation_batch_for_x


def _residualize(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.ones(len(z), dtype=float), z.astype(float)])
    beta, *_ = np.linalg.lstsq(a, y.astype(float), rcond=None)
    return y.astype(float) - (a @ beta)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from scipy.stats import pearsonr

    if len(x) < 3:
        return float("nan"), float("nan")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return float("nan"), float("nan")
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return float("nan"), float("nan")
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    if len(x) < 4:
        return float("nan"), float("nan")
    xr = _residualize(x, z)
    yr = _residualize(y, z)
    return _safe_corr(xr, yr)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    out = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    if not np.any(mask):
        return out
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
    out[mask] = q_masked
    return out


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


def _cluster_boot_quantile(df: pd.DataFrame, tau: float, n_boot: int, seed: int) -> tuple[float, float, float]:
    formula = "final_val_nll ~ magic_m2 + snapshot_entropy + C(model_family) + C(architecture) + C(regime)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", IterationLimitWarning)
        fit = smf.quantreg(formula, data=df).fit(q=tau, max_iter=3500)
    point = float(fit.params.get("magic_m2", float("nan")))
    cond_ids = np.array(sorted(df["condition_id"].astype(str).unique().tolist()))
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(cond_ids, size=len(cond_ids), replace=True)
        parts = [df[df["condition_id"].astype(str) == cid] for cid in pick]
        bdf = pd.concat(parts, ignore_index=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", IterationLimitWarning)
                bf = smf.quantreg(formula, data=bdf).fit(q=tau, max_iter=3500)
            v = float(bf.params.get("magic_m2", float("nan")))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if len(vals) < 12:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return point, float(lo), float(hi)


def _approximate_magic_m2_state(psi: np.ndarray, n_sites: int, n_pauli_samples: int, seed: int) -> float:
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


def build_annni_hamiltonian(n_sites: int, j1: float, kappa: float, hx: float, hz: float) -> sp.csr_matrix:
    dim = 2**n_sites
    h_tot = sp.csr_matrix((dim, dim), dtype=np.complex128)
    j2 = kappa * j1
    for i in range(n_sites - 1):
        h_tot = h_tot - j1 * _two_site_pauli(n_sites, i, i + 1, "Z", "Z")
    for i in range(n_sites - 2):
        h_tot = h_tot + j2 * _two_site_pauli(n_sites, i, i + 2, "Z", "Z")
    for i in range(n_sites):
        h_tot = h_tot - hx * _local_pauli(n_sites, i, "X")
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
class Stage5Config:
    stage3_base_dir: str
    out_dir: str
    base_architectures: list[str]
    n_sites: list[int]
    kappa_values: list[float]
    hz_values: list[float]
    h0: float
    h1_values: list[float]
    j1: float
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
    quantiles: list[float]
    exact_magic_max_sites: int
    approx_magic_samples: int


def _load_cfg(path: Path, out_override: str | None) -> Stage5Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out_dir = str(out_override) if out_override else str(payload.get("out_dir", "outputs/stage5_prx/external_regime_v1"))
    return Stage5Config(
        stage3_base_dir=str(payload.get("stage3_base_dir", "outputs/stage3_prx/universal_extension_v8")),
        out_dir=out_dir,
        base_architectures=[str(x) for x in payload.get("base_architectures", ["gru", "made"])],
        n_sites=[int(x) for x in payload.get("n_sites", [6, 8, 10])],
        kappa_values=[float(x) for x in payload.get("kappa_values", [0.2, 0.4])],
        hz_values=[float(x) for x in payload.get("hz_values", [0.0, 0.3])],
        h0=float(payload.get("h0", 0.2)),
        h1_values=[float(x) for x in payload.get("h1_values", [0.8, 1.4, 2.0])],
        j1=float(payload.get("j1", 1.0)),
        t_max=float(payload.get("t_max", 2.0)),
        n_steps=int(payload.get("n_steps", 11)),
        architectures=[str(x) for x in payload.get("architectures", ["gru", "made"])],
        seeds=[int(x) for x in payload.get("seeds", [11, 17])],
        hidden_size=int(payload.get("hidden_size", 56)),
        epochs=int(payload.get("epochs", 40)),
        measurement_samples=int(payload.get("measurement_samples", 3200)),
        snapshot_count=int(payload.get("snapshot_count", 6)),
        val_fraction=float(payload.get("val_fraction", 0.2)),
        lr=float(payload.get("lr", 1e-3)),
        batch_size=int(payload.get("batch_size", 256)),
        threshold_nll=float(payload.get("threshold_nll", 4.0)),
        bootstrap=int(payload.get("bootstrap", 220)),
        permutations=int(payload.get("permutations", 220)),
        lambda_quantile=float(payload.get("lambda_quantile", 0.4)),
        fdr_alpha=float(payload.get("fdr_alpha", 0.05)),
        quantiles=[float(x) for x in payload.get("quantiles", [0.1, 0.2, 0.3])],
        exact_magic_max_sites=int(payload.get("exact_magic_max_sites", 8)),
        approx_magic_samples=int(payload.get("approx_magic_samples", 10000)),
    )


def _nnqs_cfg(cfg: Stage5Config, seed: int) -> NNQSConfig:
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
    y = df["beta_magic"].to_numpy(dtype=float)
    lo = df["beta_magic_ci_low"].to_numpy(dtype=float)
    hi = df["beta_magic_ci_high"].to_numpy(dtype=float)
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o", capsize=4, color="#1259a7")
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("beta_magic (95% CI)")
    ax.set_title("Stage-5 cross-family beta-law bounds")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_regime_minimax(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    g = (
        df.groupby(["model_family", "architecture"], as_index=False)["beta_magic_ci_low"]
        .min()
        .rename(columns={"beta_magic_ci_low": "worst_ci_low"})
        .sort_values(["model_family", "architecture"])
    )
    labels = [f"{m}\n{a}" for m, a in zip(g["model_family"], g["architecture"], strict=False)]
    x = np.arange(len(g))
    y = g["worst_ci_low"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    ax.bar(x, y, color="#7a1f5c", alpha=0.9)
    ax.axhline(0.0, color="k", lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("worst-case beta CI low across regimes")
    ax.set_title("Stage-5 adversarial regime bound (min over regimes)")
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_regime_heatmap(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    regimes = sorted(df["regime"].astype(str).unique().tolist())
    cells = sorted({(str(m), str(a)) for m, a in zip(df["model_family"], df["architecture"], strict=False)})
    mat = np.full((len(regimes), len(cells)), np.nan, dtype=float)
    for i, reg in enumerate(regimes):
        for j, (fam, arch) in enumerate(cells):
            sub = df[
                (df["regime"].astype(str) == reg)
                & (df["model_family"].astype(str) == fam)
                & (df["architecture"].astype(str) == arch)
            ]
            if len(sub) == 0:
                continue
            mat[i, j] = float(sub["beta_magic_ci_low"].iloc[0])
    fig, ax = plt.subplots(figsize=(12.5, 5.3))
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels([f"{f}\n{a}" for f, a in cells], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(regimes)))
    ax.set_yticklabels(regimes)
    ax.set_title("Stage-5 regime boundary map (beta CI low)")
    for i in range(len(regimes)):
        for j in range(len(cells)):
            v = mat[i, j]
            txt = "NA" if not np.isfinite(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.86)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_quantile_bound(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    x = df["tau"].to_numpy(dtype=float)
    y = df["beta_magic"].to_numpy(dtype=float)
    lo = df["ci_low"].to_numpy(dtype=float)
    hi = df["ci_high"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=4, color="#1f6f3b")
    ax.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax.set_xlabel("quantile tau")
    ax.set_ylabel("beta_magic")
    ax.set_title("Stage-5 lower-envelope quantile bound")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-5 external-model + regime expansion")
    parser.add_argument("--config", type=str, default="configs/stage5_external_v1.yaml")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--registry-out", type=str, default="outputs/stage5_prx/stage5/effect_registry.json")
    parser.add_argument("--report-out", type=str, default="report/stage5_external_regime.md")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config), out_override=args.out_dir)
    out_dir = Path(cfg.out_dir)
    fig_dir = out_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Base from Stage-3.
    base_dir = ROOT / cfg.stage3_base_dir
    base_cond = pd.read_csv(base_dir / "all_condition_stats_stage3.csv")
    base_snap = pd.read_csv(base_dir / "all_snapshot_all_stage3.csv")
    if cfg.base_architectures:
        base_cond = base_cond[base_cond["architecture"].astype(str).isin(cfg.base_architectures)].copy()
        base_snap = base_snap[base_snap["architecture"].astype(str).isin(cfg.base_architectures)].copy()
    base_cond["regime"] = base_cond["regime"].fillna(base_cond["model_family"].astype(str) + "_default")
    base_snap["regime"] = base_snap["regime"].fillna(base_snap["model_family"].astype(str) + "_default")

    cond_rows: list[dict[str, Any]] = []
    snap_rows: list[pd.DataFrame] = []

    annni_cond_path = out_dir / "annni_condition_stats.csv"
    annni_snap_path = out_dir / "annni_snapshot_all.csv"
    done_ids: set[str] = set()
    if annni_cond_path.exists():
        prev = pd.read_csv(annni_cond_path)
        if not prev.empty:
            cond_rows.extend(prev.to_dict(orient="records"))
            done_ids = set(prev["condition_id"].astype(str).tolist())
    if annni_snap_path.exists():
        prevs = pd.read_csv(annni_snap_path)
        if not prevs.empty:
            snap_rows.append(prevs)

    run_id = int(len(cond_rows))
    times = np.linspace(0.0, cfg.t_max, cfg.n_steps)
    for n_sites in cfg.n_sites:
        for kappa in cfg.kappa_values:
            for hz in cfg.hz_values:
                regime = f"stage5_annni_k{kappa:.2f}_hz{hz:.2f}"
                h0_mat = build_annni_hamiltonian(n_sites=n_sites, j1=cfg.j1, kappa=kappa, hx=cfg.h0, hz=hz)
                psi0 = _ground_state(h0_mat)
                for h1 in cfg.h1_values:
                    h1_mat = build_annni_hamiltonian(n_sites=n_sites, j1=cfg.j1, kappa=kappa, hx=h1, hz=hz)
                    states = evolve_state(h1_mat, psi0, times, method="krylov")
                    lam = rate_function(psi0, states, n_sites=n_sites)
                    entropy = bipartite_entropy_time(states, n_sites=n_sites)
                    if n_sites <= cfg.exact_magic_max_sites:
                        m2 = magic_over_time(states=states, n_sites=n_sites, alphas=[2.0])[2.0]
                    elif cfg.approx_magic_samples > 0:
                        seed_base = int(2000 * n_sites + round(100 * h1) + round(30 * kappa) + round(50 * hz))
                        m2 = _approximate_magic_m2_over_time(
                            states=states,
                            n_sites=n_sites,
                            n_pauli_samples=cfg.approx_magic_samples,
                            seed_base=seed_base,
                        )
                    else:
                        raise ValueError("Missing magic path for large n_sites; set approx_magic_samples > 0.")
                    max_lambda = float(np.max(lam))
                    for arch in cfg.architectures:
                        for seed in cfg.seeds:
                            cond_id = (
                                f"annni_N{n_sites}_k{kappa:.3f}_hz{hz:.3f}_h{h1:.3f}_{arch}_s{seed}"
                            )
                            if cond_id in done_ids:
                                continue
                            nn_cfg = _nnqs_cfg(cfg, seed=seed)
                            df_snap, _ = run_snapshot_study(
                                states=states,
                                times=times,
                                n_sites=n_sites,
                                magic_m2=m2,
                                cfg=nn_cfg,
                                device="cpu",
                                model_type=arch,
                            )
                            df_snap = df_snap.copy()
                            df_snap["snapshot_entropy"] = entropy[df_snap["snapshot_index"].to_numpy(dtype=int)]
                            df_snap["condition_id"] = cond_id
                            df_snap["model_family"] = "annni"
                            df_snap["architecture"] = arch
                            df_snap["regime"] = regime
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
                            rp, pp = _partial_corr(x, y, z)
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
                                    "pearson_magic_vs_val_nll_partial_entropy": rp,
                                    "pearson_p_magic_vs_val_nll_partial_entropy": pp,
                                    "max_lambda": max_lambda,
                                    "model_family": "annni",
                                    "regime": regime,
                                    "quench_delta": float(h1 - cfg.h0),
                                }
                            )
                            done_ids.add(cond_id)
                            run_id += 1
                            if run_id % 8 == 0:
                                pd.DataFrame(cond_rows).to_csv(annni_cond_path, index=False)
                                pd.concat(snap_rows, axis=0, ignore_index=True).to_csv(annni_snap_path, index=False)
                                print(f"[annni checkpoint] conditions={len(cond_rows)}", flush=True)

    annni_cond = pd.DataFrame(cond_rows)
    annni_snap = pd.concat(snap_rows, axis=0, ignore_index=True) if snap_rows else pd.DataFrame()
    if len(annni_snap):
        annni_snap = annni_snap.drop_duplicates(
            subset=["condition_id", "snapshot_index", "time", "architecture", "seed"]
        ).reset_index(drop=True)
    annni_cond.to_csv(annni_cond_path, index=False)
    annni_snap.to_csv(annni_snap_path, index=False)

    all_cond = pd.concat([base_cond, annni_cond], ignore_index=True, sort=False)
    all_snap = pd.concat([base_snap, annni_snap], ignore_index=True, sort=False)
    all_snap = all_snap.drop_duplicates(
        subset=["condition_id", "snapshot_index", "time", "architecture", "seed"]
    ).reset_index(drop=True)
    all_cond["regime"] = all_cond["regime"].fillna(all_cond["model_family"].astype(str) + "_default")
    all_snap["regime"] = all_snap["regime"].fillna(all_snap["model_family"].astype(str) + "_default")
    all_cond.to_csv(out_dir / "all_condition_stats_stage5.csv", index=False)
    all_snap.to_csv(out_dir / "all_snapshot_all_stage5.csv", index=False)

    corridor = _corridor_filter(all_cond, lambda_quantile=cfg.lambda_quantile)
    corridor_ids = set(corridor["condition_id"].astype(str).tolist())
    snap_corr = all_snap[all_snap["condition_id"].astype(str).isin(corridor_ids)].copy().reset_index(drop=True)
    corridor.to_csv(out_dir / "corridor_condition_stats_stage5.csv", index=False)
    snap_corr.to_csv(out_dir / "corridor_snapshot_all_stage5.csv", index=False)

    cell_rows: list[dict[str, Any]] = []
    for i, ((model, arch), subc) in enumerate(corridor.groupby(["model_family", "architecture"])):
        subs = snap_corr[(snap_corr["model_family"] == model) & (snap_corr["architecture"] == arch)].copy()
        b, lo, hi = _bootstrap_beta_magic(subs, n_boot=cfg.bootstrap, seed=args.seed + i * 13)
        p_perm = _perm_p_beta_magic(subs, n_perm=cfg.permutations, seed=args.seed + i * 17)
        cell_rows.append(
            {
                "model_family": model,
                "architecture": arch,
                "n_conditions": int(len(subc)),
                "n_snapshots": int(len(subs)),
                "beta_magic": b,
                "beta_magic_ci_low": lo,
                "beta_magic_ci_high": hi,
                "beta_ci_positive": bool(np.isfinite(lo) and lo > 0.0),
                "perm_p_beta_magic": p_perm,
            }
        )
    cell_df = pd.DataFrame(cell_rows).sort_values(["model_family", "architecture"]).reset_index(drop=True)
    cell_df["perm_q_beta_magic"] = _bh_fdr(cell_df["perm_p_beta_magic"].to_numpy(dtype=float))
    cell_df["perm_q_pass"] = cell_df["perm_q_beta_magic"].to_numpy(dtype=float) < cfg.fdr_alpha
    cell_df.to_csv(out_dir / "stage5_cell_metrics.csv", index=False)

    regime_rows: list[dict[str, Any]] = []
    for i, ((model, arch, regime), sub) in enumerate(
        snap_corr.groupby(["model_family", "architecture", "regime"])
    ):
        b, lo, hi = _bootstrap_beta_magic(sub, n_boot=cfg.bootstrap, seed=args.seed + 500 + i * 11)
        regime_rows.append(
            {
                "model_family": model,
                "architecture": arch,
                "regime": regime,
                "n_snapshots": int(len(sub)),
                "beta_magic": b,
                "beta_magic_ci_low": lo,
                "beta_magic_ci_high": hi,
                "beta_ci_positive": bool(np.isfinite(lo) and lo > 0.0),
            }
        )
    regime_df = pd.DataFrame(regime_rows).sort_values(["model_family", "architecture", "regime"]).reset_index(drop=True)
    regime_df.to_csv(out_dir / "stage5_regime_metrics.csv", index=False)

    q_rows: list[dict[str, Any]] = []
    for i, tau in enumerate(cfg.quantiles):
        b, lo, hi = _cluster_boot_quantile(
            snap_corr,
            tau=tau,
            n_boot=max(120, cfg.bootstrap),
            seed=args.seed + 2000 + i * 19,
        )
        q_rows.append(
            {
                "tau": float(tau),
                "beta_magic": b,
                "ci_low": lo,
                "ci_high": hi,
                "positive_lower_bound": bool(np.isfinite(lo) and lo > 0.0),
            }
        )
    q_df = pd.DataFrame(q_rows).sort_values("tau").reset_index(drop=True)
    q_df.to_csv(out_dir / "stage5_quantile_bound.csv", index=False)

    _plot_beta_forest(cell_df, fig_dir / "fig_stage5_beta_forest.png")
    _plot_regime_minimax(regime_df, fig_dir / "fig_stage5_regime_minimax.png")
    _plot_regime_heatmap(regime_df, fig_dir / "fig_stage5_regime_heatmap.png")
    _plot_quantile_bound(q_df, fig_dir / "fig_stage5_quantile_bound.png")

    beta_cell_gate = bool(len(cell_df) > 0 and np.all(cell_df["beta_ci_positive"].to_numpy(dtype=bool)))
    beta_fdr_gate = bool(len(cell_df) > 0 and np.all(cell_df["perm_q_pass"].to_numpy(dtype=bool)))
    regime_gate = bool(len(regime_df) > 0 and np.all(regime_df["beta_ci_positive"].to_numpy(dtype=bool)))
    theory_gate = bool(len(q_df) > 0 and np.all(q_df["positive_lower_bound"].to_numpy(dtype=bool)))

    worst = (
        regime_df.groupby(["model_family", "architecture"], as_index=False)["beta_magic_ci_low"]
        .min()
        .rename(columns={"beta_magic_ci_low": "worst_beta_ci_low"})
    )
    minimax_gate = bool(len(worst) > 0 and np.all(worst["worst_beta_ci_low"].to_numpy(dtype=float) > 0.0))

    combined_gate = bool(beta_cell_gate and beta_fdr_gate and regime_gate and theory_gate and minimax_gate)
    summary = {
        "config": args.config,
        "out_dir": str(out_dir),
        "n_conditions_all_stage5": int(len(all_cond)),
        "n_conditions_corridor_stage5": int(len(corridor)),
        "n_snapshots_corridor_stage5": int(len(snap_corr)),
        "n_families_corridor": int(corridor["model_family"].nunique()),
        "n_regimes_corridor": int(corridor["regime"].nunique()),
        "lambda_quantile": float(cfg.lambda_quantile),
        "fdr_alpha": float(cfg.fdr_alpha),
        "beta_cell_gate": beta_cell_gate,
        "beta_fdr_gate": beta_fdr_gate,
        "regime_gate": regime_gate,
        "theory_quantile_gate": theory_gate,
        "minimax_regime_gate": minimax_gate,
        "combined_external_regime_theory_gate": combined_gate,
    }
    (out_dir / "stage5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    reg = {
        "schema_version": "1.0",
        "study_id": "stage5_prx",
        "phase": "stage5",
        "source_stage3_base_dir": str(base_dir),
        "source_out_dir": str(out_dir),
        "gate_flags": {
            "beta_cell_gate": beta_cell_gate,
            "beta_fdr_gate": beta_fdr_gate,
            "regime_gate": regime_gate,
            "theory_quantile_gate": theory_gate,
            "minimax_regime_gate": minimax_gate,
            "combined_external_regime_theory_gate": combined_gate,
        },
        "cell_records": cell_df.to_dict(orient="records"),
        "regime_records": regime_df.to_dict(orient="records"),
        "quantile_records": q_df.to_dict(orient="records"),
        "minimax_records": worst.to_dict(orient="records"),
    }
    reg_path = ROOT / args.registry_out
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    lines = [
        "# Stage 5 External Model + Regime Expansion",
        "",
        "Adds ANNNI as a fourth family and introduces adversarial regime gates.",
        "",
        f"- all conditions: `{len(all_cond)}`",
        f"- corridor conditions: `{len(corridor)}`",
        f"- corridor snapshots: `{len(snap_corr)}`",
        f"- families in corridor: `{corridor['model_family'].nunique()}`",
        f"- regimes in corridor: `{corridor['regime'].nunique()}`",
        "",
        "## Gates",
        f"- beta cell gate: `{beta_cell_gate}`",
        f"- permutation FDR gate: `{beta_fdr_gate}`",
        f"- regime gate: `{regime_gate}`",
        f"- minimax regime gate: `{minimax_gate}`",
        f"- quantile theory gate: `{theory_gate}`",
        f"- **combined external+regime+theory gate**: `{combined_gate}`",
        "",
        "## Artifacts",
        f"- `{out_dir / 'stage5_summary.json'}`",
        f"- `{out_dir / 'stage5_cell_metrics.csv'}`",
        f"- `{out_dir / 'stage5_regime_metrics.csv'}`",
        f"- `{out_dir / 'stage5_quantile_bound.csv'}`",
        f"- `{fig_dir / 'fig_stage5_beta_forest.png'}`",
        f"- `{fig_dir / 'fig_stage5_regime_minimax.png'}`",
        f"- `{fig_dir / 'fig_stage5_regime_heatmap.png'}`",
        f"- `{fig_dir / 'fig_stage5_quantile_bound.png'}`",
        f"- `{reg_path}`",
    ]
    report_path = ROOT / args.report_out
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_dir / 'stage5_summary.json'}")
    print(f"Wrote {out_dir / 'stage5_cell_metrics.csv'}")
    print(f"Wrote {reg_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
