from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _site_masks(n_sites: int) -> np.ndarray:
    return np.array([1 << (n_sites - 1 - i) for i in range(n_sites)], dtype=np.uint32)


def _z_eigs(n_sites: int) -> np.ndarray:
    dim = 2**n_sites
    idx = np.arange(dim, dtype=np.uint32)
    masks = _site_masks(n_sites)
    eigs = np.empty((n_sites, dim), dtype=np.float64)
    for i, m in enumerate(masks):
        bits = (idx & m) > 0
        eigs[i] = np.where(bits, -1.0, 1.0)
    return eigs


def energy_density(states: np.ndarray, hamiltonian: sp.csr_matrix, n_sites: int) -> np.ndarray:
    out = np.zeros(states.shape[0], dtype=np.float64)
    for t_idx, psi in enumerate(states):
        e = np.vdot(psi, hamiltonian @ psi)
        out[t_idx] = float(np.real(e) / n_sites)
    return out


def z_expectations(psi: np.ndarray, n_sites: int) -> np.ndarray:
    probs = np.abs(psi) ** 2
    eigs = _z_eigs(n_sites)
    return eigs @ probs


def z_expectations_time(states: np.ndarray, n_sites: int) -> np.ndarray:
    eigs = _z_eigs(n_sites)
    probs = np.abs(states) ** 2
    return np.einsum("td,nd->tn", probs, eigs)


def staggered_mass_from_z(z_vals: np.ndarray) -> np.ndarray:
    n_sites = z_vals.shape[-1]
    stagger = np.array([(-1) ** i for i in range(n_sites)], dtype=np.float64)
    if z_vals.ndim == 1:
        return np.array([float(np.mean(stagger * z_vals))])
    return np.mean(z_vals * stagger[None, :], axis=1)


def connected_correlator_matrix(psi: np.ndarray, n_sites: int) -> np.ndarray:
    probs = np.abs(psi) ** 2
    eigs = _z_eigs(n_sites)
    z = eigs @ probs
    corr = np.zeros((n_sites, n_sites), dtype=np.float64)
    for i in range(n_sites):
        for j in range(n_sites):
            zz = np.sum(eigs[i] * eigs[j] * probs)
            corr[i, j] = zz - z[i] * z[j]
    return corr


def connected_correlator_time(states: np.ndarray, n_sites: int, i: int, j: int) -> np.ndarray:
    eigs = _z_eigs(n_sites)
    zi = eigs[i]
    zj = eigs[j]
    probs = np.abs(states) ** 2
    exp_i = np.einsum("td,d->t", probs, zi)
    exp_j = np.einsum("td,d->t", probs, zj)
    exp_ij = np.einsum("td,d->t", probs, zi * zj)
    return exp_ij - exp_i * exp_j


def bipartite_entropy(psi: np.ndarray, n_sites: int, cut: int | None = None, eps: float = 1e-14) -> float:
    if cut is None:
        cut = n_sites // 2
    left_dim = 2**cut
    right_dim = 2 ** (n_sites - cut)
    rho = psi.reshape(left_dim, right_dim)
    svals = np.linalg.svd(rho, compute_uv=False)
    probs = np.clip(svals**2, eps, 1.0)
    probs = probs / probs.sum()
    return float(-np.sum(probs * np.log(probs)))


def bipartite_entropy_time(states: np.ndarray, n_sites: int, cut: int | None = None) -> np.ndarray:
    return np.array([bipartite_entropy(psi, n_sites, cut=cut) for psi in states], dtype=np.float64)
