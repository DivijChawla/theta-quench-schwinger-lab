from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _check_times(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a non-empty 1D array")
    if np.any(np.diff(t) < -1e-12):
        raise ValueError("times must be sorted in ascending order")
    return t


def evolve_dense(hamiltonian: sp.csr_matrix, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    t = _check_times(times)
    h_dense = hamiltonian.toarray()
    evals, evecs = np.linalg.eigh(h_dense)
    coeffs = np.einsum("ji,j->i", evecs.conj(), psi0)
    phase = np.exp(-1j * np.outer(t, evals))
    states = phase * coeffs[None, :]
    out = np.einsum("ti,ji->tj", states, evecs)
    return out


def evolve_krylov(hamiltonian: sp.csr_matrix, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    t = _check_times(times)
    a = (-1j) * hamiltonian
    if t.size > 1:
        dt = np.diff(t)
        if np.max(np.abs(dt - dt[0])) < 1e-12:
            return np.asarray(
                spla.expm_multiply(
                    a,
                    psi0,
                    start=float(t[0]),
                    stop=float(t[-1]),
                    num=int(t.size),
                    endpoint=True,
                )
            )

    out = np.zeros((t.size, psi0.size), dtype=np.complex128)
    for i, time in enumerate(t):
        out[i] = spla.expm_multiply(a * float(time), psi0)
    return out


def evolve_trotter_second_order(
    h_terms: list[sp.csr_matrix],
    psi0: np.ndarray,
    times: np.ndarray,
    n_steps_per_unit: int = 40,
) -> np.ndarray:
    """Second-order Suzuki evolution for a list of Hermitian terms.

    This method is optional and mainly useful for gate-model relevance checks.
    """
    t = _check_times(times)
    if len(h_terms) == 0:
        raise ValueError("h_terms must contain at least one term")

    dim = psi0.size
    out = np.zeros((t.size, dim), dtype=np.complex128)
    out[0] = psi0

    current_state = psi0.copy()
    current_t = t[0]
    for idx in range(1, t.size):
        target_t = t[idx]
        delta_t = target_t - current_t
        n_sub = max(1, int(np.ceil(abs(delta_t) * n_steps_per_unit)))
        dt = delta_t / n_sub

        half_unitaries = [spla.expm((-1j) * (dt / 2.0) * term).tocsr() for term in h_terms]
        reverse_unitaries = list(reversed(half_unitaries))

        for _ in range(n_sub):
            for u in half_unitaries:
                current_state = u @ current_state
            for u in reverse_unitaries:
                current_state = u @ current_state

        out[idx] = current_state
        current_t = target_t

    return out


def evolve_state(
    hamiltonian: sp.csr_matrix,
    psi0: np.ndarray,
    times: np.ndarray,
    method: str = "krylov",
) -> np.ndarray:
    if method == "krylov":
        return evolve_krylov(hamiltonian, psi0, times)
    if method == "dense":
        return evolve_dense(hamiltonian, psi0, times)
    raise ValueError(f"Unknown evolution method: {method}")


def norm_drift(states: np.ndarray) -> float:
    norms = np.sum(np.abs(states) ** 2, axis=1)
    return float(np.max(np.abs(norms - 1.0)))


def dense_vs_krylov_error(
    hamiltonian: sp.csr_matrix,
    psi0: np.ndarray,
    times: np.ndarray,
) -> float:
    dense = evolve_dense(hamiltonian, psi0, times)
    krylov = evolve_krylov(hamiltonian, psi0, times)
    err = np.max(np.linalg.norm(dense - krylov, axis=1))
    return float(err)
