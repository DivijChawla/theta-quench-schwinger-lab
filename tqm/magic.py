from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .pauli import iter_pauli_expectations


@dataclass
class MagicResult:
    alphas: list[float]
    values: dict[float, float]


def _real_pauli_expectations(exps: np.ndarray) -> np.ndarray:
    if np.max(np.abs(np.imag(exps))) > 1e-7:
        # Hermitian Paulis should give real expectation values; keep robust to small drift.
        return np.real(exps)
    return np.real(exps)


def stabilizer_renyi(
    psi: np.ndarray,
    n_sites: int,
    alpha: float = 2.0,
    z_batch_size: int = 128,
) -> float:
    if abs(alpha - 1.0) < 1e-12:
        raise ValueError("alpha=1 is not implemented in this routine")

    d = 2**n_sites
    accum = 0.0
    for _, _, exps in iter_pauli_expectations(psi, n_sites=n_sites, z_batch_size=z_batch_size):
        c = _real_pauli_expectations(exps)
        xi = np.maximum((c**2) / d, 0.0)
        accum += float(np.sum(xi**alpha))

    return float((1.0 / (1.0 - alpha)) * math.log(accum) - math.log(d))


def stabilizer_renyi_multi_alpha(
    psi: np.ndarray,
    n_sites: int,
    alphas: list[float],
    z_batch_size: int = 128,
) -> MagicResult:
    d = 2**n_sites
    accum = {float(alpha): 0.0 for alpha in alphas}

    for _, _, exps in iter_pauli_expectations(psi, n_sites=n_sites, z_batch_size=z_batch_size):
        c = _real_pauli_expectations(exps)
        xi = np.maximum((c**2) / d, 0.0)
        for alpha in alphas:
            if abs(alpha - 1.0) < 1e-12:
                raise ValueError("alpha=1 is not implemented in this routine")
            accum[float(alpha)] += float(np.sum(xi**alpha))

    values = {
        alpha: float((1.0 / (1.0 - alpha)) * math.log(val) - math.log(d)) for alpha, val in accum.items()
    }
    return MagicResult(alphas=[float(a) for a in alphas], values=values)


def magic_over_time(
    states: np.ndarray,
    n_sites: int,
    alphas: list[float],
    z_batch_size: int = 128,
) -> dict[float, np.ndarray]:
    out = {float(alpha): np.zeros(states.shape[0], dtype=np.float64) for alpha in alphas}
    for t_idx, psi in enumerate(states):
        result = stabilizer_renyi_multi_alpha(
            psi=psi,
            n_sites=n_sites,
            alphas=alphas,
            z_batch_size=z_batch_size,
        )
        for alpha in alphas:
            out[float(alpha)][t_idx] = result.values[float(alpha)]
    return out


def computational_zero_state(n_sites: int) -> np.ndarray:
    psi = np.zeros(2**n_sites, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def ghz_state(n_sites: int) -> np.ndarray:
    psi = np.zeros(2**n_sites, dtype=np.complex128)
    psi[0] = 1.0 / np.sqrt(2.0)
    psi[-1] = 1.0 / np.sqrt(2.0)
    return psi


def t_state_tensor(n_sites: int) -> np.ndarray:
    single = np.array([1.0, np.exp(1j * np.pi / 4.0)], dtype=np.complex128) / np.sqrt(2.0)
    psi = single.copy()
    for _ in range(n_sites - 1):
        psi = np.kron(psi, single)
    return psi


def magic_sanity_check(n_sites: int = 4, alpha: float = 2.0) -> dict[str, float]:
    z = stabilizer_renyi(computational_zero_state(n_sites), n_sites=n_sites, alpha=alpha)
    ghz = stabilizer_renyi(ghz_state(n_sites), n_sites=n_sites, alpha=alpha)
    t = stabilizer_renyi(t_state_tensor(n_sites), n_sites=n_sites, alpha=alpha)
    return {"|0...0>": z, "GHZ": ghz, "|T>^⊗n": t}
