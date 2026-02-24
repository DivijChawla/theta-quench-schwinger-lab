from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp


@dataclass
class SchwingerComponents:
    h_total: sp.csr_matrix
    h_pm: sp.csr_matrix
    h_zz: sp.csr_matrix
    h_z: sp.csr_matrix
    h_const: float
    linear_coeffs: np.ndarray
    pair_coeffs: dict[tuple[int, int], float]


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


@lru_cache(maxsize=None)
def identity(n_sites: int) -> sp.csr_matrix:
    dim = 2**n_sites
    return sp.eye(dim, format="csr", dtype=np.complex128)


def couplings_from_physical_params(mass: float, coupling_g: float, lattice_spacing: float) -> tuple[float, float, float]:
    kinetic = 1.0 / (2.0 * lattice_spacing)
    mass_coeff = mass / 2.0
    electric = 0.5 * (coupling_g**2) * lattice_spacing
    return kinetic, mass_coeff, electric


def electric_decomposition_coeffs(n_sites: int, theta: float) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
    if n_sites < 2:
        raise ValueError("n_sites must be >= 2")

    alpha = theta / (2.0 * np.pi)
    s = np.array([(-1) ** i for i in range(n_sites)], dtype=np.float64)
    n_links = n_sites - 1

    linear = np.zeros(n_sites, dtype=np.float64)
    pair: dict[tuple[int, int], float] = {}

    w_i = np.maximum(n_links - np.arange(n_sites), 0)
    const = n_links * (alpha**2)
    linear += alpha * w_i
    const += alpha * float(np.sum(w_i * s))

    for i in range(n_sites):
        w_ii = max(n_links - i, 0)
        if w_ii <= 0:
            continue
        const += 0.5 * w_ii
        linear[i] += 0.5 * w_ii * s[i]

    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            w_ij = max(n_links - max(i, j), 0)
            if w_ij <= 0:
                continue
            pair[(i, j)] = pair.get((i, j), 0.0) + 0.5 * w_ij
            linear[i] += 0.5 * w_ij * s[j]
            linear[j] += 0.5 * w_ij * s[i]
            const += 0.5 * w_ij * s[i] * s[j]

    return linear, pair, const


def build_schwinger_hamiltonian(
    n_sites: int,
    mass: float,
    coupling_g: float,
    lattice_spacing: float,
    theta: float,
) -> SchwingerComponents:
    kinetic, mass_coeff, electric_coeff = couplings_from_physical_params(
        mass=mass,
        coupling_g=coupling_g,
        lattice_spacing=lattice_spacing,
    )

    h_pm = sp.csr_matrix((2**n_sites, 2**n_sites), dtype=np.complex128)
    for i in range(n_sites - 1):
        xx = _two_site_pauli(n_sites, i, i + 1, "X", "X")
        yy = _two_site_pauli(n_sites, i, i + 1, "Y", "Y")
        h_pm = h_pm + 0.5 * kinetic * (xx + yy)

    linear_e, pair_e, const_e = electric_decomposition_coeffs(n_sites=n_sites, theta=theta)

    h_zz = sp.csr_matrix((2**n_sites, 2**n_sites), dtype=np.complex128)
    for (i, j), coeff in pair_e.items():
        h_zz = h_zz + electric_coeff * coeff * _two_site_pauli(n_sites, i, j, "Z", "Z")

    h_z = sp.csr_matrix((2**n_sites, 2**n_sites), dtype=np.complex128)
    for i in range(n_sites):
        mass_sign = (-1) ** i
        coeff = mass_coeff * mass_sign + electric_coeff * linear_e[i]
        h_z = h_z + coeff * _local_pauli(n_sites, i, "Z")

    h_const = electric_coeff * const_e
    h_total = (h_pm + h_zz + h_z + h_const * identity(n_sites)).tocsr()

    return SchwingerComponents(
        h_total=h_total,
        h_pm=h_pm.tocsr(),
        h_zz=h_zz.tocsr(),
        h_z=h_z.tocsr(),
        h_const=float(np.real(h_const)),
        linear_coeffs=linear_e,
        pair_coeffs=pair_e,
    )


def is_hermitian(matrix: sp.csr_matrix, atol: float = 1e-10) -> bool:
    diff = matrix - matrix.getH()
    if diff.nnz == 0:
        return True
    return np.max(np.abs(diff.data)) < atol
