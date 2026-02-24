from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.linalg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqm.config import load_config
from tqm.evolve import evolve_dense, evolve_krylov
from tqm.magic import stabilizer_renyi
from tqm.pipeline import prepare_quench_problem
from tqm.schwinger_hamiltonian import build_schwinger_hamiltonian


@dataclass
class CheckResult:
    name: str
    value: float
    threshold: float
    passed: bool


def _site_mask(n: int, i: int) -> int:
    return 1 << (n - 1 - i)


def _z_vals_for_state(idx: int, n: int) -> np.ndarray:
    z = np.empty(n, dtype=np.float64)
    for i in range(n):
        bit = 1 if (idx & _site_mask(n, i)) else 0
        z[i] = -1.0 if bit else 1.0
    return z


def build_hamiltonian_independent(n: int, mass: float, coupling_g: float, a: float, theta: float) -> np.ndarray:
    dim = 2**n
    h = np.zeros((dim, dim), dtype=np.complex128)

    kinetic = 1.0 / (2.0 * a)
    electric_coeff = 0.5 * (coupling_g**2) * a
    alpha = theta / (2.0 * np.pi)

    # Diagonal part: mass + electric
    for b in range(dim):
        z = _z_vals_for_state(b, n)
        mass_term = 0.5 * mass * np.sum(((-1.0) ** np.arange(n)) * z)

        q = 0.5 * (z + ((-1.0) ** np.arange(n)))
        l = np.zeros(n - 1, dtype=np.float64)
        running = alpha
        for link in range(n - 1):
            running += q[link]
            l[link] = running
        electric_term = electric_coeff * np.sum(l**2)

        h[b, b] = mass_term + electric_term

    # Off-diagonal hopping part: sigma^+ sigma^- + h.c. with coefficient 1/(2a)
    for b in range(dim):
        for i in range(n - 1):
            bi = 1 if (b & _site_mask(n, i)) else 0
            bj = 1 if (b & _site_mask(n, i + 1)) else 0
            if bi != bj:
                b_swapped = b ^ (_site_mask(n, i) | _site_mask(n, i + 1))
                h[b, b_swapped] += kinetic

    # Enforce exact Hermitian symmetrization for numeric robustness.
    h = 0.5 * (h + h.conj().T)
    return h


def _sigma(label: str) -> np.ndarray:
    if label == "I":
        return np.eye(2, dtype=np.complex128)
    if label == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if label == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if label == "Z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    raise ValueError(label)


def _pauli_from_masks(n: int, x_mask: int, z_mask: int) -> np.ndarray:
    ops = []
    for i in range(n):
        shift = n - 1 - i
        x = (x_mask >> shift) & 1
        z = (z_mask >> shift) & 1
        if x == 0 and z == 0:
            ops.append(_sigma("I"))
        elif x == 1 and z == 0:
            ops.append(_sigma("X"))
        elif x == 0 and z == 1:
            ops.append(_sigma("Z"))
        else:
            ops.append(_sigma("Y"))
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def brute_force_magic_m2(psi: np.ndarray, n: int) -> float:
    d = 2**n
    total = 0.0
    for x_mask in range(2**n):
        for z_mask in range(2**n):
            p = _pauli_from_masks(n, x_mask, z_mask)
            c = np.vdot(psi, p @ psi)
            xi = (np.real(c) ** 2) / d
            total += xi**2
    return float(-np.log(total) - np.log(d))


def _run_check(name: str, value: float, threshold: float, cmp: Callable[[float, float], bool]) -> CheckResult:
    return CheckResult(name=name, value=value, threshold=threshold, passed=cmp(value, threshold))


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent manual verification for theta-quench project")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-small", type=int, default=5, help="small N for independent brute-force checks")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    cfg = load_config(args.config)

    n_small = min(args.n_small, cfg.model.n_sites)
    th = float(cfg.model.theta1_values[-1])

    h_pkg = build_schwinger_hamiltonian(
        n_sites=n_small,
        mass=cfg.model.mass,
        coupling_g=cfg.model.coupling_g,
        lattice_spacing=cfg.model.lattice_spacing,
        theta=th,
    ).h_total.toarray()
    h_ref = build_hamiltonian_independent(
        n=n_small,
        mass=cfg.model.mass,
        coupling_g=cfg.model.coupling_g,
        a=cfg.model.lattice_spacing,
        theta=th,
    )

    h_diff = float(np.max(np.abs(h_pkg - h_ref)))

    # Dynamics checks on small system
    cfg_small = load_config(args.config)
    cfg_small.model.n_sites = n_small
    cfg_small.evolution.n_steps = min(cfg_small.evolution.n_steps, 41)
    cfg_small.evolution.t_max = min(cfg_small.evolution.t_max, 2.0)
    prep = prepare_quench_problem(cfg_small, theta1=th)
    times = cfg_small.times()

    states_dense = evolve_dense(prep.h1.h_total, prep.psi0, times)
    states_krylov = evolve_krylov(prep.h1.h_total, prep.psi0, times)
    dyn_diff = float(np.max(np.linalg.norm(states_dense - states_krylov, axis=1)))

    norms = np.sum(np.abs(states_dense) ** 2, axis=1)
    norm_drift = float(np.max(np.abs(norms - 1.0)))

    # One-time independent expm check
    t0 = 0.73
    with np.errstate(all="ignore"):
        u = scipy.linalg.expm((-1j) * prep.h1.h_total.toarray() * t0)
        psi_expm = u @ prep.psi0
    psi_dense = evolve_dense(prep.h1.h_total, prep.psi0, np.array([0.0, t0]))[1]
    expm_diff = float(np.linalg.norm(psi_expm - psi_dense))

    # Magic check at n<=4 for brute-force Pauli agreement
    rng = np.random.default_rng(args.seed)
    n_magic = min(4, n_small)
    dim = 2**n_magic
    raw = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi = raw / np.linalg.norm(raw)
    m2_fast = stabilizer_renyi(psi=psi, n_sites=n_magic, alpha=2.0, z_batch_size=64)
    m2_bruteforce = brute_force_magic_m2(psi=psi, n=n_magic)
    magic_diff = float(abs(m2_fast - m2_bruteforce))

    checks = [
        _run_check("H_pkg_vs_independent_max_abs", h_diff, 1e-10, lambda v, t: v < t),
        _run_check("dense_vs_krylov_state_diff", dyn_diff, 1e-10, lambda v, t: v < t),
        _run_check("dense_norm_drift", norm_drift, 1e-12, lambda v, t: v < t),
        _run_check("dense_vs_expm_state_diff", expm_diff, 1e-10, lambda v, t: v < t),
        _run_check("magic_fast_vs_bruteforce_M2_diff", magic_diff, 1e-10, lambda v, t: v < t),
    ]

    print("Independent verification checks")
    print("=" * 72)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        print(f"{status:4s} | {c.name:36s} | value={c.value:.3e} | threshold={c.threshold:.1e}")

    n_pass = sum(int(c.passed) for c in checks)
    print("-" * 72)
    print(f"Summary: {n_pass}/{len(checks)} passed")

    if n_pass != len(checks):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
