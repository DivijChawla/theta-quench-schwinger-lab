from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .ed import ground_state
from .evolve import dense_vs_krylov_error, evolve_state, norm_drift
from .loschmidt import rate_function
from .magic import magic_over_time, magic_sanity_check
from .observables import (
    bipartite_entropy_time,
    connected_correlator_time,
    energy_density,
    staggered_mass_from_z,
    z_expectations_time,
)
from .schwinger_hamiltonian import build_schwinger_hamiltonian, is_hermitian


@dataclass
class QuenchRun:
    theta0: float
    theta1: float
    times: np.ndarray
    energy_density: np.ndarray
    staggered_mass: np.ndarray
    connected_corr_01: np.ndarray
    loschmidt_rate: np.ndarray
    entanglement_entropy: np.ndarray
    magic: dict[float, np.ndarray]
    norm_drift: float
    dense_krylov_error: float
    hermitian_h0: bool
    hermitian_h1: bool
    state_trajectory: np.ndarray


@dataclass
class PreparedProblem:
    h0: Any
    h1: Any
    e0: float
    psi0: np.ndarray


def prepare_quench_problem(cfg: ExperimentConfig, theta1: float) -> PreparedProblem:
    n = cfg.model.n_sites
    h0 = build_schwinger_hamiltonian(
        n_sites=n,
        mass=cfg.model.mass,
        coupling_g=cfg.model.coupling_g,
        lattice_spacing=cfg.model.lattice_spacing,
        theta=cfg.model.theta0,
    )
    h1 = build_schwinger_hamiltonian(
        n_sites=n,
        mass=cfg.model.mass,
        coupling_g=cfg.model.coupling_g,
        lattice_spacing=cfg.model.lattice_spacing,
        theta=theta1,
    )

    e0, psi0 = ground_state(h0.h_total)
    return PreparedProblem(h0=h0, h1=h1, e0=e0, psi0=psi0)


def run_single_quench(
    cfg: ExperimentConfig,
    theta1: float,
    compute_dense_krylov: bool = False,
) -> QuenchRun:
    n = cfg.model.n_sites
    times = cfg.times()
    prep = prepare_quench_problem(cfg, theta1)

    method = cfg.evolution.method
    if method == "krylov" and n <= cfg.evolution.dense_max_sites:
        method = "dense"

    states = evolve_state(prep.h1.h_total, prep.psi0, times, method=method)

    z_t = z_expectations_time(states, n)
    staggered = staggered_mass_from_z(z_t)

    mag: dict[float, np.ndarray] = {}
    if cfg.magic.enabled and n <= cfg.magic.max_sites_exact:
        mag = magic_over_time(
            states=states,
            n_sites=n,
            alphas=[float(a) for a in cfg.magic.alphas],
            z_batch_size=cfg.magic.z_batch_size,
        )

    dk_error = float("nan")
    if compute_dense_krylov and n <= cfg.evolution.dense_max_sites:
        dk_error = dense_vs_krylov_error(prep.h1.h_total, prep.psi0, times)

    return QuenchRun(
        theta0=cfg.model.theta0,
        theta1=theta1,
        times=times,
        energy_density=energy_density(states, prep.h1.h_total, n),
        staggered_mass=staggered,
        connected_corr_01=connected_correlator_time(states, n, 0, min(1, n - 1)),
        loschmidt_rate=rate_function(prep.psi0, states, n_sites=n),
        entanglement_entropy=bipartite_entropy_time(states, n_sites=n),
        magic=mag,
        norm_drift=norm_drift(states),
        dense_krylov_error=dk_error,
        hermitian_h0=is_hermitian(prep.h0.h_total),
        hermitian_h1=is_hermitian(prep.h1.h_total),
        state_trajectory=states,
    )


def save_quench_run(run: QuenchRun, out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "theta0": np.array([run.theta0], dtype=np.float64),
        "theta1": np.array([run.theta1], dtype=np.float64),
        "times": run.times,
        "energy_density": run.energy_density,
        "staggered_mass": run.staggered_mass,
        "connected_corr_01": run.connected_corr_01,
        "loschmidt_rate": run.loschmidt_rate,
        "entanglement_entropy": run.entanglement_entropy,
        "norm_drift": np.array([run.norm_drift]),
        "dense_krylov_error": np.array([run.dense_krylov_error]),
        "hermitian_h0": np.array([run.hermitian_h0]),
        "hermitian_h1": np.array([run.hermitian_h1]),
        "states_real": np.real(run.state_trajectory),
        "states_imag": np.imag(run.state_trajectory),
    }
    for alpha, values in run.magic.items():
        payload[f"magic_alpha_{alpha:g}"] = values

    np.savez(out, **payload)


def load_states_from_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    states = data["states_real"] + 1j * data["states_imag"]
    return data["times"], states


def run_magic_sanity(cfg: ExperimentConfig) -> dict[str, float]:
    alpha = float(cfg.magic.alphas[0]) if cfg.magic.alphas else 2.0
    n = min(cfg.model.n_sites, 5)
    return magic_sanity_check(n_sites=n, alpha=alpha)
