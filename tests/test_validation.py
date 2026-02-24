from __future__ import annotations

import numpy as np

from tqm.config import ExperimentConfig
from tqm.magic import magic_sanity_check
from tqm.pipeline import run_single_quench
from tqm.schwinger_hamiltonian import build_schwinger_hamiltonian, is_hermitian


def tiny_cfg() -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.model.n_sites = 6
    cfg.model.theta1_values = [0.7]
    cfg.evolution.n_steps = 25
    cfg.evolution.t_max = 2.0
    cfg.magic.alphas = [2.0]
    cfg.nnqs.epochs = 5
    cfg.nnqs.measurement_samples = 512
    return cfg


def test_hermiticity() -> None:
    cfg = tiny_cfg()
    h = build_schwinger_hamiltonian(
        n_sites=cfg.model.n_sites,
        mass=cfg.model.mass,
        coupling_g=cfg.model.coupling_g,
        lattice_spacing=cfg.model.lattice_spacing,
        theta=0.9,
    )
    assert is_hermitian(h.h_total)


def test_dense_krylov_match() -> None:
    cfg = tiny_cfg()
    run = run_single_quench(cfg, theta1=cfg.model.theta1_values[0], compute_dense_krylov=True)
    assert run.norm_drift < 1e-10
    assert run.dense_krylov_error < 1e-8


def test_magic_sanity_hierarchy() -> None:
    vals = magic_sanity_check(n_sites=3, alpha=2.0)
    assert vals["|T>^⊗n"] > vals["|0...0>"]
    assert vals["|T>^⊗n"] > vals["GHZ"]
    assert np.isfinite(vals["|0...0>"])
