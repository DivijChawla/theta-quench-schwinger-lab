from __future__ import annotations

import numpy as np

from tqm.config import ExperimentConfig
from tqm.evolve import evolve_state
from tqm.pipeline import prepare_quench_problem


def test_sector_krylov_matches_krylov_for_single_sector_state() -> None:
    cfg = ExperimentConfig()
    cfg.model.n_sites = 6
    cfg.evolution.n_steps = 15
    cfg.evolution.t_max = 1.5
    prep = prepare_quench_problem(cfg, theta1=0.7)

    times = cfg.times()
    full = evolve_state(prep.h1.h_total, prep.psi0, times, method="krylov")
    sec = evolve_state(prep.h1.h_total, prep.psi0, times, method="sector_krylov")
    err = np.max(np.linalg.norm(full - sec, axis=1))
    assert err < 1e-8
