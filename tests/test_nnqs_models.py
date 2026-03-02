from __future__ import annotations

import numpy as np

from tqm.config import NNQSConfig
from tqm.nnqs.train import train_nnqs_on_state


def _basis_state(n_sites: int, index: int = 0) -> np.ndarray:
    psi = np.zeros(2**n_sites, dtype=np.complex128)
    psi[index] = 1.0
    return psi


def test_made_training_smoke() -> None:
    cfg = NNQSConfig(
        enabled=True,
        snapshot_count=1,
        measurement_samples=256,
        val_fraction=0.2,
        hidden_size=16,
        epochs=3,
        lr=1e-3,
        batch_size=64,
        threshold_nll=3.0,
        seed=3,
    )
    res = train_nnqs_on_state(
        psi=_basis_state(4),
        n_sites=4,
        cfg=cfg,
        seed=3,
        model_type="made",
    )
    assert np.isfinite(res.final_val_nll)
    assert np.isfinite(res.final_kl)
