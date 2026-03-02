from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.stage4_thermo_theory import _fit_size_extrapolation, _weighted_line_fit


def test_weighted_line_fit_basic() -> None:
    x = np.array([1.0 / 6.0, 1.0 / 8.0, 1.0 / 10.0], dtype=float)
    y = np.array([0.9, 1.0, 1.1], dtype=float)
    se = np.array([0.05, 0.05, 0.05], dtype=float)
    slope, intercept = _weighted_line_fit(x, y, se)
    assert np.isfinite(slope)
    assert np.isfinite(intercept)


def test_fit_size_extrapolation_smoke() -> None:
    df = pd.DataFrame(
        {
            "model_family": ["xxz", "xxz", "xxz"],
            "architecture": ["gru", "gru", "gru"],
            "n_sites": [6, 8, 10],
            "beta_magic": [0.7, 0.8, 0.9],
            "beta_ci_low": [0.5, 0.6, 0.7],
            "beta_ci_high": [0.9, 1.0, 1.1],
        }
    )
    out = _fit_size_extrapolation(df, n_boot=50, seed=123)
    assert len(out) == 1
    assert out.loc[0, "model_family"] == "xxz"
