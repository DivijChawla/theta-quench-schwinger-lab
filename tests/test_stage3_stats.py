from __future__ import annotations

import numpy as np

from experiments.stage3_universal_extension import _bh_fdr, _meta_i2


def test_bh_fdr_monotone_and_bounded() -> None:
    p = np.array([0.001, 0.01, 0.02, 0.2, 0.5], dtype=float)
    q = _bh_fdr(p)
    assert np.all(np.isfinite(q))
    assert np.all((q >= 0.0) & (q <= 1.0))
    assert q[0] <= q[1] <= q[2]


def test_meta_i2_zero_when_homogeneous() -> None:
    effect = np.array([1.0, 1.0, 1.0], dtype=float)
    ci_low = np.array([0.9, 0.9, 0.9], dtype=float)
    ci_high = np.array([1.1, 1.1, 1.1], dtype=float)
    i2 = _meta_i2(effect=effect, ci_low=ci_low, ci_high=ci_high)
    assert np.isfinite(i2)
    assert i2 == 0.0
