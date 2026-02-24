from __future__ import annotations

import numpy as np


def loschmidt_echo(psi0: np.ndarray, states: np.ndarray) -> np.ndarray:
    amp = np.einsum("td,d->t", states, psi0.conj())
    return np.abs(amp) ** 2


def rate_function(psi0: np.ndarray, states: np.ndarray, n_sites: int, eps: float = 1e-15) -> np.ndarray:
    l = np.clip(loschmidt_echo(psi0, states), eps, 1.0)
    return -(1.0 / n_sites) * np.log(l)
