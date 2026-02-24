from __future__ import annotations

import numpy as np


def state_probabilities(psi: np.ndarray) -> np.ndarray:
    probs = np.abs(psi) ** 2
    total = probs.sum()
    if total <= 0:
        raise ValueError("Invalid state with non-positive norm")
    return probs / total


def indices_to_bits(indices: np.ndarray, n_sites: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.uint32)
    shifts = np.arange(n_sites - 1, -1, -1, dtype=np.uint32)
    return ((idx[:, None] >> shifts[None, :]) & 1).astype(np.int64)


def bits_to_indices(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint32)
    n_sites = bits.shape[1]
    shifts = np.arange(n_sites - 1, -1, -1, dtype=np.uint32)
    return np.sum(bits * (1 << shifts)[None, :], axis=1, dtype=np.uint32)


def sample_bitstrings_from_state(
    psi: np.ndarray,
    n_sites: int,
    num_samples: int,
    seed: int = 0,
) -> np.ndarray:
    probs = state_probabilities(psi)
    rng = np.random.default_rng(seed)
    indices = rng.choice(np.arange(2**n_sites), size=num_samples, p=probs)
    return indices_to_bits(indices, n_sites)


def all_bitstrings(n_sites: int) -> np.ndarray:
    return indices_to_bits(np.arange(2**n_sites, dtype=np.uint32), n_sites)


def train_val_split(samples: np.ndarray, val_fraction: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = samples.shape[0]
    perm = rng.permutation(n)
    n_val = int(round(val_fraction * n))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return samples[train_idx], samples[val_idx]


def total_variation_distance(p_true: np.ndarray, p_model: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(p_true - p_model)))
