from __future__ import annotations

from collections.abc import Iterator

import numpy as np

PAULI_PHASE = np.array([1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j], dtype=np.complex128)
POPCOUNT16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


def popcount_u32(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.uint32)
    return POPCOUNT16[x & np.uint32(0xFFFF)] + POPCOUNT16[(x >> np.uint32(16)) & np.uint32(0xFFFF)]


def basis_indices(n_sites: int) -> np.ndarray:
    return np.arange(2**n_sites, dtype=np.uint32)


def pauli_expectation_batch_for_x(
    psi: np.ndarray,
    x_mask: int,
    z_masks: np.ndarray,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    n_dim = psi.shape[0]
    if indices is None:
        indices = np.arange(n_dim, dtype=np.uint32)

    z_masks = np.asarray(z_masks, dtype=np.uint32)
    x_mask_u32 = np.uint32(x_mask)

    permuted = psi[np.bitwise_xor(indices, x_mask_u32)]
    overlap = psi.conj() * permuted

    and_vals = np.bitwise_and(indices[None, :], z_masks[:, None])
    parity_bits = (popcount_u32(and_vals).astype(np.int16) & 1).astype(np.int16)
    parity = 1 - 2 * parity_bits
    fourier = np.einsum("bd,d->b", parity.astype(np.float64), overlap)

    y_counts = popcount_u32(np.bitwise_and(z_masks, x_mask_u32))
    phases = PAULI_PHASE[y_counts % 4]
    return phases * fourier


def iter_pauli_expectations(
    psi: np.ndarray,
    n_sites: int,
    z_batch_size: int = 128,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    n_masks = 2**n_sites
    indices = basis_indices(n_sites)
    z_full = np.arange(n_masks, dtype=np.uint32)

    for x_mask in range(n_masks):
        for start in range(0, n_masks, z_batch_size):
            z_batch = z_full[start : start + z_batch_size]
            exps = pauli_expectation_batch_for_x(
                psi=psi,
                x_mask=x_mask,
                z_masks=z_batch,
                indices=indices,
            )
            yield x_mask, z_batch, exps


def pauli_string_from_masks(x_mask: int, z_mask: int, n_sites: int) -> str:
    chars = []
    for i in range(n_sites):
        shift = n_sites - 1 - i
        x = (x_mask >> shift) & 1
        z = (z_mask >> shift) & 1
        if x == 0 and z == 0:
            chars.append("I")
        elif x == 1 and z == 0:
            chars.append("X")
        elif x == 0 and z == 1:
            chars.append("Z")
        else:
            chars.append("Y")
    return "".join(chars)


def all_pauli_expectations(
    psi: np.ndarray,
    n_sites: int,
    z_batch_size: int = 128,
) -> np.ndarray:
    n_masks = 2**n_sites
    out = np.zeros((n_masks, n_masks), dtype=np.complex128)
    for x_mask, z_batch, exps in iter_pauli_expectations(psi, n_sites, z_batch_size=z_batch_size):
        out[x_mask, z_batch] = exps
    return out
