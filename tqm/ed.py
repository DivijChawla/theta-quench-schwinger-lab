from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def normalize_state(psi: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(psi)
    if norm == 0:
        raise ValueError("Zero-norm state")
    return psi / norm


def ground_state(
    hamiltonian: sp.csr_matrix,
    dense_max_dim: int = 2**11,
) -> tuple[float, np.ndarray]:
    dim = hamiltonian.shape[0]
    if dim <= dense_max_dim:
        h_dense = hamiltonian.toarray()
        evals, evecs = np.linalg.eigh(h_dense)
        e0 = float(np.real(evals[0]))
        psi0 = normalize_state(evecs[:, 0])
        return e0, psi0

    evals, evecs = spla.eigsh(hamiltonian, k=1, which="SA")
    idx = int(np.argmin(np.real(evals)))
    e0 = float(np.real(evals[idx]))
    psi0 = normalize_state(evecs[:, idx])
    return e0, psi0


def dense_spectrum(hamiltonian: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    h_dense = hamiltonian.toarray()
    evals, evecs = scipy.linalg.eigh(h_dense)
    return np.real(evals), evecs
