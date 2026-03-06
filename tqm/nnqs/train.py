from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from torch.utils.data import DataLoader, TensorDataset

from ..config import NNQSConfig
from .data import all_bitstrings, sample_bitstrings_from_state, state_probabilities, total_variation_distance, train_val_split
from .model import AutoregressiveGRU, AutoregressiveMADE


@dataclass
class NNQSTrainResult:
    final_train_nll: float
    final_val_nll: float
    final_kl: float
    final_tv: float
    steps_to_threshold: int
    history: dict[str, list[float]]


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def _resolve_device(device: str | None) -> torch.device:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
    if requested.startswith("mps"):
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device, but torch.backends.mps.is_available() is False.")
    return torch.device(requested)


def _build_model(model_type: str, n_sites: int, hidden_size: int) -> torch.nn.Module:
    m = model_type.lower()
    if m == "gru":
        return AutoregressiveGRU(n_sites=n_sites, hidden_size=hidden_size)
    if m == "made":
        return AutoregressiveMADE(n_sites=n_sites, hidden_size=hidden_size)
    raise ValueError(f"Unsupported model_type: {model_type}")


def train_nnqs_on_state(
    psi: np.ndarray,
    n_sites: int,
    cfg: NNQSConfig,
    seed: int | None = None,
    device: str = "cpu",
    model_type: str = "gru",
) -> NNQSTrainResult:
    if seed is None:
        seed = cfg.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _resolve_device(device)

    samples = sample_bitstrings_from_state(
        psi=psi,
        n_sites=n_sites,
        num_samples=cfg.measurement_samples,
        seed=seed,
    )
    train_np, val_np = train_val_split(samples, val_fraction=cfg.val_fraction, seed=seed)

    train_t = _to_tensor(train_np, dev)
    val_t = _to_tensor(val_np, dev)

    train_loader = DataLoader(TensorDataset(train_t), batch_size=cfg.batch_size, shuffle=True)

    model = _build_model(model_type=model_type, n_sites=n_sites, hidden_size=cfg.hidden_size).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    hist_train: list[float] = []
    hist_val: list[float] = []
    steps_to_threshold = -1

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        count = 0
        for (batch,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = model.nll(batch).mean()
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * batch.shape[0]
            count += int(batch.shape[0])

        train_nll = running / max(count, 1)

        model.eval()
        with torch.no_grad():
            val_nll = float(model.nll(val_t).mean().item())

        hist_train.append(train_nll)
        hist_val.append(val_nll)

        if steps_to_threshold < 0 and val_nll <= cfg.threshold_nll:
            steps_to_threshold = epoch + 1

    true_probs = state_probabilities(psi)
    all_bits = _to_tensor(all_bitstrings(n_sites), dev)
    with torch.no_grad():
        logp_model = model.log_prob(all_bits).cpu().numpy()
    logp_model = logp_model - logsumexp(logp_model)
    p_model = np.exp(logp_model)

    eps = 1e-15
    kl = float(np.sum(true_probs * (np.log(true_probs + eps) - np.log(p_model + eps))))
    tv = total_variation_distance(true_probs, p_model)

    return NNQSTrainResult(
        final_train_nll=float(hist_train[-1]),
        final_val_nll=float(hist_val[-1]),
        final_kl=kl,
        final_tv=tv,
        steps_to_threshold=steps_to_threshold,
        history={"train_nll": hist_train, "val_nll": hist_val},
    )


def run_snapshot_study(
    states: np.ndarray,
    times: np.ndarray,
    n_sites: int,
    magic_m2: np.ndarray,
    cfg: NNQSConfig,
    device: str = "cpu",
    model_type: str = "gru",
) -> tuple[pd.DataFrame, dict[int, dict[str, list[float]]]]:
    n_times = states.shape[0]
    snapshot_idx = np.unique(np.linspace(0, n_times - 1, cfg.snapshot_count, dtype=int))

    rows: list[dict[str, float]] = []
    histories: dict[int, dict[str, list[float]]] = {}

    for rank, idx in enumerate(snapshot_idx):
        result = train_nnqs_on_state(
            psi=states[idx],
            n_sites=n_sites,
            cfg=cfg,
            seed=cfg.seed + rank,
            device=device,
            model_type=model_type,
        )
        rows.append(
            {
                "snapshot_index": int(idx),
                "time": float(times[idx]),
                "magic_m2": float(magic_m2[idx]),
                "final_train_nll": result.final_train_nll,
                "final_val_nll": result.final_val_nll,
                "final_kl": result.final_kl,
                "final_tv": result.final_tv,
                "steps_to_threshold": float(result.steps_to_threshold),
            }
        )
        histories[int(idx)] = result.history

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df, histories
