from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class ModelConfig:
    n_sites: int = 8
    mass: float = 0.5
    coupling_g: float = 1.0
    lattice_spacing: float = 1.0
    theta0: float = 0.0
    theta1_values: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0])


@dataclass
class EvolutionConfig:
    t_max: float = 8.0
    n_steps: int = 121
    method: str = "krylov"  # krylov|dense
    dense_max_sites: int = 9


@dataclass
class MagicConfig:
    enabled: bool = True
    alphas: list[float] = field(default_factory=lambda: [2.0])
    max_sites_exact: int = 10
    x_batch_size: int = 8
    z_batch_size: int = 128


@dataclass
class NNQSConfig:
    enabled: bool = True
    snapshot_count: int = 10
    measurement_samples: int = 16000
    val_fraction: float = 0.2
    hidden_size: int = 64
    epochs: int = 180
    lr: float = 1e-3
    batch_size: int = 256
    threshold_nll: float = 3.5
    seed: int = 7


@dataclass
class OutputConfig:
    out_dir: str = "outputs"
    fig_dir: str = "outputs/figs"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    magic: MagicConfig = field(default_factory=MagicConfig)
    nnqs: NNQSConfig = field(default_factory=NNQSConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.evolution.t_max, self.evolution.n_steps)


def _overlay_dataclass(default_obj: Any, data: dict[str, Any]) -> Any:
    values = {}
    for field_name in default_obj.__dataclass_fields__:
        values[field_name] = data.get(field_name, getattr(default_obj, field_name))
    return default_obj.__class__(**values)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if path is None:
        return cfg
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if "model" in payload:
        cfg.model = _overlay_dataclass(cfg.model, payload["model"])
    if "evolution" in payload:
        cfg.evolution = _overlay_dataclass(cfg.evolution, payload["evolution"])
    if "magic" in payload:
        cfg.magic = _overlay_dataclass(cfg.magic, payload["magic"])
    if "nnqs" in payload:
        cfg.nnqs = _overlay_dataclass(cfg.nnqs, payload["nnqs"])
    if "output" in payload:
        cfg.output = _overlay_dataclass(cfg.output, payload["output"])
    return cfg
