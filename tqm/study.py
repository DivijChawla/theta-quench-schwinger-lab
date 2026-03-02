from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StudyRunSpec:
    name: str
    config: str
    out_dir: str
    n_sites: list[int] = field(default_factory=list)
    theta1: list[float] = field(default_factory=list)
    architectures: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    hidden_sizes: list[int] = field(default_factory=list)
    bootstrap: int = 200
    permutations: int = 300
    approx_magic_samples: int = 0
    epochs: int | None = None
    measurement_samples: int | None = None
    snapshot_count: int | None = None
    checkpoint_every: int = 10


@dataclass
class StudyAggregateSpec:
    publishability_out: str = "report/publishability_status.md"
    regime_sensitivity_out: str = "report/regime_sensitivity.md"


@dataclass
class StudyPhaseSpec:
    name: str
    runs: list[StudyRunSpec] = field(default_factory=list)
    aggregate: StudyAggregateSpec = field(default_factory=StudyAggregateSpec)


@dataclass
class StudyConfig:
    study_id: str
    primary_endpoint: str
    power_target: str
    models: list[str]
    size_grid: list[int]
    regime_grid: list[dict[str, Any]]
    phases: dict[str, StudyPhaseSpec] = field(default_factory=dict)


def _parse_run_spec(raw: dict[str, Any]) -> StudyRunSpec:
    return StudyRunSpec(
        name=str(raw["name"]),
        config=str(raw["config"]),
        out_dir=str(raw["out_dir"]),
        n_sites=[int(x) for x in raw.get("n_sites", [])],
        theta1=[float(x) for x in raw.get("theta1", [])],
        architectures=[str(x) for x in raw.get("architectures", [])],
        seeds=[int(x) for x in raw.get("seeds", [])],
        hidden_sizes=[int(x) for x in raw.get("hidden_sizes", [])],
        bootstrap=int(raw.get("bootstrap", 200)),
        permutations=int(raw.get("permutations", 300)),
        approx_magic_samples=int(raw.get("approx_magic_samples", 0)),
        epochs=int(raw["epochs"]) if raw.get("epochs") is not None else None,
        measurement_samples=int(raw["measurement_samples"]) if raw.get("measurement_samples") is not None else None,
        snapshot_count=int(raw["snapshot_count"]) if raw.get("snapshot_count") is not None else None,
        checkpoint_every=int(raw.get("checkpoint_every", 10)),
    )


def _parse_phase_spec(name: str, raw: dict[str, Any]) -> StudyPhaseSpec:
    runs = [_parse_run_spec(item) for item in raw.get("runs", [])]
    agg = StudyAggregateSpec(
        publishability_out=str(raw.get("aggregate", {}).get("publishability_out", "report/publishability_status.md")),
        regime_sensitivity_out=str(raw.get("aggregate", {}).get("regime_sensitivity_out", "report/regime_sensitivity.md")),
    )
    return StudyPhaseSpec(name=name, runs=runs, aggregate=agg)


def load_study_config(study_id: str, root: str | Path | None = None) -> StudyConfig:
    base = Path(root) if root is not None else Path.cwd()
    path = base / "configs" / "studies" / f"{study_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Study config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    required = ["study_id", "primary_endpoint", "power_target", "models", "size_grid", "regime_grid"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Missing required study fields in {path}: {missing}")

    phases_raw = payload.get("phases", {})
    phases = {name: _parse_phase_spec(name, raw) for name, raw in phases_raw.items()}

    return StudyConfig(
        study_id=str(payload["study_id"]),
        primary_endpoint=str(payload["primary_endpoint"]),
        power_target=str(payload["power_target"]),
        models=[str(x) for x in payload.get("models", [])],
        size_grid=[int(x) for x in payload.get("size_grid", [])],
        regime_grid=[dict(x) for x in payload.get("regime_grid", [])],
        phases=phases,
    )
