from __future__ import annotations

from pathlib import Path

from tqm.study import load_study_config


def test_stage1_registry_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_study_config("stage1_prxq", root=root)
    assert cfg.study_id == "stage1_prxq"
    assert cfg.primary_endpoint == "pearson_magic_vs_val_nll_partial_entropy"
    assert "stage1" in cfg.phases
    assert len(cfg.phases["stage1"].runs) >= 1


def test_stage2_registry_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_study_config("stage2_prlx", root=root)
    assert cfg.study_id == "stage2_prlx"
    assert "stage2" in cfg.phases
    assert len(cfg.phases["stage2"].runs) >= 1
