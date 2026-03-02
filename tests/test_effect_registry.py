from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_effect_registry_builder_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "effect_registry.json"
    cmd = [
        sys.executable,
        "experiments/build_effect_registry.py",
        "--study-id",
        "stage1_prxq",
        "--phase",
        "stage1",
        "--summary",
        "baseline=outputs/novelty/novelty_summary.json",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=root)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["study_id"] == "stage1_prxq"
    assert payload["phase"] == "stage1"
    assert "effects" in payload and len(payload["effects"]) > 0
    assert "gate_flags" in payload
