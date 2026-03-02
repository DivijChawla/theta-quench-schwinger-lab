from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_METRIC = "pearson_magic_vs_val_nll_partial_entropy"
SECONDARY_METRICS = [
    "pearson_magic_vs_val_nll",
    "pearson_magic_vs_kl",
    "pearson_magic_vs_tv",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_summary_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec.strip())
        return path.parent.name, path
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def _extract_effect_rows(run_label: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, value in payload.get("meta", {}).items():
        if ":" in key:
            arch, metric = key.split(":", 1)
        else:
            arch, metric = "unknown", key
        rows.append(
            {
                "run_label": run_label,
                "architecture": arch,
                "metric": metric,
                "pooled_r": value.get("pooled_r"),
                "ci_low": value.get("ci_low"),
                "ci_high": value.get("ci_high"),
                "i2": value.get("i2"),
                "combined_p": value.get("combined_p"),
                "combined_q": value.get("combined_q"),
                "verdict": value.get("verdict"),
                "n_studies": value.get("n_studies"),
            }
        )
    return rows


def _count_supported(rows: list[dict[str, Any]], metric: str) -> tuple[int, int]:
    use = [r for r in rows if r["metric"] == metric]
    if not use:
        return 0, 0
    supported = sum(1 for r in use if r.get("verdict") == "supported")
    return supported, len(use)


def _is_headline_label(label: str) -> bool:
    l = label.lower()
    if "alt" in l or "n10" in l or "n12" in l or "stress" in l:
        return False
    return ("baseline" in l) or ("seed" in l) or ("headline" in l)


def _is_boundary_label(label: str) -> bool:
    l = label.lower()
    return ("alt" in l) or ("n10" in l) or ("n12" in l) or ("stress" in l)


def _gate_flags(rows: list[dict[str, Any]], primary_endpoint: str) -> dict[str, bool]:
    primary_supported, primary_total = _count_supported(rows, primary_endpoint)
    entropy_control_supported, entropy_control_total = _count_supported(rows, PRIMARY_METRIC)
    headline_rows = [r for r in rows if r["metric"] == primary_endpoint and _is_headline_label(str(r["run_label"]))]
    headline_supported = sum(1 for r in headline_rows if r.get("verdict") == "supported")
    headline_min = max(1, int(np.ceil(0.75 * len(headline_rows)))) if headline_rows else 1
    boundary_rows = [r for r in rows if r["metric"] == primary_endpoint and _is_boundary_label(str(r["run_label"]))]
    boundary_documented = any(r.get("verdict") in {"mixed", "unsupported"} for r in boundary_rows)
    return {
        "primary_supported_majority": primary_total > 0 and primary_supported >= (primary_total // 2 + 1),
        "entropy_control_supported_any": entropy_control_total > 0 and entropy_control_supported >= 1,
        "has_secondary_metrics": any(r["metric"] in SECONDARY_METRICS for r in rows),
        "primary_supported_headline_region": len(headline_rows) > 0 and headline_supported >= headline_min,
        "boundary_documented": boundary_documented,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standardized effect registry for study runs")
    parser.add_argument("--study-id", type=str, required=True)
    parser.add_argument("--phase", type=str, required=True, choices=["stage1", "stage2"])
    parser.add_argument("--primary-endpoint", type=str, default=PRIMARY_METRIC)
    parser.add_argument("--power-target", type=str, default="r=0.25-0.35")
    parser.add_argument("--summary", action="append", default=[], help="label=path/to/novelty_summary.json")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    if not args.summary:
        raise ValueError("At least one --summary is required")

    effects: list[dict[str, Any]] = []
    run_index: list[dict[str, Any]] = []
    for spec in args.summary:
        label, path = _parse_summary_spec(spec)
        payload = _load_json(path)
        run_index.append(
            {
                "label": label,
                "summary_path": str(path),
                "n_sites": payload.get("n_sites", []),
                "theta1_values": payload.get("theta1_values", []),
                "architectures": payload.get("architectures", []),
                "seeds": payload.get("seeds", []),
                "nnqs_conditions": payload.get("nnqs_conditions"),
                "nnqs_snapshots": payload.get("nnqs_snapshots"),
                "runtime_seconds": payload.get("runtime_seconds"),
            }
        )
        effects.extend(_extract_effect_rows(run_label=label, payload=payload))

    gates = _gate_flags(effects, primary_endpoint=args.primary_endpoint)
    registry = {
        "schema_version": "1.0",
        "study_id": args.study_id,
        "phase": args.phase,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "primary_endpoint": args.primary_endpoint,
        "power_target": args.power_target,
        "runs": run_index,
        "effects": effects,
        "gate_flags": gates,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
