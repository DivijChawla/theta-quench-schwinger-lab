from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqm.study import StudyConfig, StudyRunSpec, load_study_config


def _list_arg(values: list[object]) -> str:
    return ",".join(str(v) for v in values)


def _build_novelty_cmd(run: StudyRunSpec) -> list[str]:
    cmd = [
        sys.executable,
        "experiments/novelty_robustness.py",
        "--config",
        run.config,
        "--out-dir",
        run.out_dir,
        "--bootstrap",
        str(run.bootstrap),
        "--permutations",
        str(run.permutations),
        "--checkpoint-every",
        str(run.checkpoint_every),
    ]
    if run.n_sites:
        cmd.extend(["--n-sites", _list_arg(run.n_sites)])
    if run.theta1:
        cmd.extend(["--theta1", _list_arg(run.theta1)])
    if run.architectures:
        cmd.extend(["--architectures", _list_arg(run.architectures)])
    if run.seeds:
        cmd.extend(["--seeds", _list_arg(run.seeds)])
    if run.hidden_sizes:
        cmd.extend(["--hidden-sizes", _list_arg(run.hidden_sizes)])
    if run.approx_magic_samples > 0:
        cmd.extend(["--approx-magic-samples", str(run.approx_magic_samples)])
    if run.epochs is not None:
        cmd.extend(["--epochs", str(run.epochs)])
    if run.measurement_samples is not None:
        cmd.extend(["--measurement-samples", str(run.measurement_samples)])
    if run.snapshot_count is not None:
        cmd.extend(["--snapshot-count", str(run.snapshot_count)])
    return cmd


def _build_stage3_cmd(run: StudyRunSpec) -> list[str]:
    return [
        sys.executable,
        "experiments/stage3_universal_extension.py",
        "--config",
        run.config,
        "--out-dir",
        run.out_dir,
        "--registry-out",
        "outputs/stage3_prx/stage3/effect_registry.json",
        "--report-out",
        "report/stage3_extension.md",
    ]


def _run_cmd(cmd: list[str], dry_run: bool = False) -> float:
    rendered = " ".join(shlex.quote(x) for x in cmd)
    print(f"[run] {rendered}")
    if dry_run:
        return 0.0
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT)
    return float(time.perf_counter() - t0)


def _write_compute_budget(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "label", "command", "runtime_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _summary_spec(run: StudyRunSpec) -> str:
    return f"{run.name}={Path(run.out_dir) / 'novelty_summary.json'}"


def _pick_run_path(summary_by_name: dict[str, Path], include: list[str], exclude: list[str] | None = None) -> Path | None:
    exc = exclude or []
    for name, path in summary_by_name.items():
        lowered = name.lower()
        if all(token in lowered for token in include) and not any(token in lowered for token in exc):
            if path.exists():
                return path
    return None


def run_phase(study: StudyConfig, phase: str, dry_run: bool = False) -> None:
    if phase not in study.phases:
        raise ValueError(f"Phase '{phase}' not found for study {study.study_id}")
    phase_cfg = study.phases[phase]
    out_root = ROOT / "outputs" / study.study_id / phase
    out_root.mkdir(parents=True, exist_ok=True)

    budget_rows: list[dict[str, object]] = []
    summary_specs: list[str] = []
    summary_by_name: dict[str, Path] = {}

    for idx, run in enumerate(phase_cfg.runs, start=1):
        cmd = _build_stage3_cmd(run) if phase == "stage3" else _build_novelty_cmd(run)
        runtime = _run_cmd(cmd, dry_run=dry_run)
        budget_rows.append(
            {
                "step": f"novelty_run_{idx}",
                "label": run.name,
                "command": " ".join(cmd),
                "runtime_seconds": runtime,
            }
        )
        if phase != "stage3":
            summary_specs.append(_summary_spec(run))
            summary_by_name[run.name] = Path(run.out_dir) / "novelty_summary.json"

    if phase == "stage3":
        _write_compute_budget(out_root / "compute_budget.csv", budget_rows)
        print(f"Wrote {out_root / 'compute_budget.csv'}")
        return

    if not dry_run:
        baseline_path = summary_by_name.get("baseline_n6n8_multiarch")
        if baseline_path is None or not baseline_path.exists():
            baseline_path = _pick_run_path(summary_by_name, include=["baseline"], exclude=["alt"])
        if baseline_path is None:
            baseline_path = _pick_run_path(summary_by_name, include=["baseline"])
        if baseline_path is None and summary_by_name:
            candidate = next(iter(summary_by_name.values()))
            baseline_path = candidate if candidate.exists() else None

        n10_path = summary_by_name.get("n10_power_exact")
        if n10_path is None or not n10_path.exists():
            n10_path = _pick_run_path(summary_by_name, include=["n10"])

        alt_baseline_path = summary_by_name.get("alt_regime_baseline")
        if alt_baseline_path is None or not alt_baseline_path.exists():
            alt_baseline_path = _pick_run_path(summary_by_name, include=["alt", "baseline"])
        alt_heavy_path = summary_by_name.get("alt_regime_heavy_mass")
        if alt_heavy_path is None or not alt_heavy_path.exists():
            alt_heavy_path = _pick_run_path(summary_by_name, include=["alt", "heavy"])
        alt_strong_path = summary_by_name.get("alt_regime_strong_coupling")
        if alt_strong_path is None or not alt_strong_path.exists():
            alt_strong_path = _pick_run_path(summary_by_name, include=["alt", "strong"])

        publish_cmd = [
            sys.executable,
            "experiments/build_publishability_final.py",
            "--out",
            phase_cfg.aggregate.publishability_out,
        ]
        if baseline_path and baseline_path.exists():
            for arch in ["gru", "made", "rbm", "independent"]:
                publish_cmd.extend(["--baseline", f"{arch}={baseline_path}"])
            publish_cmd.extend(["--seed", str(baseline_path)])
        if n10_path and n10_path.exists():
            publish_cmd.extend(["--n10", str(n10_path)])
        if alt_baseline_path and alt_baseline_path.exists():
            publish_cmd.extend(["--regime", f"alt_baseline={alt_baseline_path}"])
        if alt_heavy_path and alt_heavy_path.exists():
            publish_cmd.extend(["--regime", f"alt_heavy_mass={alt_heavy_path}"])
        if alt_strong_path and alt_strong_path.exists():
            publish_cmd.extend(["--regime", f"alt_strong_coupling={alt_strong_path}"])
        runtime = _run_cmd(publish_cmd, dry_run=False)
        budget_rows.append(
            {
                "step": "publishability",
                "label": "publishability_status",
                "command": " ".join(publish_cmd),
                "runtime_seconds": runtime,
            }
        )

        regime_cmd = [
            sys.executable,
            "experiments/build_regime_sensitivity.py",
            "--out",
            phase_cfg.aggregate.regime_sensitivity_out,
        ]
        if baseline_path and baseline_path.exists():
            for arch in ["gru", "made", "rbm", "independent"]:
                regime_cmd.extend(["--baseline", f"{arch}={baseline_path}"])
        if alt_baseline_path and alt_baseline_path.exists():
            regime_cmd.extend(["--alt", f"alt_baseline={alt_baseline_path}"])
        if alt_heavy_path and alt_heavy_path.exists():
            regime_cmd.extend(["--alt", f"alt_heavy_mass={alt_heavy_path}"])
        if alt_strong_path and alt_strong_path.exists():
            regime_cmd.extend(["--alt", f"alt_strong_coupling={alt_strong_path}"])
        runtime = _run_cmd(regime_cmd, dry_run=False)
        budget_rows.append(
            {
                "step": "regime_sensitivity",
                "label": "regime_sensitivity",
                "command": " ".join(regime_cmd),
                "runtime_seconds": runtime,
            }
        )

        effect_cmd = [
            sys.executable,
            "experiments/build_effect_registry.py",
            "--study-id",
            study.study_id,
            "--phase",
            phase,
            "--primary-endpoint",
            study.primary_endpoint,
            "--power-target",
            study.power_target,
            "--out",
            str(out_root / "effect_registry.json"),
        ]
        for spec in summary_specs:
            effect_cmd.extend(["--summary", spec])

        runtime = _run_cmd(effect_cmd, dry_run=False)
        budget_rows.append(
            {
                "step": "effect_registry",
                "label": "effect_registry",
                "command": " ".join(effect_cmd),
                "runtime_seconds": runtime,
            }
        )

    _write_compute_budget(out_root / "compute_budget.csv", budget_rows)
    print(f"Wrote {out_root / 'compute_budget.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run study registry phase")
    parser.add_argument("--study", type=str, required=True, help="Study id from configs/studies/<id>.yaml")
    parser.add_argument("--phase", type=str, required=True, choices=["stage1", "stage2", "stage3"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    study = load_study_config(args.study, root=ROOT)
    run_phase(study, phase=args.phase, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
