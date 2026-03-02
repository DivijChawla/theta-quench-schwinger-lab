from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqm.study import load_study_config


def _run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _summary_path(path_or_name: str) -> Path:
    p = Path(path_or_name)
    if p.suffix == ".json":
        return ROOT / p
    return ROOT / path_or_name / "novelty_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen camera-ready artifact")
    parser.add_argument("--study", type=str, default="stage1_prxq")
    parser.add_argument("--phase", type=str, default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--artifact-root", type=str, default="artifacts/camera_ready")
    args = parser.parse_args()

    study = load_study_config(args.study, root=ROOT)
    if args.phase not in study.phases:
        raise ValueError(f"Phase '{args.phase}' not found for study '{args.study}'")
    phase = study.phases[args.phase]

    summary_by_name = {
        run.name: _summary_path(run.out_dir)
        for run in phase.runs
    }
    summary_specs = [f"{name}={path}" for name, path in summary_by_name.items()]

    baseline = summary_by_name.get("baseline_n6n8_multiarch")
    n10 = summary_by_name.get("n10_power_exact")
    alt1 = summary_by_name.get("alt_regime_baseline")
    alt2 = summary_by_name.get("alt_regime_heavy_mass")
    alt3 = summary_by_name.get("alt_regime_strong_coupling")

    if baseline is None:
        raise ValueError("Study is missing required baseline run: baseline_n6n8_multiarch")
    if n10 is None:
        raise ValueError("Study is missing required N=10 run: n10_power_exact")

    publish_cmd = [
        sys.executable,
        "experiments/build_publishability_final.py",
        "--baseline",
        f"gru={baseline}",
        "--baseline",
        f"made={baseline}",
        "--baseline",
        f"rbm={baseline}",
        "--baseline",
        f"independent={baseline}",
        "--seed",
        str(baseline),
        "--n10",
        str(n10),
        "--out",
        "report/publishability_status.md",
    ]
    if alt1 is not None:
        publish_cmd.extend(["--regime", f"alt_baseline={alt1}"])
    if alt2 is not None:
        publish_cmd.extend(["--regime", f"alt_heavy_mass={alt2}"])
    if alt3 is not None:
        publish_cmd.extend(["--regime", f"alt_strong_coupling={alt3}"])
    _run(publish_cmd)

    regime_cmd = [
        sys.executable,
        "experiments/build_regime_sensitivity.py",
        "--baseline",
        f"gru={baseline}",
        "--baseline",
        f"made={baseline}",
        "--baseline",
        f"rbm={baseline}",
        "--baseline",
        f"independent={baseline}",
        "--out",
        "report/regime_sensitivity.md",
    ]
    if alt1 is not None:
        regime_cmd.extend(["--alt", f"alt_baseline={alt1}"])
    if alt2 is not None:
        regime_cmd.extend(["--alt", f"alt_heavy_mass={alt2}"])
    if alt3 is not None:
        regime_cmd.extend(["--alt", f"alt_strong_coupling={alt3}"])
    _run(regime_cmd)

    _run(
        [
            sys.executable,
            "experiments/build_effect_registry.py",
            "--study-id",
            study.study_id,
            "--phase",
            args.phase,
            "--primary-endpoint",
            study.primary_endpoint,
            "--power-target",
            study.power_target,
            "--out",
            f"outputs/{study.study_id}/{args.phase}/effect_registry.json",
            *sum([["--summary", spec] for spec in summary_specs], []),
        ]
    )
    _run(
        [
            sys.executable,
            "experiments/claim_gate.py",
            "--publishability",
            "report/publishability_status.md",
            "--targets",
            "README.md,report/main.tex",
        ]
    )
    boundary_cmd = [
        sys.executable,
        "experiments/build_regime_boundary_plot.py",
        "--out",
        "outputs/figs/fig6_regime_boundary_magic_valnll.png",
        "--baseline",
        str(baseline),
        "--n10",
        str(n10),
    ]
    if alt1 is not None:
        boundary_cmd.extend(["--alt-baseline", str(alt1)])
    if alt2 is not None:
        boundary_cmd.extend(["--alt-heavy-mass", str(alt2)])
    if alt3 is not None:
        boundary_cmd.extend(["--alt-strong-coupling", str(alt3)])
    _run(boundary_cmd)
    _run(
        [
            sys.executable,
            "experiments/build_compute_budget_table.py",
            "--glob",
            f"outputs/{study.study_id}/{args.phase}/compute_budget.csv",
            "--out",
            "report/compute_budget.md",
        ]
    )
    _run([sys.executable, "experiments/build_showcase.py", "--outputs", "outputs", "--docs", "docs"])
    _run([sys.executable, "experiments/build_report_pdf.py", "--outputs", "outputs", "--out", "report/short_report.pdf"])

    stage_registry = ROOT / "outputs" / args.study / args.phase / "effect_registry.json"
    if stage_registry.exists():
        _run(
            [
                sys.executable,
                "experiments/check_stage_gate.py",
                "--registry",
                str(stage_registry),
                "--require-primary-headline",
                "--require-entropy-control",
            ]
        )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    freeze_dir = ROOT / args.artifact_root / f"{args.study}_{args.phase}_{ts}"
    freeze_dir.mkdir(parents=True, exist_ok=True)

    # Core reproducibility bundle.
    _copy_tree(ROOT / "configs", freeze_dir / "configs")
    _copy_tree(ROOT / "report", freeze_dir / "report")
    _copy_tree(ROOT / "docs", freeze_dir / "docs")
    _copy_tree(ROOT / "cached_runs", freeze_dir / "cached_runs")

    # Study outputs and effect registry.
    _copy_tree(ROOT / "outputs" / args.study / args.phase, freeze_dir / "outputs" / args.study / args.phase)
    _copy_file(ROOT / "report" / "publishability_status.md", freeze_dir / "report" / "publishability_status.md")
    _copy_file(ROOT / "report" / "regime_sensitivity.md", freeze_dir / "report" / "regime_sensitivity.md")
    _copy_file(ROOT / "report" / "short_report.pdf", freeze_dir / "report" / "short_report.pdf")
    _copy_file(ROOT / "README.md", freeze_dir / "README.md")

    manifest = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "study": args.study,
        "phase": args.phase,
        "frozen_dir": str(freeze_dir),
        "required_outputs": [
            f"outputs/{args.study}/{args.phase}/effect_registry.json",
            "report/publishability_status.md",
            "report/regime_sensitivity.md",
            "report/short_report.pdf",
        ],
    }
    with (freeze_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    archive_base = freeze_dir.with_suffix("")
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=freeze_dir)
    print(f"Wrote frozen bundle: {freeze_dir}")
    print(f"Wrote archive: {archive_path}")


if __name__ == "__main__":
    main()
