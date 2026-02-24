from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def maybe_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy canonical cached outputs for repository distribution")
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--cache", type=str, default="cached_runs")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    files = [
        outputs / "validation.json",
        outputs / "quench_summary.csv",
        outputs / "sweep_grid.npz",
        outputs / "nnqs" / "nnqs_snapshot_metrics.csv",
        outputs / "nnqs" / "nnqs_histories.json",
    ]

    figures = [
        outputs / "figs" / "fig1_loschmidt_family.png",
        outputs / "figs" / "fig2_magic_lambda_overlay.png",
        outputs / "figs" / "fig3_heatmaps_lambda_magic.png",
        outputs / "figs" / "fig4_nnqs_loss_vs_magic.png",
        outputs / "figs" / "fig4b_nnqs_val_curves.png",
        outputs / "figs" / "fig5_entropy_vs_magic.png",
        outputs / "figs" / "quench_dynamics.gif",
    ]

    count = 0
    for src in files:
        rel = src.relative_to(outputs)
        if maybe_copy(src, cache / rel):
            count += 1

    for src in figures:
        rel = src.relative_to(outputs)
        if maybe_copy(src, cache / rel):
            count += 1

    print(f"Copied {count} artifact(s) to {cache}")


if __name__ == "__main__":
    main()
