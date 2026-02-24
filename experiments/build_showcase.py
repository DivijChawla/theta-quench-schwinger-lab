from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


FIG_NAMES = [
    "fig1_loschmidt_family.png",
    "fig2_magic_lambda_overlay.png",
    "fig3_heatmaps_lambda_magic.png",
    "fig4_nnqs_loss_vs_magic.png",
    "fig4b_nnqs_val_curves.png",
    "fig5_entropy_vs_magic.png",
    "quench_dynamics.gif",
]


def maybe_float(x: float | np.floating) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def copy_figures(outputs_figs: Path, docs_figs: Path) -> list[str]:
    docs_figs.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in FIG_NAMES:
        src = outputs_figs / name
        if src.exists():
            shutil.copy2(src, docs_figs / name)
            copied.append(name)
    return copied


def build_metrics(outputs_dir: Path) -> dict:
    validation_path = outputs_dir / "validation.json"
    quench_path = outputs_dir / "quench_summary.csv"
    nnqs_path = outputs_dir / "nnqs" / "nnqs_snapshot_metrics.csv"

    payload: dict = {
        "has_validation": validation_path.exists(),
        "has_quench_summary": quench_path.exists(),
        "has_nnqs_metrics": nnqs_path.exists(),
    }

    if validation_path.exists():
        with validation_path.open("r", encoding="utf-8") as f:
            payload["validation"] = json.load(f)

    if quench_path.exists():
        q = pd.read_csv(quench_path)
        payload["quench"] = {
            "n_quenches": int(len(q)),
            "theta1_min": maybe_float(np.min(q["theta1"])),
            "theta1_max": maybe_float(np.max(q["theta1"])),
            "max_lambda_min": maybe_float(np.min(q["max_lambda"])),
            "max_lambda_max": maybe_float(np.max(q["max_lambda"])),
            "max_magic_m2_min": maybe_float(np.min(q.get("max_magic_m2", np.nan))),
            "max_magic_m2_max": maybe_float(np.max(q.get("max_magic_m2", np.nan))),
            "corr_theta1_vs_max_lambda": maybe_float(np.corrcoef(q["theta1"], q["max_lambda"])[0, 1])
            if len(q) > 1
            else float("nan"),
            "corr_theta1_vs_max_magic_m2": maybe_float(
                np.corrcoef(q["theta1"], q["max_magic_m2"])[0, 1]
            )
            if len(q) > 1 and "max_magic_m2" in q.columns
            else float("nan"),
        }

    if nnqs_path.exists():
        n = pd.read_csv(nnqs_path)
        if len(n) > 1:
            pear_nll = pearsonr(n["magic_m2"], n["final_val_nll"])
            spear_nll = spearmanr(n["magic_m2"], n["final_val_nll"])
            pear_kl = pearsonr(n["magic_m2"], n["final_kl"])
            payload["nnqs"] = {
                "n_snapshots": int(len(n)),
                "pearson_magic_vs_val_nll": maybe_float(pear_nll.statistic),
                "pearson_p_magic_vs_val_nll": maybe_float(pear_nll.pvalue),
                "spearman_magic_vs_val_nll": maybe_float(spear_nll.statistic),
                "spearman_p_magic_vs_val_nll": maybe_float(spear_nll.pvalue),
                "pearson_magic_vs_kl": maybe_float(pear_kl.statistic),
                "pearson_p_magic_vs_kl": maybe_float(pear_kl.pvalue),
                "magic_min": maybe_float(np.min(n["magic_m2"])),
                "magic_max": maybe_float(np.max(n["magic_m2"])),
            }
        else:
            payload["nnqs"] = {"n_snapshots": int(len(n))}

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GitHub Pages showcase assets from outputs")
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--docs", type=str, default="docs")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs)
    docs_dir = Path(args.docs)
    docs_figs = docs_dir / "figs"
    docs_data = docs_dir / "data"
    docs_data.mkdir(parents=True, exist_ok=True)

    copied = copy_figures(outputs_dir / "figs", docs_figs)
    metrics = build_metrics(outputs_dir)
    metrics["copied_figures"] = copied

    out_metrics = docs_data / "metrics.json"
    with out_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Copied {len(copied)} figure(s) to {docs_figs}")
    print(f"Wrote metrics: {out_metrics}")


if __name__ == "__main__":
    main()
