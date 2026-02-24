from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqm.config import load_config
from tqm.nnqs.train import run_snapshot_study
from tqm.pipeline import run_single_quench


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed robustness sweep for NNQS-vs-magic trend")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seeds", type=str, default="11,17,23,29,31")
    parser.add_argument("--theta1", type=float, default=None)
    parser.add_argument("--out", type=str, default="outputs/nnqs/seed_sweep_metrics.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    theta1 = float(args.theta1) if args.theta1 is not None else float(max(cfg.model.theta1_values))

    run = run_single_quench(cfg, theta1=theta1)
    if 2.0 not in run.magic:
        raise RuntimeError("Need M2 in cfg.magic.alphas for this sweep")
    m2 = run.magic[2.0]

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows = []
    for seed in seeds:
        c = copy.deepcopy(cfg)
        c.nnqs.seed = seed
        df, _ = run_snapshot_study(
            states=run.state_trajectory,
            times=run.times,
            n_sites=c.model.n_sites,
            magic_m2=m2,
            cfg=c.nnqs,
        )

        pear_nll = pearsonr(df["magic_m2"], df["final_val_nll"])
        spear_nll = spearmanr(df["magic_m2"], df["final_val_nll"])
        pear_kl = pearsonr(df["magic_m2"], df["final_kl"])

        rows.append(
            {
                "seed": seed,
                "pearson_magic_vs_val_nll": float(pear_nll.statistic),
                "pearson_p_val_nll": float(pear_nll.pvalue),
                "spearman_magic_vs_val_nll": float(spear_nll.statistic),
                "spearman_p_val_nll": float(spear_nll.pvalue),
                "pearson_magic_vs_kl": float(pear_kl.statistic),
                "pearson_p_kl": float(pear_kl.pvalue),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
