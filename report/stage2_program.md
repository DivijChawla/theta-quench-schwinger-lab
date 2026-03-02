# Stage 2 Program (PRL/PRX Universal-Law Track)

Stage 2 is implemented and executed for a cross-model corridor study (Schwinger + TFIM), with explicit all-architecture and expressive-only gates, split into beta-law and correlation-sign criteria.

## Goal

Test whether the magic-learnability relation survives:
- finite-size scaling beyond small-\(N\),
- broad regime sweeps in \((m,g,\theta,t)\),
- architecture/model-class changes.

## Entry point

```bash
python3 -m tqm.run --study stage2_prlx --phase stage2
```

Cross-model/bounds entrypoint:

```bash
python3 experiments/stage2_universal_law.py --config configs/stage2_universal.yaml --out-dir outputs/stage2_prlx/universal_scan_v2
python3 experiments/stage2_corridor_robustness.py --in-dir outputs/stage2_prlx/universal_scan_v2 --quantiles 0.2,0.3,0.4,0.5,0.6
python3 experiments/stage2_corridor_robustness.py --in-dir outputs/stage2_prlx/universal_scan_v2 --architectures gru,made --quantiles 0.2,0.3,0.4,0.5,0.6
```

## Required outcomes for universal-claim transition

1. Positive primary endpoint support in majority of stage-2 conditions.
2. Stable sign under finite-size extrapolation diagnostics.
3. Cross-regime robustness or explicit theorem explaining boundary.
4. External independent rerun of frozen artifact.

Current status (see `report/stage2_completion.md`):
- all-architecture universal beta-law: supported
- all-architecture corridor sign-law: unsupported
- expressive-model corridor sign-law (GRU/MADE): supported in tested range

## Artifacts

- `outputs/stage2_prlx/stage2/effect_registry.json`
- `outputs/stage2_prlx/stage2/compute_budget.csv`
- `outputs/stage2_prlx/universal_scan_v2/universal_law_summary.json`
- `outputs/stage2_prlx/universal_scan_v2/cross_model_pooled_effects.csv`
- `outputs/stage2_prlx/universal_scan_v2/cross_model_beta_bounds.csv`
- `outputs/stage2_prlx/universal_scan_v2/finite_size_extrapolation.csv`
- `outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_all_arch.json`
- `outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_expressive.json`
- `outputs/stage2_prlx/universal_scan_v2/stage2_mechanism_summary.json`
- `outputs/stage2_prlx/stage2/stage2_audit.json`
