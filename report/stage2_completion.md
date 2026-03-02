# Stage 2 Completion Status

## Universal-Law Gates
- all_arch_beta_claim: `True`
- all_arch_beta_quantile_robust: `True`
- all_arch_mechanism_beta_positive: `True`
- all_arch_universal_corridor_claim: `False`
- all_arch_quantile_robust: `False`
- expressive_beta_claim: `True`
- expressive_beta_quantile_robust: `True`
- expressive_universal_corridor_claim: `True`
- expressive_quantile_robust: `True`

## Final Stage 2 Claim
- all-architecture universal beta-law: `supported`
- all-architecture universal corridor sign-law: `unsupported`
- expressive-model universal beta-law (GRU/MADE): `supported`
- expressive-model universal corridor sign-law (GRU/MADE): `supported`

## Dataset Scale
- total conditions: `1188`
- corridor conditions: `592`
- corridor snapshots: `3992`

## Robustness Sweeps
- all architectures quantile sweep pass (sign+beta): `False`
- all architectures quantile sweep pass (beta-only): `True`
- expressive-only quantile sweep pass (sign+beta): `True`
- expressive-only quantile sweep pass (beta-only): `True`

## Key Artifacts
- `outputs/stage2_prlx/universal_scan_v2/cross_model_pooled_effects.csv`
- `outputs/stage2_prlx/universal_scan_v2/cross_model_beta_bounds.csv`
- `outputs/stage2_prlx/universal_scan_v2/finite_size_extrapolation.csv`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_crossmodel_pooled_r.png`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_beta_bounds.png`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_finite_size.png`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_pinsker_bound.png`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_corridor_robustness_all_arch.png`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_corridor_robustness_expressive.png`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_mechanism_arch_slope.png`