# Stage 2 Universal-Law Scan

This report evaluates a cross-model corridor claim using Schwinger + TFIM data.

- total conditions analyzed: `1188`
- corridor-filtered conditions: `592` (lambda quantile `0.4`)
- corridor snapshots: `3992`

## Corridor Claim Gate
- sign-consistency gate (CI low > 0 for pooled primary r): `False`
- beta-lower-bound gate (95% CI low > 0): `True`
- Pinsker sanity gate (KL >= 2 TV^2): `True`
- **universal beta-law claim**: `True`
- **universal corridor claim**: `False`

## Expressive-Model Gate (GRU + MADE)
- expressive sign gate: `True`
- expressive beta gate: `True`
- universal expressive beta-law claim: `True`
- **universal expressive-model corridor claim**: `True`

Interpretation:
- The beta-law can hold while pooled-correlation sign fails due architecture/model heterogeneity.
- If all-architecture correlation gate fails but expressive gate passes, the strongest sign-stable claim is an expressive-NNQS corridor law.

## Files
- `outputs/stage2_prlx/universal_scan_v2/cross_model_pooled_effects.csv`
- `outputs/stage2_prlx/universal_scan_v2/cross_model_beta_bounds.csv`
- `outputs/stage2_prlx/universal_scan_v2/finite_size_extrapolation.csv`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_crossmodel_pooled_r.png`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_beta_bounds.png`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_finite_size.png`
- `outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_pinsker_bound.png`