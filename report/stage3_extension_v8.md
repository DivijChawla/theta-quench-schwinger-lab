# Stage 3 Universal-Law Extension

Cross-model extension adds XXZ quenches on top of Stage 2 (Schwinger + TFIM).

- all conditions: `708`
- corridor conditions: `368`
- corridor snapshots: `2428`
- lambda quantile: `0.4`

## Null-disproof gates
- beta CI gate (`beta_ci_low > 0` in all cells): `True`
- permutation FDR gate (`q < 0.05` in all cells): `True`
- Pinsker gate: `True`
- **extended universal beta-law**: `True`

## Sign-law gate
- pooled partial-correlation sign gate (`ci_low_partial > 0` all cells): `True`
- **extended universal sign-law**: `True`

## Heterogeneity
- beta-law I2: `0.9287`
- partial-correlation I2: `0.9831`

## Artifacts
- `outputs/stage3_prx/universal_extension_v8/stage3_cell_metrics.csv`
- `outputs/stage3_prx/universal_extension_v8/stage3_summary.json`
- `outputs/stage3_prx/universal_extension_v8/figs/fig_stage3_beta_forest.png`
- `outputs/stage3_prx/universal_extension_v8/figs/fig_stage3_perm_qvalues.png`
- `outputs/stage3_prx/universal_extension_v8/figs/fig_stage3_partial_sign_heatmap.png`
- `/Users/divijchawla/Documents/Codex/isolated_labs/theta_quench_magic_lab/outputs/stage3_prx/stage3_v8/effect_registry.json`