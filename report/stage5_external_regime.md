# Stage 5 External Model + Regime Expansion

Adds ANNNI as a fourth family and introduces adversarial regime gates.

- all conditions: `852`
- corridor conditions: `456`
- corridor snapshots: `2356`
- families in corridor: `4`
- regimes in corridor: `10`

## Gates
- beta cell gate: `True`
- permutation FDR gate: `True`
- regime gate: `False`
- minimax regime gate: `False`
- quantile theory gate: `True`
- **combined external+regime+theory gate**: `False`

## Artifacts
- `outputs/stage5_prx/external_regime_v1/stage5_summary.json`
- `outputs/stage5_prx/external_regime_v1/stage5_cell_metrics.csv`
- `outputs/stage5_prx/external_regime_v1/stage5_regime_metrics.csv`
- `outputs/stage5_prx/external_regime_v1/stage5_quantile_bound.csv`
- `outputs/stage5_prx/external_regime_v1/figs/fig_stage5_beta_forest.png`
- `outputs/stage5_prx/external_regime_v1/figs/fig_stage5_regime_minimax.png`
- `outputs/stage5_prx/external_regime_v1/figs/fig_stage5_regime_heatmap.png`
- `outputs/stage5_prx/external_regime_v1/figs/fig_stage5_quantile_bound.png`
- `/Users/divijchawla/Documents/Codex/isolated_labs/theta_quench_magic_lab/outputs/stage5_prx/stage5/effect_registry.json`