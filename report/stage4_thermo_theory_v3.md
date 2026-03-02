# Stage 4 Thermodynamic + Theory Checks

This stage attempts to promote the beta-law into a stronger physics-law statement by adding:
- finite-size extrapolation (`N -> infinity`) on beta-law cells
- lower-envelope (quantile) regression checks as an empirical bound proxy

- thermo gate (`beta_inf_ci_low > 0` all cells): `False`
- thermo gate expressive-only (`gru,made`): `False`
- quantile lower-envelope gate (`beta_tau_ci_low > 0` for all taus): `True`
- inherited stage3 beta gate: `True`
- inherited stage3 beta gate expressive-only: `True`
- **combined gate (thermo+theory+stage3)**: `False`
- **combined expressive gate (thermo+theory+stage3)**: `False`

## Files
- `outputs/stage4_prl/thermo_theory_v3/stage4_beta_by_size.csv`
- `outputs/stage4_prl/thermo_theory_v3/stage4_beta_thermo_extrapolation.csv`
- `outputs/stage4_prl/thermo_theory_v3/stage4_quantile_bound.csv`
- `outputs/stage4_prl/thermo_theory_v3/figs/fig_stage4_beta_by_size.png`
- `outputs/stage4_prl/thermo_theory_v3/figs/fig_stage4_beta_inf_forest.png`
- `outputs/stage4_prl/thermo_theory_v3/figs/fig_stage4_quantile_bound.png`
- `outputs/stage4_prl/thermo_theory_v3/stage4_summary.json`