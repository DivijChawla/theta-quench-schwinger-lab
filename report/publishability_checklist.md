# Publishability Checklist (Schwinger theta-quench + magic + NNQS)

## Completed
- Deterministic configs and reproducible CLI pipeline.
- Independent validation checks (`experiments/manual_verification.py`) pass.
- Cross-architecture baseline evidence at small N (`gru`, `rbm`, `independent`) generated.
- Full-theta seed robustness upgrade completed (`outputs/novelty_gru_seeds8_n6_fulltheta`).
- N=10 exact-magic stress + higher-power rerun completed (`outputs/novelty_n10_exact*`).
- Multi-regime boundary replication completed (three alternate regimes).
- Consolidated claim reports generated:
  - `report/publishability_status.md`
  - `report/regime_sensitivity.md`
  - `report/high_impact_upgrades.md`

## Claim-Gating Outcome
- Supported now:
  - robust small-N baseline correlation between magic and NNQS validation NLL.
  - persistence after entropy control in baseline-like settings.
- Not supported now:
  - universal cross-regime or N=10-size generalized hardness law.

## Paper Framing Required For Honesty
- Present as: **scoped discovery of a conditional relationship** (regime- and size-dependent).
- Do not present as: universal theorem-like law for all Schwinger quenches/NNQS settings.

## Next Upgrades for a stronger paper tier
1. Expand N=10/N=12 time window and sampling budget to test whether negative N=10 correlation is physical or finite-window artifact.
2. Add uncertainty bands directly on headline figures in `report/main.tex`.
3. Add one out-of-family model class (e.g., lightweight transformer autoregressive) for architecture diversity.
