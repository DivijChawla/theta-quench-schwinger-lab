# High-Impact Upgrades: Final Completion

## Upgrade 1: Full-theta Seed Robustness (non-smoke)
Run:
- `outputs/novelty_gru_seeds8_n6_fulltheta/novelty_summary.json`
- Settings: `N=6`, `theta1={0.0,0.8,1.2,1.6,2.0,2.4}`, GRU, `8` seeds, `epochs=45`, `snapshot_count=5`, `measurement_samples=4000`.

Key outcome:
- `gru:magic -> val NLL` = **supported** (`r=0.857`, `q=1.79e-12`).
- `gru:magic -> val NLL | entropy` = **supported** (`r=0.913`, `q=2.01e-10`).

Implication:
- Seed-level robustness for the core learnability signal is strong in the baseline-like setting.

## Upgrade 2: Stronger N=10 Exact-Magic Stress
Runs:
- `outputs/novelty_n10_exact/novelty_summary.json`
- `outputs/novelty_n10_exact_power/novelty_summary.json` (higher snapshot/sample/epoch budget)

Key outcomes:
- `magic -> val NLL` remains **unsupported** across GRU/RBM/independent at `N=10` in current short-time exact setup.
- In the higher-power N=10 run, pooled `r` for `magic -> val NLL` is negative for all three architectures (GRU/RBM/independent).

Implication:
- Current evidence does not support a universal size-extrapolated hardness statement.
- This is scientifically useful: it tightens claim scope and avoids overclaiming.

## Upgrade 3: Multi-Regime Boundary Mapping
New regimes:
- `outputs/novelty_alt_regime_heavy_mass/novelty_summary.json`
- `outputs/novelty_alt_regime_strong_coupling/novelty_summary.json`
- plus prior `outputs/novelty_alt_regime/novelty_summary.json`

Key outcomes:
- `magic -> val NLL` verdict varies by regime (supported/mixed/unsupported depending on `(m,g)` and architecture).
- Baseline support does **not** transfer uniformly to alternate regimes.

Implication:
- The defensible claim is now explicit and stronger:
  - **magic-learnability coupling is conditional (regime- and size-dependent), not universal.**

## Infrastructure Upgrade (for reliability)
- Added checkpointing/progress to `experiments/novelty_robustness.py` via `--checkpoint-every`.
- Long runs now emit periodic progress and write partial CSVs (`*.partial.csv`) for interruption safety.
