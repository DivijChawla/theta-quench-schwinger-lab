# Regime Sensitivity Report

Compares baseline regime vs multiple alternate `(m,g)` regimes for core claims.

| Regime | Architecture | Metric | Baseline Verdict | Baseline r | Alt Verdict | Alt r |
|---|---|---|---|---:|---|---:|
| alt_baseline | gru | magic vs val nll | supported | 0.893 | unsupported | -0.019 |
| alt_baseline | gru | magic vs val nll partial entropy | supported | 0.923 | unsupported | -0.117 |
| alt_baseline | gru | entropy vs val nll | supported | 0.441 | mixed | 0.249 |
| alt_baseline | made | magic vs val nll | supported | 0.754 | unsupported | -0.070 |
| alt_baseline | made | magic vs val nll partial entropy | supported | 0.775 | unsupported | -0.223 |
| alt_baseline | made | entropy vs val nll | mixed | 0.318 | unsupported | 0.221 |
| alt_baseline | rbm | magic vs val nll | supported | 0.890 | unsupported | -0.095 |
| alt_baseline | rbm | magic vs val nll partial entropy | supported | 0.921 | unsupported | -0.138 |
| alt_baseline | rbm | entropy vs val nll | supported | 0.450 | unsupported | 0.154 |
| alt_baseline | independent | magic vs val nll | supported | 0.891 | unsupported | -0.086 |
| alt_baseline | independent | magic vs val nll partial entropy | supported | 0.921 | unsupported | -0.133 |
| alt_baseline | independent | entropy vs val nll | supported | 0.451 | unsupported | 0.161 |
| alt_heavy_mass | gru | magic vs val nll | supported | 0.893 | mixed | 0.341 |
| alt_heavy_mass | gru | magic vs val nll partial entropy | supported | 0.923 | mixed | 0.230 |
| alt_heavy_mass | gru | entropy vs val nll | supported | 0.441 | mixed | 0.266 |
| alt_heavy_mass | made | magic vs val nll | supported | 0.754 | unsupported | 0.012 |
| alt_heavy_mass | made | magic vs val nll partial entropy | supported | 0.775 | unsupported | 0.059 |
| alt_heavy_mass | made | entropy vs val nll | mixed | 0.318 | unsupported | -0.017 |
| alt_heavy_mass | rbm | magic vs val nll | supported | 0.890 | mixed | 0.325 |
| alt_heavy_mass | rbm | magic vs val nll partial entropy | supported | 0.921 | unsupported | 0.140 |
| alt_heavy_mass | rbm | entropy vs val nll | supported | 0.450 | mixed | 0.288 |
| alt_heavy_mass | independent | magic vs val nll | supported | 0.891 | mixed | 0.333 |
| alt_heavy_mass | independent | magic vs val nll partial entropy | supported | 0.921 | unsupported | 0.147 |
| alt_heavy_mass | independent | entropy vs val nll | supported | 0.451 | mixed | 0.296 |
| alt_strong_coupling | gru | magic vs val nll | supported | 0.893 | unsupported | 0.020 |
| alt_strong_coupling | gru | magic vs val nll partial entropy | supported | 0.923 | unsupported | -0.631 |
| alt_strong_coupling | gru | entropy vs val nll | supported | 0.441 | mixed | 0.506 |
| alt_strong_coupling | made | magic vs val nll | supported | 0.754 | unsupported | -0.099 |
| alt_strong_coupling | made | magic vs val nll partial entropy | supported | 0.775 | unsupported | -0.551 |
| alt_strong_coupling | made | entropy vs val nll | mixed | 0.318 | mixed | 0.449 |
| alt_strong_coupling | rbm | magic vs val nll | supported | 0.890 | unsupported | -0.166 |
| alt_strong_coupling | rbm | magic vs val nll partial entropy | supported | 0.921 | unsupported | -0.662 |
| alt_strong_coupling | rbm | entropy vs val nll | supported | 0.450 | mixed | 0.370 |
| alt_strong_coupling | independent | magic vs val nll | supported | 0.891 | unsupported | -0.161 |
| alt_strong_coupling | independent | magic vs val nll partial entropy | supported | 0.921 | unsupported | -0.667 |
| alt_strong_coupling | independent | entropy vs val nll | supported | 0.451 | mixed | 0.373 |

## Interpretation
- The `magic -> validation NLL` claim is regime-sensitive:
- alt_baseline | gru: `supported` -> `unsupported`
- alt_baseline | made: `supported` -> `unsupported`
- alt_baseline | rbm: `supported` -> `unsupported`
- alt_baseline | independent: `supported` -> `unsupported`
- alt_heavy_mass | gru: `supported` -> `mixed`
- alt_heavy_mass | made: `supported` -> `unsupported`
- alt_heavy_mass | rbm: `supported` -> `mixed`
- alt_heavy_mass | independent: `supported` -> `mixed`
- alt_strong_coupling | gru: `supported` -> `unsupported`
- alt_strong_coupling | made: `supported` -> `unsupported`
- alt_strong_coupling | rbm: `supported` -> `unsupported`
- alt_strong_coupling | independent: `supported` -> `unsupported`

## Flip Summary (magic -> val nll)
| Architecture | flips / regimes |
|---|---:|
| gru | 3 / 3 |
| made | 3 / 3 |
| rbm | 3 / 3 |
| independent | 3 / 3 |
- Practical claim boundary: treat magic-learnability relation as conditional on parameter regime, not universal.