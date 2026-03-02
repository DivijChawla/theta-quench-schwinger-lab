# Consolidated Publishability Status (Final)

## Baseline Cross-Architecture Evidence
| Architecture | Metric | Verdict | pooled r | 95% CI | q-value |
|---|---|---|---:|---|---:|
| gru | magic vs val nll | supported | 0.893 | [0.867, 0.913] | 4.14e-62 |
| gru | magic vs val nll partial entropy | supported | 0.923 | [0.905, 0.938] | 3.39e-54 |
| gru | entropy vs val nll | supported | 0.441 | [0.345, 0.527] | 0.002 |
| gru | magic vs kl | unsupported | -0.068 | [-0.179, 0.045] | 0.703 |
| gru | magic vs tv | mixed | 0.176 | [0.064, 0.283] | 0.140 |
| independent | magic vs val nll | supported | 0.891 | [0.865, 0.912] | 1.41e-60 |
| independent | magic vs val nll partial entropy | supported | 0.921 | [0.902, 0.937] | 1.27e-51 |
| independent | entropy vs val nll | supported | 0.451 | [0.356, 0.536] | 0.001 |
| independent | magic vs kl | supported | 0.959 | [0.949, 0.967] | 1.69e-85 |
| independent | magic vs tv | supported | 0.902 | [0.879, 0.921] | 4.28e-64 |
| made | magic vs val nll | supported | 0.754 | [0.700, 0.798] | 4.77e-26 |
| made | magic vs val nll partial entropy | supported | 0.775 | [0.726, 0.816] | 1.53e-19 |
| made | entropy vs val nll | mixed | 0.318 | [0.213, 0.416] | 0.312 |
| made | magic vs kl | unsupported | -0.018 | [-0.130, 0.095] | 0.547 |
| made | magic vs tv | unsupported | 0.025 | [-0.088, 0.138] | 0.117 |
| rbm | magic vs val nll | supported | 0.890 | [0.864, 0.911] | 1.41e-60 |
| rbm | magic vs val nll partial entropy | supported | 0.921 | [0.902, 0.936] | 1.27e-51 |
| rbm | entropy vs val nll | supported | 0.450 | [0.356, 0.536] | 0.001 |
| rbm | magic vs kl | supported | 0.960 | [0.950, 0.968] | 1.69e-85 |
| rbm | magic vs tv | supported | 0.901 | [0.877, 0.920] | 4.24e-64 |

## Robustness Upgrades
### Seed Robustness (N=6,8, full theta grid, 10 seeds, GRU)
- `magic vs val nll`: supported | r=0.893 | q=4.14e-62
- `magic vs val nll partial entropy`: supported | r=0.923 | q=3.39e-54
### N=10 Exact-Magic Stress (power run)
- `gru` magic vs val nll: unsupported | r=0.070 | q=1.000
- `made` magic vs val nll: unsupported | r=0.070 | q=1.000
- `rbm` magic vs val nll: unsupported | r=0.039 | q=1.000
- `independent` magic vs val nll: unsupported | r=0.047 | q=1.000

## Regime Boundary (magic -> val nll)
| Architecture | Regime | Verdict | pooled r | q-value |
|---|---|---|---:|---:|
| gru | alt_baseline | unsupported | -0.019 | 1.000 |
| made | alt_baseline | unsupported | -0.070 | 1.000 |
| rbm | alt_baseline | unsupported | -0.095 | 1.000 |
| independent | alt_baseline | unsupported | -0.086 | 1.000 |
| gru | alt_heavy_mass | mixed | 0.341 | 0.691 |
| made | alt_heavy_mass | unsupported | 0.012 | 1.000 |
| rbm | alt_heavy_mass | mixed | 0.325 | 0.691 |
| independent | alt_heavy_mass | mixed | 0.333 | 0.691 |
| gru | alt_strong_coupling | unsupported | 0.020 | 6.79e-06 |
| made | alt_strong_coupling | unsupported | -0.099 | 5.45e-04 |
| rbm | alt_strong_coupling | unsupported | -0.166 | 1.13e-09 |
| independent | alt_strong_coupling | unsupported | -0.161 | 9.46e-10 |

### Regime Verdict Counts
| Architecture | supported | mixed | unsupported |
|---|---:|---:|---:|
| gru | 0 | 1 | 2 |
| made | 0 | 0 | 3 |
| rbm | 0 | 1 | 2 |
| independent | 0 | 1 | 2 |

## Claim Gate
- **Publishable scoped claim**: strong small-N, architecture-robust evidence for `magic -> validation NLL`, with explicit failure to generalize to current N=10 setting and alternate regimes.
- Correct framing: `magic-learnability coupling is conditional (regime- and size-dependent), not universal`.
- Do not claim universal hardness laws from current data.