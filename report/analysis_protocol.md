# Analysis Protocol (Pre-registered)

This document defines the fixed analysis plan for Stage 1 (`PRXQ/Quantum`) claims.

## Claim scope

- Primary claim: `magic -> validation NLL | entropy` in specified Schwinger regime windows.
- Secondary claims: `magic -> validation NLL`, `magic -> KL`, `magic -> TV`.
- Universal-law language is disallowed unless Stage 2 gates pass.

## Fixed endpoints

1. Primary endpoint:
   - `pearson_magic_vs_val_nll_partial_entropy`
2. Secondary endpoints:
   - `pearson_magic_vs_val_nll`
   - `pearson_magic_vs_kl`
   - `pearson_magic_vs_tv`

## Statistical pipeline

- Per-condition:
  - Pearson + Spearman correlations
  - bootstrap CI
  - permutation p-values
  - negative controls:
    - random-label baseline correlation
    - shuffled-time baseline correlation
- Cross-condition/meta:
  - Fisher pooled effect size
  - heterogeneity `I2`
  - Fisher combined p
  - BH-FDR q-values
  - verdict classes: `supported | mixed | unsupported`
- Mixed-effects regression:
  - `final_val_nll ~ magic_m2 + snapshot_entropy + C(architecture) + (1|condition_id)`

## Power target

- Design target: detect correlation effects in `r ≈ 0.25–0.35`.
- Minimum seeds per headline condition: `>= 10`.
- Power summary artifact:
  - `outputs/statistical_rigor/power_analysis.csv`

## Acceptance gates (Stage 1)

1. Cross-architecture conditional claim supported with corrected significance.
2. Primary endpoint robust in declared region.
3. Boundary map stable under seed/model ablations.
4. Independent rerun reproduces headline tables/figures.

## Prohibited interpretation

- No universal hardness claim from Stage 1 evidence.
- No extrapolation beyond evaluated size/regime grid without explicit Stage 2 evidence.
