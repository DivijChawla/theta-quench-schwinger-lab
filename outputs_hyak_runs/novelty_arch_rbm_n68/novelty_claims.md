# Novelty Claim Assessment

This report converts robustness statistics into claim verdicts with explicit thresholds.

## NNQS Learnability Claims
- `rbm | magic -> validation NLL`: **supported** | pooled r=0.842 [95% CI 0.781, 0.887], studies=40, I2=0.00, positive frac=1.00, significant frac=0.65, combined p=1.88e-22, FDR q=3.13e-22
- `rbm | entropy -> validation NLL`: **supported** | pooled r=0.461 [95% CI 0.309, 0.590], studies=40, I2=0.24, positive frac=0.78, significant frac=0.17, combined p=0.000474, FDR q=0.000474
- `rbm | magic -> validation NLL | entropy`: **supported** | pooled r=0.896 [95% CI 0.854, 0.926], studies=40, I2=0.33, positive frac=0.97, significant frac=0.50, combined p=1.91e-18, FDR q=2.39e-18
- `rbm | magic -> KL`: **supported** | pooled r=0.960 [95% CI 0.944, 0.972], studies=40, I2=0.00, positive frac=1.00, significant frac=0.93, combined p=3.47e-34, FDR q=1.74e-33
- `rbm | magic -> TV`: **supported** | pooled r=0.890 [95% CI 0.846, 0.922], studies=40, I2=0.00, positive frac=1.00, significant frac=0.85, combined p=7.91e-25, FDR q=1.98e-24

## Quench Dynamics Claims
- `N=6`: Spearman(theta,max lambda)=1.000 (p=0), Spearman(theta,max M2)=1.000 (p=0), peak-slope coupling r=0.993 (p=7.19e-05)
- `N=8`: Spearman(theta,max lambda)=1.000 (p=0), Spearman(theta,max M2)=1.000 (p=0), peak-slope coupling r=0.986 (p=0.000284)

## Safe Wording
- Use: 'robust small-N evidence' when verdict is `supported`.
- Use: 'suggestive but mixed' when verdict is `mixed`.
- Avoid universal claims when verdict is `unsupported`.