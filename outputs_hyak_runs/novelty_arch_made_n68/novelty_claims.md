# Novelty Claim Assessment

This report converts robustness statistics into claim verdicts with explicit thresholds.

## NNQS Learnability Claims
- `made | magic -> validation NLL`: **supported** | pooled r=0.755 [95% CI 0.667, 0.822], studies=40, I2=0.00, positive frac=0.97, significant frac=0.33, combined p=1.32e-12, FDR q=6.61e-12
- `made | entropy -> validation NLL`: **mixed** | pooled r=0.331 [95% CI 0.163, 0.480], studies=40, I2=0.00, positive frac=0.65, significant frac=0.07, combined p=0.0753, FDR q=0.0941
- `made | magic -> validation NLL | entropy`: **supported** | pooled r=0.745 [95% CI 0.654, 0.814], studies=40, I2=0.00, positive frac=0.97, significant frac=0.12, combined p=1.24e-06, FDR q=3.1e-06
- `made | magic -> KL`: **mixed** | pooled r=0.353 [95% CI 0.188, 0.499], studies=40, I2=0.00, positive frac=0.88, significant frac=0.00, combined p=0.821, FDR q=0.821
- `made | magic -> TV`: **mixed** | pooled r=0.469 [95% CI 0.318, 0.597], studies=40, I2=0.00, positive frac=0.88, significant frac=0.03, combined p=0.0681, FDR q=0.0941

## Quench Dynamics Claims
- `N=6`: Spearman(theta,max lambda)=1.000 (p=0), Spearman(theta,max M2)=1.000 (p=0), peak-slope coupling r=0.993 (p=7.19e-05)
- `N=8`: Spearman(theta,max lambda)=1.000 (p=0), Spearman(theta,max M2)=1.000 (p=0), peak-slope coupling r=0.986 (p=0.000284)

## Safe Wording
- Use: 'robust small-N evidence' when verdict is `supported`.
- Use: 'suggestive but mixed' when verdict is `mixed`.
- Avoid universal claims when verdict is `unsupported`.