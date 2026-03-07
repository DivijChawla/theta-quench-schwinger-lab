# Novelty Claim Assessment

This report converts robustness statistics into claim verdicts with explicit thresholds.

## NNQS Learnability Claims
- `gru | magic -> validation NLL`: **unsupported** | pooled r=-0.283 [95% CI -0.571, 0.066], studies=6, I2=0.00, positive frac=0.17, significant frac=0.00, combined p=0.529, FDR q=0.722
- `gru | entropy -> validation NLL`: **unsupported** | pooled r=-0.574 [95% CI -0.766, -0.287], studies=6, I2=0.00, positive frac=0.00, significant frac=0.17, combined p=0.0248, FDR q=0.0645
- `gru | magic -> validation NLL | entropy`: **supported** | pooled r=0.579 [95% CI 0.294, 0.769], studies=6, I2=0.15, positive frac=0.83, significant frac=0.17, combined p=0.0222, FDR q=0.0645
- `gru | magic -> KL`: **unsupported** | pooled r=-0.521 [95% CI -0.733, -0.216], studies=6, I2=0.58, positive frac=0.17, significant frac=0.33, combined p=0.0258, FDR q=0.0645
- `gru | magic -> TV`: **unsupported** | pooled r=-0.542 [95% CI -0.747, -0.244], studies=6, I2=0.60, positive frac=0.17, significant frac=0.33, combined p=0.00689, FDR q=0.0645
- `independent | magic -> validation NLL`: **unsupported** | pooled r=-0.283 [95% CI -0.571, 0.067], studies=6, I2=0.17, positive frac=0.50, significant frac=0.17, combined p=0.363, FDR q=0.553
- `independent | entropy -> validation NLL`: **unsupported** | pooled r=-0.555 [95% CI -0.755, -0.262], studies=6, I2=0.34, positive frac=0.17, significant frac=0.33, combined p=0.0195, FDR q=0.0645
- `independent | magic -> validation NLL | entropy`: **supported** | pooled r=0.505 [95% CI 0.195, 0.723], studies=6, I2=0.31, positive frac=0.83, significant frac=0.17, combined p=0.0487, FDR q=0.0912
- `independent | magic -> KL`: **unsupported** | pooled r=-0.070 [95% CI -0.404, 0.280], studies=6, I2=0.00, positive frac=0.50, significant frac=0.00, combined p=0.908, FDR q=0.951
- `independent | magic -> TV`: **unsupported** | pooled r=0.083 [95% CI -0.268, 0.414], studies=6, I2=0.00, positive frac=0.50, significant frac=0.00, combined p=0.944, FDR q=0.951
- `rbm | magic -> validation NLL`: **unsupported** | pooled r=-0.291 [95% CI -0.577, 0.058], studies=6, I2=0.18, positive frac=0.50, significant frac=0.17, combined p=0.369, FDR q=0.553
- `rbm | entropy -> validation NLL`: **unsupported** | pooled r=-0.559 [95% CI -0.757, -0.267], studies=6, I2=0.34, positive frac=0.17, significant frac=0.33, combined p=0.0191, FDR q=0.0645
- `rbm | magic -> validation NLL | entropy`: **supported** | pooled r=0.498 [95% CI 0.186, 0.718], studies=6, I2=0.33, positive frac=0.83, significant frac=0.17, combined p=0.046, FDR q=0.0912
- `rbm | magic -> KL`: **unsupported** | pooled r=-0.092 [95% CI -0.422, 0.259], studies=6, I2=0.00, positive frac=0.50, significant frac=0.00, combined p=0.925, FDR q=0.951
- `rbm | magic -> TV`: **unsupported** | pooled r=0.063 [95% CI -0.286, 0.398], studies=6, I2=0.00, positive frac=0.50, significant frac=0.00, combined p=0.951, FDR q=0.951

## Quench Dynamics Claims
- `N=10`: Spearman(theta,max lambda)=1.000 (p=0), Spearman(theta,max M2)=-0.500 (p=0.667), peak-slope coupling r=0.978 (p=0.135)

## Safe Wording
- Use: 'robust small-N evidence' when verdict is `supported`.
- Use: 'suggestive but mixed' when verdict is `mixed`.
- Avoid universal claims when verdict is `unsupported`.