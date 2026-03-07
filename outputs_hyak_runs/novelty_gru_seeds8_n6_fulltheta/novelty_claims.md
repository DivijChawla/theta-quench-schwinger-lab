# Novelty Claim Assessment

This report converts robustness statistics into claim verdicts with explicit thresholds.

## NNQS Learnability Claims
- `gru | magic -> validation NLL`: **supported** | pooled r=0.857 [95% CI 0.787, 0.905], studies=40, I2=0.00, positive frac=1.00, significant frac=0.30, combined p=3.58e-13, FDR q=1.79e-12
- `gru | entropy -> validation NLL`: **mixed** | pooled r=0.274 [95% CI 0.062, 0.463], studies=40, I2=0.00, positive frac=0.82, significant frac=0.00, combined p=1, FDR q=1
- `gru | magic -> validation NLL | entropy`: **supported** | pooled r=0.913 [95% CI 0.868, 0.943], studies=40, I2=0.00, positive frac=1.00, significant frac=0.25, combined p=8.04e-11, FDR q=2.01e-10
- `gru | magic -> KL`: **unsupported** | pooled r=-0.508 [95% CI -0.652, -0.328], studies=40, I2=0.00, positive frac=0.17, significant frac=0.15, combined p=0.0181, FDR q=0.0301
- `gru | magic -> TV`: **unsupported** | pooled r=-0.321 [95% CI -0.502, -0.113], studies=40, I2=0.00, positive frac=0.33, significant frac=0.03, combined p=0.514, FDR q=0.643

## Quench Dynamics Claims
- `N=6`: Spearman(theta,max lambda)=1.000 (p=0), Spearman(theta,max M2)=1.000 (p=0), peak-slope coupling r=0.993 (p=7.19e-05)

## Safe Wording
- Use: 'robust small-N evidence' when verdict is `supported`.
- Use: 'suggestive but mixed' when verdict is `mixed`.
- Avoid universal claims when verdict is `unsupported`.