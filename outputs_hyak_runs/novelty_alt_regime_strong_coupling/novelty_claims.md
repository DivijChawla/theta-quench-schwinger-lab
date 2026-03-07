# Novelty Claim Assessment

This report converts robustness statistics into claim verdicts with explicit thresholds.

## NNQS Learnability Claims
- `gru | magic -> validation NLL`: **unsupported** | pooled r=-0.392 [95% CI -0.788, 0.235], studies=9, I2=0.00, positive frac=0.33, significant frac=0.00, combined p=0.516, FDR q=0.922
- `gru | entropy -> validation NLL`: **unsupported** | pooled r=-0.265 [95% CI -0.728, 0.365], studies=9, I2=0.00, positive frac=0.56, significant frac=0.00, combined p=0.829, FDR q=0.922
- `gru | magic -> validation NLL | entropy`: **unsupported** | pooled r=0.134 [95% CI -0.477, 0.657], studies=9, I2=0.00, positive frac=0.56, significant frac=0.11, combined p=0.796, FDR q=0.922
- `gru | magic -> KL`: **mixed** | pooled r=0.690 [95% CI 0.193, 0.905], studies=9, I2=0.00, positive frac=0.67, significant frac=0.00, combined p=0.3, FDR q=0.922
- `gru | magic -> TV`: **mixed** | pooled r=0.424 [95% CI -0.198, 0.803], studies=9, I2=0.27, positive frac=0.89, significant frac=0.00, combined p=0.195, FDR q=0.922
- `independent | magic -> validation NLL`: **unsupported** | pooled r=-0.473 [95% CI -0.823, 0.139], studies=9, I2=0.00, positive frac=0.22, significant frac=0.00, combined p=0.668, FDR q=0.922
- `independent | entropy -> validation NLL`: **unsupported** | pooled r=-0.418 [95% CI -0.800, 0.205], studies=9, I2=0.00, positive frac=0.44, significant frac=0.00, combined p=0.766, FDR q=0.922
- `independent | magic -> validation NLL | entropy`: **unsupported** | pooled r=0.245 [95% CI -0.383, 0.718], studies=9, I2=0.51, positive frac=0.44, significant frac=0.11, combined p=0.922, FDR q=0.922
- `independent | magic -> KL`: **unsupported** | pooled r=-0.530 [95% CI -0.846, 0.063], studies=9, I2=0.00, positive frac=0.00, significant frac=0.00, combined p=0.912, FDR q=0.922
- `independent | magic -> TV`: **unsupported** | pooled r=-0.594 [95% CI -0.871, -0.030], studies=9, I2=0.00, positive frac=0.11, significant frac=0.00, combined p=0.783, FDR q=0.922
- `rbm | magic -> validation NLL`: **unsupported** | pooled r=-0.482 [95% CI -0.827, 0.127], studies=9, I2=0.00, positive frac=0.22, significant frac=0.00, combined p=0.656, FDR q=0.922
- `rbm | entropy -> validation NLL`: **unsupported** | pooled r=-0.423 [95% CI -0.802, 0.199], studies=9, I2=0.00, positive frac=0.44, significant frac=0.00, combined p=0.759, FDR q=0.922
- `rbm | magic -> validation NLL | entropy`: **unsupported** | pooled r=0.273 [95% CI -0.357, 0.732], studies=9, I2=0.56, positive frac=0.44, significant frac=0.11, combined p=0.922, FDR q=0.922
- `rbm | magic -> KL`: **unsupported** | pooled r=-0.547 [95% CI -0.853, 0.039], studies=9, I2=0.00, positive frac=0.00, significant frac=0.00, combined p=0.887, FDR q=0.922
- `rbm | magic -> TV`: **unsupported** | pooled r=-0.608 [95% CI -0.876, -0.052], studies=9, I2=0.00, positive frac=0.11, significant frac=0.00, combined p=0.746, FDR q=0.922

## Quench Dynamics Claims
- `N=6`: Spearman(theta,max lambda)=1.000 (p=0), Spearman(theta,max M2)=0.800 (p=0.2), peak-slope coupling r=0.851 (p=0.149)

## Safe Wording
- Use: 'robust small-N evidence' when verdict is `supported`.
- Use: 'suggestive but mixed' when verdict is `mixed`.
- Avoid universal claims when verdict is `unsupported`.