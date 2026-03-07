# Novelty Claim Assessment

This report converts robustness statistics into claim verdicts with explicit thresholds.

## NNQS Learnability Claims
- `gru | magic -> validation NLL`: **mixed** | pooled r=0.249 [95% CI -0.379, 0.720], studies=9, I2=0.44, positive frac=0.67, significant frac=0.11, combined p=0.114, FDR q=0.245
- `gru | entropy -> validation NLL`: **unsupported** | pooled r=-0.113 [95% CI -0.645, 0.493], studies=9, I2=0.00, positive frac=0.67, significant frac=0.00, combined p=0.943, FDR q=1
- `gru | magic -> validation NLL | entropy`: **mixed** | pooled r=0.281 [95% CI -0.349, 0.736], studies=9, I2=0.54, positive frac=0.67, significant frac=0.00, combined p=0.384, FDR q=0.524
- `gru | magic -> KL`: **unsupported** | pooled r=-0.376 [95% CI -0.781, 0.252], studies=9, I2=0.00, positive frac=0.22, significant frac=0.00, combined p=0.503, FDR q=0.629
- `gru | magic -> TV`: **unsupported** | pooled r=-0.391 [95% CI -0.788, 0.236], studies=9, I2=0.27, positive frac=0.33, significant frac=0.00, combined p=0.285, FDR q=0.427
- `independent | magic -> validation NLL`: **mixed** | pooled r=0.659 [95% CI 0.137, 0.895], studies=9, I2=0.61, positive frac=0.67, significant frac=0.33, combined p=0.0298, FDR q=0.112
- `independent | entropy -> validation NLL`: **unsupported** | pooled r=0.129 [95% CI -0.481, 0.654], studies=9, I2=0.00, positive frac=0.78, significant frac=0.00, combined p=1, FDR q=1
- `independent | magic -> validation NLL | entropy`: **mixed** | pooled r=0.435 [95% CI -0.186, 0.807], studies=9, I2=0.87, positive frac=0.67, significant frac=0.11, combined p=0.222, FDR q=0.37
- `independent | magic -> KL`: **supported** | pooled r=0.983 [95% CI 0.940, 0.995], studies=9, I2=0.00, positive frac=1.00, significant frac=0.22, combined p=0.000212, FDR q=0.00159
- `independent | magic -> TV`: **mixed** | pooled r=0.792 [95% CI 0.401, 0.939], studies=9, I2=0.00, positive frac=0.89, significant frac=0.11, combined p=0.0509, FDR q=0.127
- `rbm | magic -> validation NLL`: **mixed** | pooled r=0.651 [95% CI 0.123, 0.892], studies=9, I2=0.61, positive frac=0.67, significant frac=0.33, combined p=0.0298, FDR q=0.112
- `rbm | entropy -> validation NLL`: **unsupported** | pooled r=0.117 [95% CI -0.490, 0.648], studies=9, I2=0.00, positive frac=0.78, significant frac=0.00, combined p=1, FDR q=1
- `rbm | magic -> validation NLL | entropy`: **mixed** | pooled r=0.307 [95% CI -0.324, 0.749], studies=9, I2=0.89, positive frac=0.67, significant frac=0.11, combined p=0.218, FDR q=0.37
- `rbm | magic -> KL`: **supported** | pooled r=0.985 [95% CI 0.944, 0.996], studies=9, I2=0.00, positive frac=1.00, significant frac=0.22, combined p=0.000212, FDR q=0.00159
- `rbm | magic -> TV`: **mixed** | pooled r=0.810 [95% CI 0.442, 0.945], studies=9, I2=0.00, positive frac=0.89, significant frac=0.11, combined p=0.049, FDR q=0.127

## Quench Dynamics Claims
- `N=6`: Spearman(theta,max lambda)=1.000 (p=0), Spearman(theta,max M2)=1.000 (p=0), peak-slope coupling r=0.999 (p=0.00134)

## Safe Wording
- Use: 'robust small-N evidence' when verdict is `supported`.
- Use: 'suggestive but mixed' when verdict is `mixed`.
- Avoid universal claims when verdict is `unsupported`.