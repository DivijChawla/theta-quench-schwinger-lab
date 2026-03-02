# Statistical Rigor Report

- alpha: `0.05`
- target power: `0.8`
- mean snapshots per condition: `6`

## Power analysis
- r=0.25: power@mean_n=0.073, min_n_for_target_power=124
- r=0.30: power@mean_n=0.084, min_n_for_target_power=85
- r=0.35: power@mean_n=0.097, min_n_for_target_power=62

## Mixed-effects model
- status: `ok`
- coef(magic_m2)=1.2981, p=0
- coef(snapshot_entropy)=1.1664, p=2.87e-122
- AIC=-216.495, BIC=-162.585, n_obs=6240

Artifacts:
- `outputs/stage1_prxq/stage1/statistical_rigor/power_analysis.csv`
- `outputs/stage1_prxq/stage1/statistical_rigor/statistical_rigor.json`