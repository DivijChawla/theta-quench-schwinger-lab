# Stage 2 Corridor-Robustness Report

## All-architecture gate
- quantiles pass (sign+beta): `False`
- quantiles pass (beta-only): `True`
- quantiles pass (sign-only): `False`

| q_lambda | n_conditions | n_snapshots | min pooled CI low | min beta CI low | sign+beta claim | beta-only claim |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 776 | 4936 | -0.3811 | 0.3118 | False | True |
| 0.3 | 684 | 4624 | -0.3958 | 0.3874 | False | True |
| 0.4 | 592 | 3992 | -0.4504 | 0.3521 | False | True |
| 0.5 | 500 | 3440 | -0.4727 | 0.4068 | False | True |
| 0.6 | 408 | 3128 | -0.4737 | 0.3025 | False | True |

## Expressive-only gate (GRU, MADE)
- quantiles pass (sign+beta): `True`
- quantiles pass (beta-only): `True`
- quantiles pass (sign-only): `True`

| q_lambda | n_conditions | n_snapshots | min pooled CI low | min beta CI low | sign+beta claim | beta-only claim |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2 | 404 | 2564 | 0.4475 | 0.9420 | True | True |
| 0.3 | 356 | 2396 | 0.4480 | 0.9387 | True | True |
| 0.4 | 308 | 2068 | 0.4652 | 0.8650 | True | True |
| 0.5 | 260 | 1780 | 0.3851 | 0.8768 | True | True |
| 0.6 | 212 | 1612 | 0.2791 | 0.8863 | True | True |

Conclusion:
- Beta-only law is robust across all tested quantiles.
- Correlation-sign law remains architecture-dependent.