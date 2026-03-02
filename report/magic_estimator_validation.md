# Approximate Magic Estimator Validation

- n_sites: `8`
- theta1 values: `[0.0, 1.2, 2.4]`
- sample grid: `[1000, 2500, 5000, 10000, 20000]`

## Error envelope
- samples=1000: mean=4.2403e-01, median=4.0250e-01, max=1.1754e+00
- samples=2500: mean=2.4111e-01, median=1.8325e-01, max=6.7506e-01
- samples=5000: mean=1.5534e-01, median=1.1826e-01, max=5.1492e-01
- samples=10000: mean=1.3434e-01, median=1.1206e-01, max=3.3185e-01
- samples=20000: mean=8.0943e-02, median=7.4455e-02, max=2.1272e-01

Artifacts:
- `outputs/magic_calibration/magic_calibration_rows.csv`
- `outputs/magic_calibration/magic_calibration_summary.csv`
- `outputs/magic_calibration/figs/magic_calibration_error_vs_samples.png`