# Stage 1 Completion Status

## Gate Flags
- primary_supported_headline_region: `True`
- entropy_control_supported_any: `True`
- boundary_documented: `True`
- primary_supported_majority (global): `False`

## Study Registry Artifacts
- `outputs/stage1_prxq/stage1/effect_registry.json`
- `outputs/stage1_prxq/stage1/compute_budget.csv`
- `outputs/stage1_prxq/stage1/nnqs_snapshot_all_stage1.csv`
- `outputs/stage1_prxq/stage1/nnqs_condition_stats_stage1.csv`

## Independent Audit
- `python3 experiments/manual_verification.py --config configs/default.yaml`: PASS (5/5)
- `python3 -m pytest -q`: PASS (10 tests)
- `python3 experiments/claim_gate.py --publishability report/publishability_status.md --targets README.md,report/main.tex`: PASS
- `python3 experiments/check_stage_gate.py --registry outputs/stage1_prxq/stage1/effect_registry.json --require-primary-headline --require-entropy-control`: PASS
- consolidated audit packet: `report/independent_audit.md`

## Additional Reviewer-Driven Experiments
- Positive control synthetic family:
  - `outputs/positive_control/positive_control_summary.json`
- Approximate magic calibration:
  - `outputs/magic_calibration/magic_calibration_summary.json`
- Mixed-effects + power on Stage 1 combined snapshots:
  - `outputs/stage1_prxq/stage1/statistical_rigor/statistical_rigor.json`

## Runtime Accounting
- logged Stage 1 robustness runs: `5`
- summed runtime_seconds across Stage 1 runs: `4586.49`

## Interpretation
- Stage 1 conditional-claim acceptance is satisfied by headline-region + entropy-control + boundary-documentation gates.
- Global-majority support is intentionally not required because Stage 1 includes explicit boundary regimes by design.
- Camera-ready frozen artifact (latest): `artifacts/camera_ready/stage1_prxq_stage1_20260226T234855Z.zip`
