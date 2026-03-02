# Independent Audit (Pre-Publish)

Date: `2026-02-26`

## Code/Physics Correctness Checks
- `python3 -m pytest -q`: `10 passed`
- `python3 experiments/manual_verification.py --config configs/default.yaml`: `5/5 PASS`
  - Hamiltonian builder cross-check: PASS
  - Dense vs Krylov evolution: PASS
  - Norm conservation: PASS
  - Dense vs expm state match: PASS
  - Fast vs brute-force magic `M2`: PASS

## Claim and Overclaim Gates
- `python3 experiments/check_stage_gate.py --registry outputs/stage1_prxq/stage1/effect_registry.json --require-primary-headline --require-entropy-control`: PASS
- `python3 experiments/claim_gate.py --publishability report/publishability_status.md --targets README.md,report/main.tex`: PASS

## Stage 1 Dataset Integrity
- Combined Stage 1 condition table rows: `1080`
- Architectures present: `gru, made, rbm, independent`
- Sizes present: `N=6,8,10`
- `NaN` primary-correlation rows: `240` (all at `theta1=0.0`)
  - Interpretation: expected degenerate no-quench control (`theta0 -> theta1 = 0`) can yield near-zero variance in magic/learnability traces, so per-condition Pearson is undefined.
  - Registry meta-analysis excludes non-finite rows; this is tracked as boundary/control behavior, not a silent failure.

## Camera-Ready Reproducibility
- Command: `make camera-ready`
- Latest frozen artifact:
  - `artifacts/camera_ready/stage1_prxq_stage1_20260226T234726Z.zip`
- Camera-ready now rebuilds publishability/regime/effect-registry from registered Stage 1 outputs, not legacy default paths.
