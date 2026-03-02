# Mock PRXQ Reviewer Packet (Conditional-Mechanism Submission)

## Reviewer 1 (Methods/Statistics)

### Major concerns

1. Primary endpoint should be explicitly pre-registered and separated from secondary metrics.
2. Need model-agnostic controls showing the trend is not a training artifact.
3. Need formal effect registry with pass/fail gating rather than narrative-only claims.

### Action taken

- Added pre-registered protocol:
  - `report/analysis_protocol.md`
- Added mixed-effects + power report:
  - `report/statistical_rigor.md`
  - `outputs/stage1_prxq/stage1/statistical_rigor/statistical_rigor.json`
- Added standardized effect registry:
  - `outputs/stage1_prxq/stage1/effect_registry.json`
- Added stage gate checker:
  - `experiments/check_stage_gate.py`

## Reviewer 2 (Physics validity)

### Major concerns

1. Need explicit boundary mapping to avoid overclaiming universal behavior.
2. Need approximate-magic error envelope for larger-size use.
3. Need additional architecture to test model-family dependence.

### Action taken

- Added boundary map and report:
  - `outputs/figs/fig6_regime_boundary_magic_valnll.png`
  - `report/regime_sensitivity.md`
- Added approximate magic calibration:
  - `report/magic_estimator_validation.md`
  - `outputs/magic_calibration/magic_calibration_summary.csv`
- Added MADE architecture path integrated in pipeline.

## Reviewer 3 (Reproducibility/Artifact)

### Major concerns

1. Need one-command camera-ready freeze with deterministic artifacts.
2. Need automatic language gate to prevent overclaiming in manuscript/README.
3. Need compute budget transparency.

### Action taken

- Added camera-ready pipeline:
  - `make camera-ready`
  - `experiments/camera_ready.py`
- Updated camera-ready pipeline to rebuild publication artifacts from study-registry outputs
  (not legacy default output paths), and to regenerate `effect_registry.json` during freeze.
- Added claim gate:
  - `experiments/claim_gate.py`
- Added compute budget table builder:
  - `report/compute_budget.md`

## Remaining risks (transparent)

1. Stage 1 is conditional-claim complete, not a universal-law paper.
2. Stage 2 universal-law program still requires larger-N scaling and theorem-grade analysis.
3. Citation polish and final journal-style bibliography should be completed before submission.
