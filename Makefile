PYTHON ?= python3
CONFIG ?= configs/default.yaml
NOVELTY_CONFIG ?= configs/novelty.yaml

.PHONY: all quench nnqs validate audit seed-sweep novelty novelty-quick publishability-status regime-sensitivity showcase capsule report-pdf cache-runs release-assets report-assets journal-grade study-stage1 study-stage2 stage2-universal stage2-universal-quick stage3-universal stage4-prl stage5-prl effect-registry statistical-rigor magic-calibration positive-control camera-ready clean

all:
	$(PYTHON) -m tqm.run --config $(CONFIG) --all

quench:
	$(PYTHON) experiments/run_quench_grid.py --config $(CONFIG)

nnqs:
	$(PYTHON) experiments/run_nnqs_snapshots.py --config $(CONFIG)

validate:
	$(PYTHON) -m tqm.run --config $(CONFIG) --validate

audit:
	$(PYTHON) experiments/manual_verification.py --config $(CONFIG)

seed-sweep:
	$(PYTHON) experiments/nnqs_seed_sweep.py --config $(CONFIG)

novelty:
	$(PYTHON) experiments/novelty_robustness.py --config $(NOVELTY_CONFIG) --out-dir outputs/novelty --n-sites 6,8 --seeds 11,17,23 --hidden-sizes 48 --bootstrap 200 --permutations 300

novelty-quick:
	$(PYTHON) experiments/novelty_robustness.py --config $(CONFIG) --out-dir outputs/novelty_quick --quick --epochs 40 --measurement-samples 4000 --snapshot-count 6

publishability-status:
	$(PYTHON) experiments/build_publishability_final.py --baseline gru=outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --baseline made=outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --baseline rbm=outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --baseline independent=outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --seed outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --n10 outputs/stage1_prxq/n10_power_exact/novelty_summary.json --regime alt_baseline=outputs/stage1_prxq/alt_regime_baseline/novelty_summary.json --regime alt_heavy_mass=outputs/stage1_prxq/alt_regime_heavy_mass/novelty_summary.json --regime alt_strong_coupling=outputs/stage1_prxq/alt_regime_strong_coupling/novelty_summary.json --out report/publishability_status.md

regime-sensitivity:
	$(PYTHON) experiments/build_regime_sensitivity.py --baseline gru=outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --baseline made=outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --baseline rbm=outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --baseline independent=outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --alt alt_baseline=outputs/stage1_prxq/alt_regime_baseline/novelty_summary.json --alt alt_heavy_mass=outputs/stage1_prxq/alt_regime_heavy_mass/novelty_summary.json --alt alt_strong_coupling=outputs/stage1_prxq/alt_regime_strong_coupling/novelty_summary.json --out report/regime_sensitivity.md

effect-registry:
	$(PYTHON) experiments/build_effect_registry.py --study-id stage1_prxq --phase stage1 --primary-endpoint pearson_magic_vs_val_nll_partial_entropy --power-target "Detect r in [0.25,0.35] with >=10 seeds per headline condition" --summary baseline_n6n8_multiarch=outputs/stage1_prxq/baseline_n6n8_multiarch/novelty_summary.json --summary n10_power_exact=outputs/stage1_prxq/n10_power_exact/novelty_summary.json --summary alt_regime_baseline=outputs/stage1_prxq/alt_regime_baseline/novelty_summary.json --summary alt_regime_heavy_mass=outputs/stage1_prxq/alt_regime_heavy_mass/novelty_summary.json --summary alt_regime_strong_coupling=outputs/stage1_prxq/alt_regime_strong_coupling/novelty_summary.json --out outputs/stage1_prxq/stage1/effect_registry.json

study-stage1:
	$(PYTHON) -m tqm.run --study stage1_prxq --phase stage1

study-stage2:
	$(PYTHON) -m tqm.run --study stage2_prlx --phase stage2

stage2-universal:
	$(PYTHON) experiments/stage2_universal_law.py --config configs/stage2_universal.yaml --out-dir outputs/stage2_prlx/universal_scan_v2
	$(PYTHON) experiments/stage2_corridor_robustness.py --in-dir outputs/stage2_prlx/universal_scan_v2 --quantiles 0.2,0.3,0.4,0.5,0.6
	cp outputs/stage2_prlx/universal_scan_v2/corridor_robustness.csv outputs/stage2_prlx/universal_scan_v2/corridor_robustness_all_arch.csv
	cp outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary.json outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_all_arch.json
	$(PYTHON) experiments/stage2_corridor_robustness.py --in-dir outputs/stage2_prlx/universal_scan_v2 --architectures gru,made --quantiles 0.2,0.3,0.4,0.5,0.6
	cp outputs/stage2_prlx/universal_scan_v2/corridor_robustness.csv outputs/stage2_prlx/universal_scan_v2/corridor_robustness_expressive.csv
	cp outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary.json outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_expressive.json
	$(PYTHON) experiments/stage2_mechanism_regression.py --in-dir outputs/stage2_prlx/universal_scan_v2
	$(PYTHON) experiments/build_stage2_robustness_report.py --all-arch outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_all_arch.json --expressive outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_expressive.json --out report/stage2_corridor_robustness.md
	$(PYTHON) experiments/build_stage2_effect_registry.py --study-id stage2_prlx --phase stage2 --summary outputs/stage2_prlx/universal_scan_v2/universal_law_summary.json --all-arch-robustness outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_all_arch.json --expressive-robustness outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_expressive.json --pooled outputs/stage2_prlx/universal_scan_v2/cross_model_pooled_effects.csv --beta outputs/stage2_prlx/universal_scan_v2/cross_model_beta_bounds.csv --mechanism outputs/stage2_prlx/universal_scan_v2/stage2_mechanism_summary.json --out outputs/stage2_prlx/stage2/effect_registry.json
	$(PYTHON) experiments/stage2_claim_gate.py --registry outputs/stage2_prlx/stage2/effect_registry.json --targets README.md,report/main.tex,report/stage2_universal_law.md
	$(PYTHON) experiments/build_stage2_completion_report.py --registry outputs/stage2_prlx/stage2/effect_registry.json --summary outputs/stage2_prlx/universal_scan_v2/universal_law_summary.json --all-robustness outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_all_arch.json --expressive-robustness outputs/stage2_prlx/universal_scan_v2/corridor_robustness_summary_expressive.json --out report/stage2_completion.md
	$(PYTHON) experiments/audit_stage2.py --in-dir outputs/stage2_prlx/universal_scan_v2 --registry outputs/stage2_prlx/stage2/effect_registry.json --out outputs/stage2_prlx/stage2/stage2_audit.json
	cp outputs/stage2_prlx/universal_scan_v2/figs/fig_stage2_*.png outputs/figs/ 2>/dev/null || true

stage2-universal-quick:
	$(PYTHON) experiments/stage2_universal_law.py --config configs/stage2_universal_quick.yaml --out-dir outputs/stage2_prlx/universal_scan_quick

stage3-universal:
	$(PYTHON) experiments/stage3_universal_extension.py --config configs/stage3_universal_v3.yaml --out-dir outputs/stage3_prx/universal_extension_v3 --registry-out outputs/stage3_prx/stage3/effect_registry.json --report-out report/stage3_extension.md
	cp outputs/stage3_prx/universal_extension_v3/figs/fig_stage3_*.png outputs/figs/ 2>/dev/null || true

stage4-prl:
	$(PYTHON) experiments/stage3_universal_extension.py --config configs/stage3_universal_v8.yaml --out-dir outputs/stage3_prx/universal_extension_v8 --registry-out outputs/stage3_prx/stage3_v8/effect_registry.json --report-out report/stage3_extension_v8.md
	$(PYTHON) experiments/stage4_thermo_theory.py --stage3-dir outputs/stage3_prx/universal_extension_v8 --registry outputs/stage3_prx/stage3_v8/effect_registry.json --out-dir outputs/stage4_prl/thermo_theory_v5 --report-out report/stage4_thermo_theory.md
	cp outputs/stage3_prx/universal_extension_v8/figs/fig_stage3_*.png outputs/figs/ 2>/dev/null || true
	cp outputs/stage4_prl/thermo_theory_v5/figs/fig_stage4_*.png outputs/figs/ 2>/dev/null || true

stage5-prl:
	$(PYTHON) experiments/stage5_external_regime_expansion.py --config configs/stage5_external_v1.yaml --out-dir outputs/stage5_prx/external_regime_v1 --registry-out outputs/stage5_prx/stage5/effect_registry.json --report-out report/stage5_external_regime.md
	cp outputs/stage5_prx/external_regime_v1/figs/fig_stage5_*.png outputs/figs/ 2>/dev/null || true

statistical-rigor:
	$(PYTHON) experiments/statistical_rigor_report.py --snapshot-csv outputs/stage1_prxq/stage1/nnqs_snapshot_all_stage1.csv --condition-csv outputs/stage1_prxq/stage1/nnqs_condition_stats_stage1.csv --out-dir outputs/stage1_prxq/stage1/statistical_rigor --report report/statistical_rigor.md

magic-calibration:
	$(PYTHON) experiments/calibrate_magic_estimator.py --config configs/default.yaml --n-sites 8 --theta1 0.0,1.2,2.4 --sample-grid 1000,2500,5000,10000,20000 --n-times 8 --out-dir outputs/magic_calibration

positive-control:
	$(PYTHON) experiments/positive_control_magic_family.py --n-sites 6 --architectures gru,made,independent --seeds 11,17,23,29 --alphas 0.1,0.2,0.35,0.5,0.7,0.9 --measurement-samples 3000 --epochs 35 --out-dir outputs/positive_control

showcase:
	$(PYTHON) experiments/make_animation.py --grid outputs/sweep_grid.npz --out-gif outputs/figs/quench_dynamics.gif
	$(PYTHON) experiments/build_showcase.py --outputs outputs --docs docs

capsule:
	$(PYTHON) experiments/build_project_capsule.py --out-gif outputs/figs/project_capsule.gif --out-mp4 outputs/figs/project_capsule.mp4
	$(PYTHON) experiments/build_showcase.py --outputs outputs --docs docs

report-pdf:
	$(PYTHON) experiments/build_report_pdf.py --outputs outputs --out report/short_report.pdf

cache-runs:
	$(PYTHON) experiments/cache_runs.py --outputs outputs --cache cached_runs

release-assets: showcase report-pdf cache-runs

report-assets:
	mkdir -p report/figs
	cp outputs/figs/*.png report/figs/ 2>/dev/null || true

journal-grade:
	$(PYTHON) experiments/novelty_robustness.py --config configs/novelty.yaml --out-dir outputs/novelty_gru_seeds8_n6_fulltheta --n-sites 6 --theta1 0.0,0.8,1.2,1.6,2.0,2.4 --architectures gru --seeds 11,13,17,19,23,29,31,37 --hidden-sizes 48 --bootstrap 200 --permutations 300 --epochs 45 --measurement-samples 4000 --snapshot-count 5 --checkpoint-every 6
	$(PYTHON) experiments/novelty_robustness.py --config configs/n10_exact.yaml --out-dir outputs/novelty_n10_exact_power --n-sites 10 --theta1 0.0,1.2,2.4 --architectures gru,rbm,independent --seeds 11,17,23 --hidden-sizes 32 --bootstrap 250 --permutations 400 --epochs 45 --measurement-samples 4000 --snapshot-count 8 --checkpoint-every 3
	$(PYTHON) experiments/novelty_robustness.py --config configs/alt_regime_heavy_mass.yaml --out-dir outputs/novelty_alt_regime_heavy_mass --n-sites 6 --theta1 0.0,0.9,1.8,2.7 --architectures gru,rbm,independent --seeds 11,17,23 --hidden-sizes 48 --bootstrap 200 --permutations 300 --epochs 45 --measurement-samples 4000 --snapshot-count 4 --checkpoint-every 6
	$(PYTHON) experiments/novelty_robustness.py --config configs/alt_regime_strong_coupling.yaml --out-dir outputs/novelty_alt_regime_strong_coupling --n-sites 6 --theta1 0.0,0.9,1.8,2.7 --architectures gru,rbm,independent --seeds 11,17,23 --hidden-sizes 48 --bootstrap 200 --permutations 300 --epochs 45 --measurement-samples 4000 --snapshot-count 4 --checkpoint-every 6
	$(PYTHON) experiments/build_regime_boundary_plot.py --out outputs/figs/fig6_regime_boundary_magic_valnll.png
	$(PYTHON) experiments/build_publishability_final.py --out report/publishability_status.md
	$(PYTHON) experiments/build_regime_sensitivity.py --out report/regime_sensitivity.md
	$(PYTHON) experiments/build_effect_registry.py --study-id stage1_prxq --phase stage1 --primary-endpoint pearson_magic_vs_val_nll_partial_entropy --power-target "r=0.25-0.35" --summary baseline=outputs/novelty/novelty_summary.json --summary independent=outputs/novelty_arch_independent/novelty_summary.json --summary rbm=outputs/novelty_arch_rbm/novelty_summary.json --summary seed8=outputs/novelty_gru_seeds8_n6_fulltheta/novelty_summary.json --summary n10=outputs/novelty_n10_exact_power/novelty_summary.json --summary alt1=outputs/novelty_alt_regime/novelty_summary.json --summary alt2=outputs/novelty_alt_regime_heavy_mass/novelty_summary.json --summary alt3=outputs/novelty_alt_regime_strong_coupling/novelty_summary.json --out outputs/stage1_prxq/stage1/effect_registry.json
	$(PYTHON) experiments/statistical_rigor_report.py --snapshot-csv outputs/novelty/nnqs_snapshot_all.csv --condition-csv outputs/novelty/nnqs_condition_stats.csv --out-dir outputs/statistical_rigor --report report/statistical_rigor.md
	$(PYTHON) experiments/calibrate_magic_estimator.py --config configs/default.yaml --n-sites 8 --theta1 0.0,1.2,2.4 --sample-grid 1000,2500,5000,10000,20000 --n-times 8 --out-dir outputs/magic_calibration
	$(PYTHON) experiments/positive_control_magic_family.py --n-sites 6 --architectures gru,made,independent --seeds 11,17,23,29 --alphas 0.1,0.2,0.35,0.5,0.7,0.9 --measurement-samples 3000 --epochs 35 --out-dir outputs/positive_control
	$(PYTHON) experiments/build_compute_budget_table.py --out report/compute_budget.md
	$(PYTHON) experiments/build_showcase.py --outputs outputs --docs docs
	$(PYTHON) experiments/build_report_pdf.py --outputs outputs --out report/short_report.pdf

camera-ready:
	$(PYTHON) experiments/camera_ready.py --study stage1_prxq --phase stage1 --artifact-root artifacts/camera_ready

clean:
	rm -rf outputs
