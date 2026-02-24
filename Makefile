PYTHON ?= python3
CONFIG ?= configs/default.yaml

.PHONY: all quench nnqs validate audit seed-sweep showcase report-pdf cache-runs release-assets report-assets clean

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

showcase:
	$(PYTHON) experiments/make_animation.py --grid outputs/sweep_grid.npz --out-gif outputs/figs/quench_dynamics.gif
	$(PYTHON) experiments/build_showcase.py --outputs outputs --docs docs

report-pdf:
	$(PYTHON) experiments/build_report_pdf.py --outputs outputs --out report/short_report.pdf

cache-runs:
	$(PYTHON) experiments/cache_runs.py --outputs outputs --cache cached_runs

release-assets: showcase report-pdf cache-runs

report-assets:
	mkdir -p report/figs
	cp outputs/figs/*.png report/figs/ 2>/dev/null || true

clean:
	rm -rf outputs
