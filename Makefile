PY ?= .venv/bin/python

.PHONY: test test-all train-smoke lint golden

# Fast CPU suite — the Phase 3 gate: green in < 2 min
test:
	$(PY) -m pytest -q -m "not slow and not gpu" --timeout 240

# Everything (includes micro-training characterization)
test-all:
	$(PY) -m pytest -q -m "not gpu"

# Regenerate golden characterization artifacts (only when a behavior change is intended)
golden:
	$(PY) tests/golden/regenerate.py

# End-to-end smoke on the fixture corpus
train-smoke:
	$(PY) run.py all --config configs/smoke.toml
