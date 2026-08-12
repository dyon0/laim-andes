"""LAIM orchestration layer.

Thin, testable wiring around the legacy `ars` stages: typed config (TOML + CLI),
structured logging, run manifests, a full evaluation surface, and a single
entry point (`run.py`). The numeric pipeline lives in `ars/`; this package must
not duplicate it.
"""
