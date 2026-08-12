"""Compare two same-seed pipeline runs for identical model-relevant outputs.

Usage:
    python baseline/check_determinism.py '/tmp/mas-monitor#A' '/tmp/mas-monitor#B'

Compares, for each experiment: per-epoch loss curves (bit-level as floats),
selected threshold, val metrics, test metrics. Prints a JSON verdict.
"""
import json
import sys
from pathlib import Path

a, b = Path(sys.argv[1]), Path(sys.argv[2])
out = {'identical': True, 'experiments': {}}

exps_a = {p.parent.name: p for p in (a / 'traces_train' / 'models').glob('*/results.json')}
exps_b = {p.parent.name: p for p in (b / 'traces_train' / 'models').glob('*/results.json')}
if set(exps_a) != set(exps_b):
    out['identical'] = False
    out['experiment_sets'] = {'a': sorted(exps_a), 'b': sorted(exps_b)}

KEYS = ('best_threshold', 'epi_losses', 'sem_losses', 'combined_losses',
        'test_metrics', 'val_metrics', 'epi_train_mse', 'epi_val_mse')
for name in sorted(set(exps_a) & set(exps_b)):
    ra, rb = json.loads(exps_a[name].read_text()), json.loads(exps_b[name].read_text())
    diffs = {k: {'a': ra.get(k), 'b': rb.get(k)} for k in KEYS if ra.get(k) != rb.get(k)}
    out['experiments'][name] = {'identical': not diffs, 'diffs': diffs}
    if diffs:
        out['identical'] = False

print(json.dumps(out, indent=2, ensure_ascii=False))
