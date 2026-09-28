"""Resource attribution: which stage / step burns CPU, leaves the GPU idle or
compiles XLA (the platform reports whole-run averages only)."""
import json
import time
from pathlib import Path

import pytest

from laim.runlog import ResourceMonitor, pop_phase, push_phase


def _spin(seconds: float) -> None:
    end = time.monotonic() + seconds
    x = 0
    while time.monotonic() < end:
        x += 1


def test_cpu_is_attributed_to_the_busy_step():
    mon = ResourceMonitor(interval=0.05).start()
    try:
        push_phase('stage_x', stage=True)
        push_phase('busy'); _spin(0.6); pop_phase('busy')
        push_phase('idle'); time.sleep(0.6); pop_phase('idle')
        pop_phase('stage_x', stage=True)
    finally:
        summary = mon.stop()
    busy, idle = summary['by_step']['stage_x / busy'], summary['by_step']['stage_x / idle']
    assert busy['cpu_core_s'] > 0.3 and busy['cpu_cores_avg'] > 0.5
    assert idle['cpu_core_s'] < 0.2
    assert summary['by_stage']['stage_x']['wall_s'] >= 1.0


def test_xla_compiles_are_attributed_to_their_step():
    import jax
    import jax.numpy as jp
    mon = ResourceMonitor(interval=0.05).start()
    try:
        push_phase('stage_c', stage=True)
        push_phase('compile_here')
        jax.jit(lambda v: jp.sin(v) * 3.0 + 0.123)(jp.ones((7, 13, 3)))   # fresh shape
        pop_phase('compile_here')
        pop_phase('stage_c', stage=True)
    finally:
        summary = mon.stop()
    step = summary['by_step']['stage_c / compile_here']
    assert step['xla_compiles'] >= 1 and step['xla_compile_s'] >= 0


def test_benchmark_steps_become_phases():
    from ars.tools.performance.perf import PHASE_HOOKS, benchmark

    @benchmark('шаг для теста')
    def work():
        _spin(0.3)

    mon = ResourceMonitor(interval=0.05).start()
    try:
        push_phase('stage_b', stage=True)
        work()
        pop_phase('stage_b', stage=True)
    finally:
        summary = mon.stop()
    assert 'stage_b / шаг для теста' in summary['by_step']
    assert PHASE_HOOKS == []                      # the hook is removed on stop


def test_run_records_resources_in_the_manifest(tmp_path):
    from laim.config import load_config
    from laim.pipeline import run
    out = run(load_config(None, [f'paths.output_root={tmp_path}',
                                 f'paths.train_spans={Path(__file__).parents[1] / "data" / "traces_1k_sample.parquet"}']),
              'validate')
    manifest = json.loads((Path(out['run_dir']) / 'manifest.json').read_text())
    res = manifest['metrics']['resources']
    assert 'validate' in res['by_stage']
    assert res['total']['wall_s'] >= 0 and 'cpu_cores_avg' in res['total']
    assert manifest['metrics']['runtime_device']['jax_backend'] == 'cpu'
