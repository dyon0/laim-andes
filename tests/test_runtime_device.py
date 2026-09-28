"""device=gpu must mean GPU (regression after AUDIT_05: a platform training
with device=gpu ran entirely on CPU — 20 min became ~6 h).

Root cause: the pre-run config checks imported `ars` BEFORE the runtime device
was applied; `ars` stages import c0__env_setup (JAX_PLATFORMS=cpu,
CUDA_VISIBLE_DEVICES='' unless ARS_DEVICE=gpu) and perf.Hardware evaluated
jax.default_backend() at IMPORT, which initializes the JAX backend — after
that, Device.force('gpu') cannot move JAX anymore.

The checks run in fresh subprocesses: the pytest process itself has long
initialized JAX, so in-process they would prove nothing.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_PRELUDE = f'''
import json, os, sys
sys.path.insert(0, {str(REPO)!r})
for k in ('ARS_DEVICE', 'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES'):
    os.environ.pop(k, None)

def jax_initialized():
    if 'jax' not in sys.modules:
        return False
    from jax._src import xla_bridge
    return xla_bridge.backends_are_initialized()

def torch_cuda_initialized():
    t = sys.modules.get('torch')
    return bool(t is not None and t.cuda.is_initialized())
'''


def _run(body: str, timeout: int = 240) -> dict:
    r = subprocess.run([sys.executable, '-c', _PRELUDE + body], capture_output=True,
                       text=True, timeout=timeout, cwd=REPO)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_importing_ars_and_laim_initializes_no_backend():
    """No module may initialize the JAX backend or torch CUDA at import: an
    import can happen before the device is applied (perf.Hardware used to)."""
    out = _run('''
import importlib, pkgutil
offenders, failed = [], []
names = [m.name for pkg in ('ars', 'laim')
         for m in pkgutil.walk_packages(importlib.import_module(pkg).__path__, pkg + '.')]
for name in names:
    try:
        importlib.import_module(name)
    except Exception as exc:                      # a broken import is its own failure
        failed.append(f'{name}: {type(exc).__name__}: {exc}')
        continue
    if jax_initialized() or torch_cuda_initialized():
        offenders.append(name)                    # the FIRST module that did it
        break
print(json.dumps({'offenders': offenders, 'failed': failed}))
''')
    assert out['failed'] == []
    assert out['offenders'] == [], f'initializes a backend at import: {out["offenders"]}'


_STOP_AT_FORCE = '''
import ars.configuration.c0__device as dev
state = {}
class Stop(Exception):
    pass
def fake_force(self):
    state.update(name=self.name, jax_initialized_before_force=jax_initialized(),
                 torch_cuda_initialized_before_force=torch_cuda_initialized(),
                 ARS_DEVICE=os.environ.get('ARS_DEVICE'))
    raise Stop
dev.Device.force = fake_force
'''


def test_platform_train_applies_the_device_before_touching_backends(tmp_path):
    """The exact platform entry (run.py::main -> run_node -> run_train) with
    device=gpu: when the device is applied, JAX must still be uninitialized."""
    out = _run(_STOP_AT_FORCE + f'''
from laim.platform import run_node
try:
    run_node(mode='train', device='gpu', output_root={str(tmp_path / 'runs')!r},
             path_traces_train={str(tmp_path / 'missing.parquet')!r},
             path_embedder={str(tmp_path / 'emb')!r},
             experiments='["hub_mse_mse_08_4"]')
except Stop:
    pass
print(json.dumps(state))
''')
    assert out['jax_initialized_before_force'] is False     # the regression
    assert out['torch_cuda_initialized_before_force'] is False
    assert out['name'] == 'gpu' and out['ARS_DEVICE'] == 'gpu'


@pytest.mark.parametrize('command', ['train', 'infer'])
def test_cli_run_applies_the_device_before_touching_backends(tmp_path, command):
    out = _run(_STOP_AT_FORCE + f'''
from laim.config import load_config
from laim.pipeline import run
cfg = load_config(None, ['runtime.device=gpu', 'paths.output_root={tmp_path / "runs"}',
                         'paths.infer_spans={tmp_path / "x.parquet"}'])
try:
    run(cfg, {command!r}, model_dir={str(tmp_path)!r})
except Stop:
    pass
print(json.dumps(state))
''')
    assert out['jax_initialized_before_force'] is False     # the regression
    assert out['torch_cuda_initialized_before_force'] is False
    assert out['name'] == 'gpu' and out['ARS_DEVICE'] == 'gpu'


def test_gpu_requested_without_a_gpu_fails_loudly(tmp_path):
    """On a machine where JAX/torch cannot use a GPU, device=gpu must stop the
    run with an explanation — never fall back to hours on CPU."""
    out = _run(f'''
from laim.config import load_config
from laim.pipeline import _apply_runtime
try:
    _apply_runtime(load_config(None, ['runtime.device=gpu']))
    print(json.dumps({{'raised': None}}))
except RuntimeError as exc:
    print(json.dumps({{'raised': str(exc)}}))
''')
    assert out['raised'] is not None
    assert 'runtime.device=gpu' in out['raised'] and 'Refusing' in out['raised']


def test_verification_passes_when_jax_and_torch_see_a_gpu(monkeypatch):
    import jax
    import torch
    from laim.config import load_config
    from laim.pipeline import verify_runtime_device
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    monkeypatch.setattr(jax, 'devices', lambda: ['cuda:0'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)
    report = verify_runtime_device(load_config(None, ['runtime.device=gpu']))
    assert report['jax_backend'] == 'gpu' and report['torch_cuda_devices'] == 8


def test_verification_names_the_half_that_fell_back(monkeypatch):
    import jax
    import torch
    from laim.config import load_config
    from laim.pipeline import verify_runtime_device
    monkeypatch.setattr(jax, 'default_backend', lambda: 'cpu')
    monkeypatch.setattr(jax, 'devices', lambda: ['TFRT_CPU_0'])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 8)
    with pytest.raises(RuntimeError, match="JAX backend is 'cpu'"):
        verify_runtime_device(load_config(None, ['runtime.device=gpu']))
