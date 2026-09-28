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
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_PRELUDE = f'''
import importlib.abc, json, os, sys
sys.path.insert(0, {str(REPO)!r})
for k in ('ARS_DEVICE', 'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES'):
    os.environ.pop(k, None)

# every torch.cuda probe (is_available / device_count / _lazy_init), with the
# CUDA_VISIBLE_DEVICES value at call time: cudaGetDeviceCount -> cuInit reads
# it ONCE per process, so a probe made while it is '' pins torch AND JAX to
# 0 GPUs even though torch.cuda.is_initialized() stays False
TORCH_PROBES = []

class _TorchCudaHook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != 'torch.cuda':
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        exec_module = spec.loader.exec_module
        def patched(module):
            exec_module(module)
            for fn in ('is_available', 'device_count', '_lazy_init'):
                orig = getattr(module, fn)
                def wrap(*a, _orig=orig, _fn=fn, **k):
                    TORCH_PROBES.append((_fn, os.environ.get('CUDA_VISIBLE_DEVICES')))
                    return _orig(*a, **k)
                setattr(module, fn, wrap)
        spec.loader.exec_module = patched
        return spec

sys.meta_path.insert(0, _TorchCudaHook())

def jax_initialized():
    if 'jax' not in sys.modules:
        return False
    from jax._src import xla_bridge
    return xla_bridge.backends_are_initialized()

def torch_cuda_touched():
    t = sys.modules.get('torch')
    return bool(TORCH_PROBES) or bool(t is not None and t.cuda.is_initialized())
'''


@pytest.fixture(scope='module')
def flash_attn_stub(tmp_path_factory) -> Path:
    """The platform image ships flash-attn; with it installed, transformers
    probes torch.cuda.is_available() while `sentence_transformers` is being
    imported. A metadata-only stub reproduces that here."""
    root = tmp_path_factory.mktemp('flash_attn_stub')
    (root / 'flash_attn').mkdir()
    (root / 'flash_attn' / '__init__.py').write_text("__version__ = '2.8.3'\n")
    dist = root / 'flash_attn-2.8.3.dist-info'
    dist.mkdir()
    (dist / 'METADATA').write_text('Metadata-Version: 2.1\nName: flash-attn\nVersion: 2.8.3\n')
    (dist / 'top_level.txt').write_text('flash_attn\n')
    (dist / 'RECORD').write_text('flash_attn/__init__.py,,\n')
    return root


def _run(body: str, pythonpath: Path | None = None, timeout: int = 240) -> dict:
    env = dict(os.environ)
    if pythonpath is not None:
        env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(pythonpath), env.get('PYTHONPATH'))))
    r = subprocess.run([sys.executable, '-c', _PRELUDE + body], capture_output=True,
                       text=True, timeout=timeout, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_importing_ars_and_laim_initializes_no_backend(flash_attn_stub):
    """No module may initialize the JAX backend or probe torch CUDA at
    import: an import can happen before the device is applied (perf.Hardware
    used to initialize JAX; s1__data's module-level sentence_transformers
    import probed torch CUDA whenever flash-attn is installed)."""
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
    if jax_initialized() or torch_cuda_touched():
        offenders.append(name)                    # the FIRST module that did it
        break
print(json.dumps({'offenders': offenders, 'failed': failed,
                  'flash_attn_stub': 'flash_attn' in sys.modules or bool(
                      __import__('importlib.util').util.find_spec('flash_attn'))}))
''', pythonpath=flash_attn_stub)
    assert out['flash_attn_stub'] is True
    assert out['failed'] == []
    assert out['offenders'] == [], f'initializes a backend at import: {out["offenders"]}'


_STOP_AT_FORCE = '''
import ars.configuration.c0__device as dev
state = {}
class Stop(Exception):
    pass
def fake_force(self):
    state.update(name=self.name, jax_initialized_before_force=jax_initialized(),
                 torch_cuda_touched_before_force=torch_cuda_touched(),
                 ARS_DEVICE=os.environ.get('ARS_DEVICE'))
    raise Stop
dev.Device.force = fake_force
'''


def test_platform_train_applies_the_device_before_touching_backends(tmp_path, flash_attn_stub):
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
''', pythonpath=flash_attn_stub)
    assert out['jax_initialized_before_force'] is False     # the regression
    assert out['torch_cuda_touched_before_force'] is False
    assert out['name'] == 'gpu' and out['ARS_DEVICE'] == 'gpu'


@pytest.mark.parametrize('command', ['train', 'infer'])
def test_cli_run_applies_the_device_before_touching_backends(tmp_path, command, flash_attn_stub):
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
''', pythonpath=flash_attn_stub)
    assert out['jax_initialized_before_force'] is False     # the regression
    assert out['torch_cuda_touched_before_force'] is False
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
