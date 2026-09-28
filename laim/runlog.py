"""Structured logging + run manifest.

Every pipeline run gets a directory under `paths.output_root` containing:
  manifest.json  — config, config hash, data fingerprints, seeds, package
                   versions, git SHA, stage timings, metrics, artifact paths
  run.log        — plain-text log (root logger)
The manifest is written incrementally (crash-safe: whatever completed is on disk).
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


class _EarlyBuffer(logging.Handler):
    """Keeps records logged before the run directory exists (setup_logging
    replays them into run.log)."""
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def early_logging(level: str = 'INFO') -> None:
    """Logging for the part of a run BEFORE the run directory exists (the
    platform's run_node: thread alignment, GPU topology, port staging...).
    Without it those INFO records went nowhere — the root logger had no
    handler yet, and Python's last-resort handler prints WARNING+ only.
    Records go to stderr now and are replayed into run.log by setup_logging.
    Idempotent; a process that already configured logging is left alone."""
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level.upper())
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(sh)
    root.addHandler(_EarlyBuffer())


def setup_logging(run_dir: Path, level: str = 'INFO') -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level.upper())
    fmt = logging.Formatter(_LOG_FORMAT)
    early = [r for h in root.handlers if isinstance(h, _EarlyBuffer) for r in h.records]
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(run_dir / 'run.log', encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    for record in early:            # already on stderr; the file gets them too
        if record.levelno >= root.level:
            fh.handle(record)
    return logging.getLogger('laim')


# full-content sha256 only up to this size; larger files get a sampled hash
# (platform ports carry ~56 GB — a full read would dominate the run)
_FULL_HASH_MAX_BYTES = 2 << 30
_SAMPLE_BYTES = 64 << 20


def _hash_file(p: Path) -> tuple[str, str]:
    size = p.stat().st_size
    h = hashlib.sha256()
    if size <= _FULL_HASH_MAX_BYTES:
        with open(p, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest(), 'full'
    with open(p, 'rb') as f:
        h.update(str(size).encode())
        h.update(f.read(_SAMPLE_BYTES))
        f.seek(-_SAMPLE_BYTES, 2)
        h.update(f.read(_SAMPLE_BYTES))
    return h.hexdigest(), 'sampled-head-tail'


def file_fingerprint(path: str | Path) -> dict:
    """Fingerprint a file, a directory of parts, or a glob pattern.

    Directories/globs (how SberDS delivers dataframe ports) are fingerprinted
    by a manifest hash over sorted (relative path, size) pairs — reading tens
    of GB of parts for a content hash is not affordable at run start.
    """
    p = Path(path)
    if '*' in str(path):
        import glob as _glob
        files = sorted(Path(f) for f in _glob.glob(str(path), recursive=True)
                       if Path(f).is_file())
        if not files:
            return {'path': str(path), 'exists': False}
        root = Path(str(path).split('*', 1)[0]).parent
        return _dir_fingerprint(str(path), root, files)
    if not p.exists():
        return {'path': str(p), 'exists': False}
    if p.is_dir():
        files = sorted(f for f in p.rglob('*') if f.is_file())
        return _dir_fingerprint(str(p), p, files)
    digest, mode = _hash_file(p)
    return {'path': str(p), 'exists': True, 'bytes': p.stat().st_size,
            'sha256': digest, 'mode': mode}


def _dir_fingerprint(label: str, root: Path, files: list[Path]) -> dict:
    h = hashlib.sha256()
    total = 0
    for f in files:
        size = f.stat().st_size
        total += size
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = f
        h.update(f'{rel.as_posix()}\t{size}\n'.encode())
    return {'path': label, 'exists': True, 'bytes': total,
            'sha256': h.hexdigest(), 'mode': 'name-size-manifest',
            'files': len(files)}


def gpu_topology() -> dict:
    """GPU inventory via nvidia-smi — no CUDA context is created, so this is
    safe to call before torch/jax initialize and costs nothing on CPU hosts."""
    import os
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,driver_version',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'available': False, 'reason': f'{type(exc).__name__}: {exc}'}
    if r.returncode != 0:
        return {'available': False, 'reason': (r.stderr or r.stdout).strip()[:200]}
    gpus = []
    for line in r.stdout.strip().splitlines():
        try:
            idx, name, total, used, driver = (x.strip() for x in line.split(',', 4))
            gpus.append({'index': int(idx), 'name': name,
                         'memory_total_mib': int(total), 'memory_used_mib': int(used),
                         'driver': driver})
        except ValueError:
            continue
    return {'available': bool(gpus), 'count': len(gpus), 'gpus': gpus,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'xla_preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
            'xla_mem_fraction': os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class Manifest:
    def __init__(self, run_dir: Path, config: Any) -> None:
        self.path = run_dir / 'manifest.json'
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                             text=True, cwd=Path(__file__).parents[1]).stdout.strip()
        self.data: dict = {
            'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'git_sha': git,
            'python': sys.version,
            'platform': platform.platform(),
            'config': config.to_dict() if hasattr(config, 'to_dict') else config,
            'config_hash': config.config_hash() if hasattr(config, 'config_hash') else None,
            'inputs': {},
            'stages': {},
            'metrics': {},
            'artifacts': {},
        }
        self.flush()

    def record_input(self, name: str, path: str | Path) -> dict:
        fp = file_fingerprint(path)
        self.data['inputs'][name] = fp
        self.flush()
        return fp

    def record_stage(self, name: str, seconds: float, **extra: Any) -> None:
        self.data['stages'][name] = {'seconds': round(seconds, 3), **extra}
        self.flush()

    def record_metrics(self, name: str, metrics: dict) -> None:
        self.data['metrics'][name] = metrics
        self.flush()

    def record_artifact(self, name: str, path: str | Path) -> None:
        self.data['artifacts'][name] = str(path)
        self.flush()

    def flush(self) -> None:
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False, default=_jsonable))


class StageTimer:
    def __init__(self, manifest: Manifest, name: str, log: logging.Logger) -> None:
        self.manifest, self.name, self.log = manifest, name, log

    def __enter__(self) -> 'StageTimer':
        self.t0 = time.time()
        push_phase(self.name, stage=True)
        self.log.info('stage %s: start', self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dt = time.time() - self.t0
        pop_phase(self.name, stage=True)
        status = 'ok' if exc_type is None else f'failed: {exc_type.__name__}: {exc}'
        self.manifest.record_stage(self.name, dt, status=status)
        self.log.info('stage %s: %s (%.1fs)', self.name, status, dt)


# ------------------------------------------------------ resource attribution
#
# The platform reports only whole-run averages (e.g. "cpu usage 56%, avg sm
# load 27%"), which cannot say WHICH step burns CPU while the GPU idles. The
# monitor below attributes every sample to the current stage (StageTimer) and
# step (the innermost ars @benchmark), and XLA compile time via jax.monitoring.

_PHASE_LOCK = threading.Lock()
_STAGE: list[str] = []
_STEPS: list[str] = []


_MONITORS: list['ResourceMonitor'] = []


def _tick_monitors() -> None:
    for mon in tuple(_MONITORS):              # close the running phase exactly
        mon.tick()


def push_phase(name: str, stage: bool = False) -> None:
    _tick_monitors()
    with _PHASE_LOCK:
        (_STAGE if stage else _STEPS).append(name)


def pop_phase(name: str, stage: bool = False) -> None:
    _tick_monitors()
    with _PHASE_LOCK:
        stack = _STAGE if stage else _STEPS
        if name in stack:                       # tolerate unbalanced exits
            del stack[len(stack) - 1 - stack[::-1].index(name)]
        if stage:
            _STEPS.clear()


def current_phase() -> tuple[str, str]:
    with _PHASE_LOCK:
        stage = _STAGE[-1] if _STAGE else 'setup'
        step = _STEPS[-1] if _STEPS else '-'
    return stage, step


def _benchmark_hook(name: str, entering: bool) -> None:
    (push_phase if entering else pop_phase)(name)


def cpu_quota_cores() -> float | None:
    """The container's CPU budget (cgroup v2, then v1); None if unlimited."""
    try:
        raw = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if raw[0] != 'max':
            return int(raw[0]) / int(raw[1])
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
        period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
        if quota > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


class ResourceMonitor:
    """Per stage and per step: wall seconds, process CPU core-seconds (all
    threads; child processes excluded), average busy cores, peak RSS, GPU
    utilization / memory (one long-running `nvidia-smi -lms`, never an
    in-process CUDA call — it must not touch the device set, see
    tests/test_runtime_device.py) and XLA compile seconds / count
    (jax.monitoring). Start it after the runtime device is applied."""

    _COMPILE_EVENT = '/jax/core/compile/backend_compile_duration'

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self._acc: dict[tuple[str, str], dict[str, float]] = {}
        self._gpu: dict[int, tuple[float, float]] = {}
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._smi: subprocess.Popen | None = None
        self._listener = None
        self._lock = threading.Lock()

    def _bucket(self, key: tuple[str, str]) -> dict[str, float]:   # call under self._lock
        return self._acc.setdefault(key, {
            'wall_s': 0.0, 'cpu_core_s': 0.0, 'gpu_util_x_s': 0.0, 'gpu_s': 0.0,
            'gpu_mem_max_mib': 0.0, 'rss_max_mb': 0.0, 'xla_compile_s': 0.0,
            'xla_compiles': 0.0})

    def start(self) -> 'ResourceMonitor':
        try:
            self._smi = subprocess.Popen(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                 '--format=csv,noheader,nounits', '-lms', str(int(self.interval * 1000))],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            t = threading.Thread(target=self._read_smi, name='laim-smi', daemon=True)
            t.start()
            self._threads.append(t)
        except OSError:
            self._smi = None                    # no nvidia-smi: CPU-only host
        try:
            import jax.monitoring as jm

            def on_duration(event: str, duration: float, **_: Any) -> None:
                if event == self._COMPILE_EVENT:
                    key = current_phase()
                    with self._lock:
                        b = self._bucket(key)
                        b['xla_compile_s'] += duration
                        b['xla_compiles'] += 1
            jm.register_event_duration_secs_listener(on_duration)
            self._listener = on_duration
        except Exception:                        # monitoring must never break a run
            self._listener = None
        try:
            from ars.tools.performance import perf
            if _benchmark_hook not in perf.PHASE_HOOKS:
                perf.PHASE_HOOKS.append(_benchmark_hook)
        except Exception:
            pass
        import psutil
        self._proc = psutil.Process()
        self._last = (time.monotonic(), self._cpu())
        _MONITORS.append(self)
        t = threading.Thread(target=self._sample, name='laim-monitor', daemon=True)
        t.start()
        self._threads.append(t)
        return self

    def _cpu(self) -> float:
        return sum(self._proc.cpu_times()[:2])   # user + system, all threads

    def _read_smi(self) -> None:
        assert self._smi is not None and self._smi.stdout is not None
        for line in self._smi.stdout:
            try:
                idx, util, mem = (float(x) for x in line.split(','))
                self._gpu[int(idx)] = (util, mem)
            except ValueError:
                continue

    def tick(self) -> None:
        """Attribute everything since the previous tick to the CURRENT phase —
        called periodically and at every phase boundary (push/pop_phase), so
        short steps are measured exactly, not missed between samples."""
        key = current_phase()
        rss = self._proc.memory_info().rss / 2**20
        gpu = dict(self._gpu)
        with self._lock:
            now, c = time.monotonic(), self._cpu()
            dt, dc = now - self._last[0], c - self._last[1]
            self._last = (now, c)
            b = self._bucket(key)
            b['wall_s'] += dt
            b['cpu_core_s'] += dc
            b['rss_max_mb'] = max(b['rss_max_mb'], rss)
            if gpu:
                utils = [u for u, _ in gpu.values()]
                b['gpu_util_x_s'] += sum(utils) / len(utils) * dt
                b['gpu_s'] += dt
                b['gpu_mem_max_mib'] = max(b['gpu_mem_max_mib'], max(m for _, m in gpu.values()))

    def _sample(self) -> None:
        while not self._stop.wait(self.interval):
            self.tick()

    def stop(self) -> dict:
        if self in _MONITORS:
            _MONITORS.remove(self)
        self.tick()                               # the tail since the last sample
        self._stop.set()
        if self._smi is not None:
            self._smi.terminate()
        for t in self._threads:
            t.join(timeout=5)
        if self._listener is not None:
            try:
                import jax.monitoring as jm
                jm.unregister_event_duration_listener(self._listener)
            except Exception:
                pass
        try:
            from ars.tools.performance import perf
            if _benchmark_hook in perf.PHASE_HOOKS:
                perf.PHASE_HOOKS.remove(_benchmark_hook)
        except Exception:
            pass
        return self.summary()

    @staticmethod
    def _finish(b: dict[str, float]) -> dict[str, Any]:
        wall = b['wall_s']
        return {
            'wall_s': round(wall, 1),
            'cpu_core_s': round(b['cpu_core_s'], 1),
            'cpu_cores_avg': round(b['cpu_core_s'] / wall, 2) if wall else None,
            'gpu_util_avg_pct': round(b['gpu_util_x_s'] / b['gpu_s'], 1) if b['gpu_s'] else None,
            'gpu_mem_max_mib': round(b['gpu_mem_max_mib']) if b['gpu_s'] else None,
            'rss_max_mb': round(b['rss_max_mb']),
            'xla_compile_s': round(b['xla_compile_s'], 1),
            'xla_compiles': int(b['xla_compiles']),
        }

    def summary(self) -> dict:
        with self._lock:
            acc = {k: dict(v) for k, v in self._acc.items()}

        def merge(keys) -> dict[str, float]:
            out = {k: 0.0 for k in ('wall_s', 'cpu_core_s', 'gpu_util_x_s', 'gpu_s',
                                    'gpu_mem_max_mib', 'rss_max_mb', 'xla_compile_s',
                                    'xla_compiles')}
            for key in keys:
                b = acc[key]
                for f in out:
                    out[f] = max(out[f], b[f]) if f.endswith('_max_mib') or f.endswith('_max_mb') else out[f] + b[f]
            return out
        stages = dict.fromkeys(k[0] for k in acc)
        return {
            'interval_s': self.interval,
            'cpu_quota_cores': cpu_quota_cores(),
            'gpu_sampled': self._smi is not None and bool(self._gpu),
            'total': self._finish(merge(list(acc))),
            'by_stage': {s: self._finish(merge([k for k in acc if k[0] == s])) for s in stages},
            'by_step': {f'{s} / {st}': self._finish(acc[(s, st)])
                        for (s, st) in sorted(acc, key=lambda k: -acc[k]['wall_s'])},
        }


def log_resource_summary(log: logging.Logger, summary: dict, top: int = 15) -> None:
    fmt = lambda k, v: (f'{k}: wall {v["wall_s"]}s, cpu {v["cpu_core_s"]} core-s '
                        f'({v["cpu_cores_avg"]} cores), gpu {v["gpu_util_avg_pct"]}%, '
                        f'xla compile {v["xla_compile_s"]}s/{v["xla_compiles"]}')
    log.info('resources total — %s', fmt('run', summary['total']))
    for k, v in summary['by_stage'].items():
        log.info('resources stage — %s', fmt(k, v))
    for k, v in list(summary['by_step'].items())[:top]:
        log.info('resources step — %s', fmt(k, v))


class monitored:
    """`with monitored(manifest, log):` — resource attribution for a run; the
    summary lands in manifest.metrics.resources and the log even on failure."""

    def __init__(self, manifest: Manifest, log: logging.Logger, interval: float = 1.0) -> None:
        self.manifest, self.log, self.monitor = manifest, log, ResourceMonitor(interval)

    def __enter__(self) -> ResourceMonitor:
        return self.monitor.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            summary = self.monitor.stop()
            self.manifest.record_metrics('resources', summary)
            log_resource_summary(self.log, summary)
        except Exception as e:                   # never mask the run's own error
            self.log.warning('resource monitor failed: %s', e)
