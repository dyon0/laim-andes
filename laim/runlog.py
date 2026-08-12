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
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def setup_logging(run_dir: Path, level: str = 'INFO') -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(level.upper())
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(run_dir / 'run.log', encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger('laim')


def file_fingerprint(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {'path': str(p), 'exists': False}
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return {'path': str(p), 'exists': True, 'bytes': p.stat().st_size,
            'sha256': h.hexdigest()}


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

    def record_input(self, name: str, path: str | Path) -> None:
        self.data['inputs'][name] = file_fingerprint(path)
        self.flush()

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
        self.log.info('stage %s: start', self.name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        dt = time.time() - self.t0
        status = 'ok' if exc_type is None else f'failed: {exc_type.__name__}: {exc}'
        self.manifest.record_stage(self.name, dt, status=status)
        self.log.info('stage %s: %s (%.1fs)', self.name, status, dt)
