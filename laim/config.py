"""Typed run configuration: defaults ← TOML file ← CLI dotted overrides.

Design: one flat tree of frozen dataclasses. Every knob the audit flagged as
hardcoded is surfaced here. `to_s1_overrides()` / `to_s2_overrides()` map onto
the legacy stage configs so the numeric path stays byte-identical (Phase 4 is
behavior-preserving; fixes change `ars/`, not this mapping).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    train_spans: str = ''                 # parquet with training spans (required for prepare/all)
    infer_spans: str = ''                 # parquet to score (infer/all)
    embedder: str = '/tmp/models/USER-bge-m3-standin'
    output_root: str = 'runs'             # run directories live here


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = 'cpu'                   # 'cpu' | 'gpu'
    seed: int = 12345
    recast: bool = True                   # repair dtype drift on load
    progress: bool = False
    log_level: str = 'INFO'


@dataclass(frozen=True)
class DataConfig:
    """s1 knobs (mapped onto S1Config)."""
    min_fill_rate: float = 0.5
    max_static_rate: float = 0.95
    max_correlation: float = 0.999
    winsorize_epi: bool = True
    epi_normalization: str = 'robust'
    norm_train_ratio: float = 0.70
    norm_val_ratio: float = 0.15
    anom_val_ratio: float = 0.70
    anom_test_ratio: float = 0.30
    inject_anomalies: bool = True
    validation_gate: str = 'warn'         # 'off' | 'warn' | 'strict'  (wired in Phase 5)
    scale_floor: float = 0.0              # 0.0 = legacy behavior (F-05 fix raises it)


@dataclass(frozen=True)
class DetectorConfig:
    """s2 knobs (mapped onto S2Config / experiment grid)."""
    threshold_metric: str = 'youden'      # configurable per mission requirement
    select_metric: str = 'youden'
    select_on: str = 'val'                # F-02 fixed: selection on VAL ('test' = legacy bias)
    n_thresholds: int = 10000
    epochs: int = 10                      # legacy committed default (OQ-5)
    patience: int = 50
    encode_chunk: int = 1024
    seq_pad_chunk: int = 8192
    experiments: tuple[str, ...] = ('hub_mse_mse_08_4', 'hub_mse_hub_32_4_deep')


@dataclass(frozen=True)
class ClassifierConfig:
    enabled: bool = True
    seed: int = 12345


@dataclass(frozen=True)
class EvalConfig:
    target_fpr: float = 0.10
    assumed_prevalence: float = 0.01      # production anomaly rate for PPV projection
    calibration_bins: int = 10
    latency_reps: int = 50


@dataclass(frozen=True)
class SynthConfig:
    target_spans: int = 20000
    seed: int = 20250601
    out_path: str = ''


@dataclass(frozen=True)
class RunConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]

    # --- legacy stage mappings (behavior-preserving) ---

    def to_s1_overrides(self) -> dict:
        d, r = self.data, self.runtime
        return {
            'min_fill_rate': d.min_fill_rate,
            'max_static_rate': d.max_static_rate,
            'max_correlation': d.max_correlation,
            'winsorize_epi': d.winsorize_epi,
            'epi_normalization': d.epi_normalization,
            'norm_train_ratio': d.norm_train_ratio,
            'norm_val_ratio': d.norm_val_ratio,
            'anom_val_ratio': d.anom_val_ratio,
            'anom_test_ratio': d.anom_test_ratio,
            'inject_anomalies': d.inject_anomalies,
            'seed_random': r.seed, 'seed_polars': r.seed, 'seed_torch': r.seed,
            'seed_split': r.seed, 'seed_synth': r.seed, 'seed_llm': r.seed,
        }

    def to_s2_overrides(self) -> dict:
        det = self.detector
        return {
            'threshold_metric': det.threshold_metric,
            'select_metric': det.select_metric,
            'select_on': det.select_on,
            'experiments': det.experiments,
            'epochs': det.epochs,
            'patience': det.patience,
            'n_thresholds': det.n_thresholds,
            'encode_chunk': det.encode_chunk,
            'seq_pad_chunk': det.seq_pad_chunk,
            'seed': self.runtime.seed,
        }


def _merge(base: Any, patch: dict) -> Any:
    """Rebuild a frozen dataclass with a nested dict patch applied."""
    kwargs = {}
    for f in fields(base):
        val = getattr(base, f.name)
        if f.name in patch:
            p = patch[f.name]
            if dataclasses.is_dataclass(val) and isinstance(p, dict):
                kwargs[f.name] = _merge(val, p)
            elif isinstance(val, tuple) and isinstance(p, list):
                kwargs[f.name] = tuple(p)
            else:
                kwargs[f.name] = type(val)(p) if val is not None and not isinstance(p, type(val)) and not dataclasses.is_dataclass(val) else p
        else:
            kwargs[f.name] = val
    return type(base)(**kwargs)


def _coerce(raw: str) -> Any:
    low = raw.strip().lower()
    if low in ('true', 'false'):
        return low == 'true'
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def load_config(config_file: str | Path | None = None,
                overrides: list[str] | None = None) -> RunConfig:
    """defaults ← TOML ← `section.key=value` overrides."""
    cfg = RunConfig()
    if config_file:
        patch = tomllib.loads(Path(config_file).read_text())
        cfg = _merge(cfg, patch)
    for item in overrides or []:
        key, _, raw = item.partition('=')
        if not _:
            raise ValueError(f'override must be section.key=value, got {item!r}')
        parts = key.strip().split('.')
        nest: dict = {}
        node = nest
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(raw)
        cfg = _merge(cfg, nest)
    return cfg
