"""Characterization tests: pin CURRENT s1 behavior (including known bugs).

These tests exist to make the refactor safe: any unintended behavior change on
the fixed fixture corpus fails here. Tests marked `characterization_bug` pin
behavior that AUDIT_01 classifies as wrong — they are EXPECTED to change (and be
updated in the same commit) when the referenced finding is fixed.
"""
import math

import polars as pl
import pytest

from ars.configuration.c1__data import S1Config
from ars.data.features import FeaturePatterns, FeaturesSpan, RawSchema
from ars.specification.spec import DataObject

SEED = 12345


def _cfg(tmp_path) -> S1Config:
    return S1Config(
        input_parquet_files=(), output_dir=tmp_path, output_prefix='char',
        seed_random=SEED, seed_polars=SEED, seed_torch=SEED,
        seed_split=SEED, seed_synth=SEED, seed_llm=SEED)


def _with_label_stubs(spans: pl.DataFrame) -> pl.DataFrame:
    return spans.with_columns(
        pl.lit(DataObject.class_sentinel).alias(DataObject.label),
        pl.lit(DataObject.class_sentinel).alias(DataObject.sublabel),
        pl.lit(None, dtype=pl.Int8).alias(DataObject.is_anomaly))


@pytest.fixture(scope='module')
def features(fixture_spans):
    return FeaturesSpan().make_features(_with_label_stubs(fixture_spans), FeaturePatterns())


@pytest.mark.characterization
def test_local_feature_stats_pinned(features, golden):
    for name, expect in golden['local_feature_stats'].items():
        s = features[name].cast(pl.Float64)
        mean = s.mean()
        if expect['mean'] is None:
            assert mean is None, name
        else:
            assert mean == pytest.approx(expect['mean'], rel=1e-6, abs=1e-9), name
        assert (s.std() or 0.0) == pytest.approx(expect['std'], rel=1e-6, abs=1e-9), name
        assert s.null_count() == expect['nulls'], name


@pytest.mark.characterization_bug  # F-50: (end-start)-(end-start) ≡ 0
def test_duration_diff_is_identically_zero(features, golden):
    assert golden['duration_diff_is_all_zero'] is True
    assert features.select((pl.col('duration_diff') == 0).all()).item()


@pytest.mark.characterization_bug  # F-07: sentinel -1 written into a real-valued feature
def test_tool_compression_uses_sentinel_for_non_tool_spans(features):
    non_tool = features.filter(pl.col('aef_kind') != 'tool')
    # signed log1p of -1 = -log(2)
    assert non_tool.select(
        (pl.col('tool_compression') - (-math.log(2))).abs().max()).item() < 1e-6


@pytest.mark.characterization
def test_epi_feature_selection_pinned(fixture_spans, golden, tmp_path):
    from ars.stages.s1__data import calculate_features
    _, names = calculate_features(
        _with_label_stubs(fixture_spans), _cfg(tmp_path), FeaturePatterns(), RawSchema())
    assert list(names) == golden['epi_feature_names']


@pytest.mark.characterization
def test_injection_labels_pinned(fixture_spans, golden, tmp_path):
    from ars.data.anomalies_injection import InjectionConfig, inject_anomalies
    from ars.stages.s1__data import calculate_features
    spans_f, _ = calculate_features(
        _with_label_stubs(fixture_spans), _cfg(tmp_path), FeaturePatterns(), RawSchema())
    injected = inject_anomalies(
        spans_f, InjectionConfig(sem_cols=(), text_col=None), embedder=None)
    got = dict(injected.group_by('trace_id')
               .agg(pl.col('anomaly_type').drop_nulls().first())
               .sort('trace_id').iter_rows())
    assert got == golden['injection_labels']


@pytest.mark.characterization
def test_split_membership_and_normalization_pinned(fixture_spans, golden, tmp_path):
    from ars.data.anomalies_injection import InjectionConfig, inject_anomalies
    from ars.stages.s1__data import (
        build_traces, calculate_features, normalize_epi_features, stratified_split)
    cfg = _cfg(tmp_path)
    spans_f, _ = calculate_features(
        _with_label_stubs(fixture_spans), cfg, FeaturePatterns(), RawSchema())
    injected = inject_anomalies(
        spans_f, InjectionConfig(sem_cols=(), text_col=None), embedder=None)
    injected = injected.with_columns(
        (pl.col(DataObject.sublabel) != 'NonAnomaly').cast(pl.Int8).alias(DataObject.is_anomaly))
    traces = build_traces(injected, (), cfg)
    train, val, test = stratified_split(traces, DataObject.sublabel, cfg)
    got = {n: sorted(d['trace_id'].to_list())
           for n, d in (('train', train), ('val', val), ('test', test))}
    assert got == golden['split_membership']

    _, params = normalize_epi_features(train, val, test, cfg)
    assert params['method'] == golden['normalization']['method']
    for got_v, exp_v in zip(params['shift'], golden['normalization']['shift'], strict=True):
        assert got_v == pytest.approx(exp_v, rel=1e-6, abs=1e-9)
    for got_v, exp_v in zip(params['scale'], golden['normalization']['scale'], strict=True):
        assert got_v == pytest.approx(exp_v, rel=1e-6, abs=1e-9)


@pytest.mark.characterization_bug  # F-05: quantile-degenerate feature scale passes the 1e-6 guard
def test_min_normalization_scale_is_degenerate(golden):
    assert golden['normalization_min_scale'] < 0.01
    assert golden['normalization_min_scale'] > 1e-6  # passes the too-small guard
