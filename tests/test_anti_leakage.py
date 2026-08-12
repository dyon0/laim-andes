"""Anti-leakage tests — the failure modes that silently destroy the project.

These assert structural properties of the pipeline, not numbers.
"""
import polars as pl
import pytest

from ars.configuration.c1__data import S1Config
from ars.data.features import FeaturePatterns, RawSchema
from ars.specification.spec import DataObject

SEED = 12345


def _cfg(tmp_path) -> S1Config:
    return S1Config(
        input_parquet_files=(), output_dir=tmp_path, output_prefix='leak',
        seed_random=SEED, seed_polars=SEED, seed_torch=SEED,
        seed_split=SEED, seed_synth=SEED, seed_llm=SEED)


def _stub_labels(spans: pl.DataFrame) -> pl.DataFrame:
    return spans.with_columns(
        pl.lit(DataObject.class_sentinel).alias(DataObject.label),
        pl.lit(DataObject.class_sentinel).alias(DataObject.sublabel),
        pl.lit(None, dtype=pl.Int8).alias(DataObject.is_anomaly))


def test_no_trace_id_spans_multiple_splits(fixture_spans, tmp_path):
    """F-08: every trace_id must live in exactly one of train/val/test."""
    from ars.data.anomalies_injection import InjectionConfig, inject_anomalies
    from ars.stages.s1__data import build_traces, calculate_features, stratified_split
    cfg = _cfg(tmp_path)
    spans_f, _ = calculate_features(
        _stub_labels(fixture_spans), cfg, FeaturePatterns(), RawSchema())
    injected = inject_anomalies(
        spans_f, InjectionConfig(sem_cols=(), text_col=None), embedder=None)
    injected = injected.with_columns(
        (pl.col(DataObject.sublabel) != 'NonAnomaly').cast(pl.Int8).alias(DataObject.is_anomaly))
    traces = build_traces(injected, (), cfg)
    train, val, test = stratified_split(traces, DataObject.sublabel, cfg)
    sets = {n: set(d['trace_id'].to_list()) for n, d in
            (('train', train), ('val', val), ('test', test))}
    assert not sets['train'] & sets['val']
    assert not sets['train'] & sets['test']
    assert not sets['val'] & sets['test']


def test_train_contains_no_anomalous_traces(fixture_spans, tmp_path):
    """Anomalous (injected) traces must never reach the reconstruction train set."""
    from ars.data.anomalies_injection import InjectionConfig, inject_anomalies
    from ars.stages.s1__data import build_traces, calculate_features, stratified_split
    cfg = _cfg(tmp_path)
    spans_f, _ = calculate_features(
        _stub_labels(fixture_spans), cfg, FeaturePatterns(), RawSchema())
    injected = inject_anomalies(
        spans_f, InjectionConfig(sem_cols=(), text_col=None), embedder=None)
    injected = injected.with_columns(
        (pl.col(DataObject.sublabel) != 'NonAnomaly').cast(pl.Int8).alias(DataObject.is_anomaly))
    traces = build_traces(injected, (), cfg)
    train, _, _ = stratified_split(traces, DataObject.sublabel, cfg)
    assert train.filter(pl.col(DataObject.is_anomaly) == 1).height == 0


def test_normalization_params_derive_only_from_train(fixture_spans, tmp_path):
    """Perturbing val/test features must not move the fitted shift/scale."""
    from ars.data.anomalies_injection import InjectionConfig, inject_anomalies
    from ars.stages.s1__data import (
        build_traces, calculate_features, normalize_epi_features, stratified_split)
    cfg = _cfg(tmp_path)
    spans_f, _ = calculate_features(
        _stub_labels(fixture_spans), cfg, FeaturePatterns(), RawSchema())
    injected = inject_anomalies(
        spans_f, InjectionConfig(sem_cols=(), text_col=None), embedder=None)
    injected = injected.with_columns(
        (pl.col(DataObject.sublabel) != 'NonAnomaly').cast(pl.Int8).alias(DataObject.is_anomaly))
    traces = build_traces(injected, (), cfg)
    train, val, test = stratified_split(traces, DataObject.sublabel, cfg)

    _, params_a = normalize_epi_features(train, val, test, cfg)
    poisoned_val = val.with_columns(
        pl.col('epi_sequence').list.eval(
            pl.element().list.eval(pl.element() * 1000.0)))
    _, params_b = normalize_epi_features(train, poisoned_val, test, cfg)
    assert params_a['shift'] == params_b['shift']
    assert params_a['scale'] == params_b['scale']


def test_feature_selection_ignores_non_train_traces(fixture_spans, tmp_path):
    """F-11: selection statistics are fit on the supplied train-normal ids only —
    corrupting every other trace's numbers must not change the selected set."""
    from ars.stages.s1__data import calculate_features
    import polars.selectors as ps
    cfg = _cfg(tmp_path)
    spans = _stub_labels(fixture_spans)
    ids = spans.select('trace_id').unique().sort('trace_id')
    train_ids = ids.head(10)
    other_ids = ids.tail(ids.height - 10)

    _, names_a = calculate_features(
        spans, cfg, FeaturePatterns(), RawSchema(), selection_ids=train_ids)

    # corrupt the numeric raw inputs of the non-train traces
    corrupt = spans.with_columns(
        pl.when(pl.col('trace_id').is_in(other_ids['trace_id'].implode()))
          .then(pl.col('llm_total_tokens') * 9999 + 12345)
          .otherwise(pl.col('llm_total_tokens'))
          .alias('llm_total_tokens'))
    _, names_b = calculate_features(
        corrupt, cfg, FeaturePatterns(), RawSchema(), selection_ids=train_ids)
    assert list(names_a) == list(names_b)


def test_threshold_selection_never_sees_test():
    """F-02 (structural half): select_threshold consumes only what it is given;
    the pipeline passes VAL errors. This pins the call signature usage."""
    import inspect
    from ars.stages import s2__detector
    src = inspect.getsource(s2__detector.run_experiment)
    assert 'select_threshold(\n        errors = val_errors' in src or \
           'select_threshold(errors = val_errors' in src.replace('\n', ' ') or \
           'val_errors' in src.split('select_threshold')[1][:120]
