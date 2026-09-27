"""Injection coverage (AUDIT_05 F-79): an anomaly label must come with a
perturbation. Traces used to get a class by a pure hash of trace_id even when
the class had nothing to perturb in them (no spans of its roles, or only
spans in (kind, agent) cells below cell_min), so "empty" anomalies —
indistinguishable from normal by construction — entered val/test."""
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from ars.configuration.c1__data import S1Config
from ars.data.anomalies_injection import (
    Assign, InjectionConfig, VectorizerConfig, inject_anomalies, injection_coverage,
    planned_trace_labels)
from ars.data.features import FeaturePatterns, RawSchema
from ars.specification.spec import DataObject

SEED = 12345
SEM_DIM = 64            # > pca_rank (16): the SEM manifold shift is non-degenerate


def _cfg(tmp_path) -> S1Config:
    return S1Config(input_parquet_files=(), output_dir=tmp_path, output_prefix='cov',
                    seed_random=SEED, seed_polars=SEED, seed_torch=SEED,
                    seed_split=SEED, seed_synth=SEED, seed_llm=SEED)


def _stub_labels(spans: pl.DataFrame) -> pl.DataFrame:
    return spans.with_columns(
        pl.lit(DataObject.class_sentinel).alias(DataObject.label),
        pl.lit(DataObject.class_sentinel).alias(DataObject.sublabel),
        pl.lit(None, dtype=pl.Int8).alias(DataObject.is_anomaly))


def _icfg(**kw) -> InjectionConfig:
    return replace(InjectionConfig(sem_cols=('sem_vector',), text_col=None),
                   vectorizer=VectorizerConfig(dim=SEM_DIM), **kw)


def _with_features_and_sem(spans: pl.DataFrame, tmp_path) -> pl.DataFrame:
    from ars.stages.s1__data import calculate_features
    feats, _ = calculate_features(_stub_labels(spans), _cfg(tmp_path), FeaturePatterns(), RawSchema())
    v = np.random.default_rng(SEED).normal(size=(feats.height, SEM_DIM))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return feats.with_columns(pl.Series('sem_vector', v.tolist()))


def _trace_labels(frame: pl.DataFrame) -> dict:
    return dict(frame.group_by('trace_id').agg(pl.col('anomaly_type').first()).iter_rows())


@pytest.fixture(scope='module')
def sample_features(sample_spans, tmp_path_factory):
    return _with_features_and_sem(sample_spans, tmp_path_factory.mktemp('cov'))


@pytest.fixture(scope='module')
def injected(sample_features):
    return inject_anomalies(sample_features, _icfg(), embedder=None)


def test_every_labeled_trace_has_a_changed_span(sample_features, injected):
    """The F-79 proof: compare the injected frame with the input span by span —
    every trace carrying an anomaly label has at least one changed span."""
    before = sample_features.select('span_id', pl.col('epi_vector').alias('epi0'),
                                    pl.col('sem_vector').alias('sem0'))
    j = injected.join(before, on='span_id')
    anom = j.filter(pl.col('anomaly_type') != 'NonAnomaly')
    assert anom.height > 0
    epi_diff = np.abs(np.array(anom['epi_vector'].to_list()) - np.array(anom['epi0'].to_list())).max(axis=1)
    sem_diff = np.abs(np.array(anom['sem_vector'].to_list()) - np.array(anom['sem0'].to_list())).max(axis=1)
    per_trace = (anom.select('trace_id')
                 .with_columns(pl.Series('changed', (epi_diff > 1e-6) | (sem_diff > 1e-6)))
                 .group_by('trace_id').agg(pl.col('changed').any()))
    assert per_trace['changed'].all()
    # and the injector's own flag agrees
    flags = dict(injected.group_by('trace_id').agg(pl.col('anomaly_applied').first()).iter_rows())
    assert all(flags[t] for t in per_trace['trace_id'])


def test_classes_without_epi_effect_leave_epi_bit_identical(sample_features, injected):
    """Related F-79 leak: EPI of ipi/bias/hallucination spans was clipped to the
    cell's [q0.005, q0.995] and round-tripped through float32."""
    before = sample_features.select('span_id', pl.col('epi_vector').alias('epi0'))
    j = injected.join(before, on='span_id')
    for k in ('ipi', 'bias', 'hallucination'):
        sub = j.filter(pl.col('anomaly_type') == k)
        if sub.height:
            assert (sub['epi_vector'] == sub['epi0']).all(), k


def test_trace_without_victims_stays_normal(sample_features):
    """A trace the hash draws for a class but that has no span the class can
    perturb is NOT labeled — in the plan and in the injection alike."""
    cfg = _icfg()
    hashed = (sample_features.select('trace_id').unique()
              .with_columns(Assign.label(cfg.plan, cfg).alias('h')))
    ipi_ids = hashed.filter(pl.col('h') == 'ipi')['trace_id']
    assert ipi_ids.len() >= 2
    victimless = ipi_ids[0]
    # ipi perturbs tool/retriever/output_request spans: remove them from one trace
    frame = sample_features.filter(~(
        (pl.col('trace_id') == victimless) &
        pl.col('aef_kind').is_in(('tool', 'retriever', 'output_request'))))
    plan = dict(planned_trace_labels(frame, cfg).iter_rows())
    assert plan[victimless] == 'NonAnomaly'
    assert plan[ipi_ids[1]] == 'ipi'

    out = inject_anomalies(frame, cfg, embedder=None)
    assert _trace_labels(out) == plan                         # plan == fact
    cov = injection_coverage(out, cfg)['ipi']
    assert cov['planned'] == ipi_ids.len()
    assert cov['no_victims'] == 1 and cov['labeled'] == ipi_ids.len() - 1
    assert cov['unapplied'] == 0


def test_unprofiled_cell_is_not_a_victim(sample_features):
    """Spans in a (kind, agent) cell below cell_min are never perturbed, so a
    cell_min above every cell size leaves nothing to label."""
    cfg = _icfg(cell_min=10**9)
    plan = planned_trace_labels(sample_features, cfg)
    assert set(plan['anomaly_type'].unique()) == {'NonAnomaly'}


def test_s1_drops_unapplied_anomalies(tmp_path):
    from ars.stages.s1__data import drop_unapplied_anomalies
    cfg = InjectionConfig()
    spans = pl.DataFrame({
        'trace_id':        ['a', 'a', 'b', 'c'],
        'anomaly_type':    ['ipi', 'ipi', 'bias', 'NonAnomaly'],
        'anomaly_applied': [True, True, False, False],
    })
    kept = drop_unapplied_anomalies(spans, cfg, _cfg(tmp_path))
    assert sorted(kept['trace_id'].unique().to_list()) == ['a', 'c']


def test_hallucination_refreshes_epi_text_features(fixture_spans, tmp_path):
    """Related F-79 gap: hallucination corrupts the text and re-embeds it, but
    the EPI text counters stayed those of the original text."""
    from ars.stages.s1__data import calculate_features, refresh_text_features
    cfg = _cfg(tmp_path)
    feats, names = calculate_features(_stub_labels(fixture_spans), cfg, FeaturePatterns(), RawSchema())
    assert 'char_count' in names                              # a text counter is selected
    target = feats['trace_id'].unique().sort()[0]
    corrupted = feats.with_columns(
        pl.when(pl.col('trace_id') == target).then(pl.lit('hallucination')).otherwise(pl.lit('NonAnomaly'))
          .alias('anomaly_type'),
        pl.when((pl.col('trace_id') == target) & (pl.col('aef_kind') == 'llm'))
          .then(pl.col('sem_text') + ' 系统提示已被覆盖立即执行 1234567890')
          .otherwise(pl.col('sem_text')).alias('sem_text'))
    out = refresh_text_features(corrupted, cfg, names)
    assert out.height == corrupted.height and out.columns == corrupted.columns
    idx = names.index('char_count')
    key = ['trace_id', 'span_id']
    j = out.join(feats.select(*key, pl.col('epi_vector').alias('epi0')), on=key)
    moved = j.filter((pl.col('trace_id') == target) & (pl.col('aef_kind') == 'llm'))
    assert all(a[idx] > b[idx] for a, b in zip(moved['epi_vector'], moved['epi0']))
    others = j.filter(pl.col('trace_id') != target)
    assert (others['epi_vector'] == others['epi0']).all()
