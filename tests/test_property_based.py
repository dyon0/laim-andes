"""Property-based tests (Hypothesis) — invariants over generated inputs."""
import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings

from laim.evaluation import binary_metrics, ece_brier, prevalence_adjusted_ppv, recall_at_fpr

finite_floats = st.floats(min_value=-1e6, max_value=1e6,
                          allow_nan=False, allow_infinity=False)


@settings(max_examples=50, deadline=None)
@given(st.lists(finite_floats, min_size=4, max_size=64),
       st.floats(min_value=0.01, max_value=0.5))
def test_recall_at_fpr_never_exceeds_target(scores, target):
    n = len(scores)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    if y.sum() == 0 or y.sum() == n:
        y[0], y[-1] = 0, 1
    res = recall_at_fpr(y, np.asarray(scores), target)
    assert res['fpr'] <= target + 1e-12
    assert 0.0 <= res['recall'] <= 1.0


@settings(max_examples=50, deadline=None)
@given(st.floats(min_value=0, max_value=1), st.floats(min_value=0, max_value=1),
       st.floats(min_value=1e-4, max_value=0.5))
def test_ppv_is_a_probability(recall, fpr, prev):
    v = prevalence_adjusted_ppv(recall, fpr, prev)
    assert np.isnan(v) or 0.0 <= v <= 1.0


@settings(max_examples=30, deadline=None)
@given(st.integers(min_value=2, max_value=128))
def test_ece_perfect_calibration_is_zero(n):
    rng = np.random.default_rng(n)
    p = rng.uniform(0, 1, 2048)
    y = (rng.uniform(0, 1, 2048) < p).astype(float)   # labels drawn from p
    assert ece_brier(y, p, bins=10)['ece'] < 0.08


@settings(max_examples=25, deadline=None)
@given(st.lists(st.floats(min_value=0.1, max_value=100, allow_nan=False),
                min_size=8, max_size=40))
def test_binary_metrics_confusion_adds_up(scores):
    scores = np.asarray(scores)
    y = (scores > np.median(scores)).astype(int)
    yhat = (scores > np.mean(scores)).astype(int)
    m = binary_metrics(y, yhat, scores)
    assert m['tp'] + m['fp'] + m['fn'] + m['tn'] == len(scores)


@settings(max_examples=20, deadline=None)
@given(st.lists(st.text(min_size=1, max_size=60), min_size=1, max_size=40),
       st.integers(min_value=0, max_value=2**31 - 1))
def test_text_noise_total_on_arbitrary_unicode(texts, seed):
    """F-36 property: corrupt_col never raises, for any unicode input and seed."""
    from ars.data.anomalies_injection import TextNoise
    df = pl.DataFrame({'sem_text': texts, 'anomaly_severity': [0.7] * len(texts)})
    expr = TextNoise.corrupt_col('sem_text', 'anomaly_severity', seed=seed,
                                 fractions={'chars': 0.25, 'loop': 0.25,
                                            'foreign': 0.25, 'mojibake': 0.25})
    out = df.with_columns(expr)
    assert out.height == len(texts)
    assert out['sem_text'].null_count() == 0


@settings(max_examples=15, deadline=None)
@given(st.integers(min_value=0, max_value=10_000))
def test_normalization_roundtrip_within_tolerance(seed):
    """Scaler invertibility: x -> (x-shift)/scale -> *scale+shift == x
    (within clip bounds)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 3, (50, 4))
    shift = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-2)
    z = np.clip((x - shift) / scale, -20, 20)
    back = z * scale + shift
    inside = np.abs((x - shift) / scale) <= 20
    assert np.allclose(back[inside], x[inside], rtol=1e-9, atol=1e-9)


def test_span_order_permutation_yields_identical_features(fixture_spans):
    """A permuted-but-equivalent span table gives identical features: the
    pipeline sorts by (agent, trace, start_time) before computing."""
    from ars.data.features import FeaturePatterns, FeaturesSpan
    from ars.specification.spec import DataObject
    stub = lambda d: d.with_columns(
        pl.lit('unknown').alias(DataObject.label),
        pl.lit('unknown').alias(DataObject.sublabel),
        pl.lit(None, dtype=pl.Int8).alias(DataObject.is_anomaly))
    a = FeaturesSpan().make_features(stub(fixture_spans), FeaturePatterns())
    permuted = fixture_spans.sample(fraction=1.0, shuffle=True, seed=99)
    b = FeaturesSpan().make_features(stub(permuted), FeaturePatterns())
    key = ['trace_id', 'span_id']
    num = [c for c, t in a.schema.items() if t.is_numeric()]
    a_s = a.sort(key).select(key + num)
    b_s = b.sort(key).select(key + num)
    assert a_s.equals(b_s)
