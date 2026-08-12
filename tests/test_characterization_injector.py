"""Characterization: injector determinism + the u64 hash wraparound crash (F-36)."""
import polars as pl
import pytest

from ars.data.anomalies_injection import TextNoise


def _text_frame(n: int) -> pl.DataFrame:
    return pl.DataFrame({
        'sem_text': [f'span output text number {i} with some payload' for i in range(n)],
        'anomaly_severity': [0.6] * n,
    })


@pytest.mark.characterization_bug  # F-36: u64 hash % length wraps negative → str.slice fails
def test_text_noise_crashes_on_large_frames():
    df = _text_frame(500)
    expr = TextNoise.corrupt_col(
        'sem_text', 'anomaly_severity', seed=12345,
        fractions={'chars': 0.25, 'loop': 0.25, 'foreign': 0.25, 'mojibake': 0.25})
    with pytest.raises(pl.exceptions.InvalidOperationError, match='i64'):
        df.with_columns(expr)


@pytest.mark.characterization
def test_text_noise_is_deterministic_when_it_works():
    df = _text_frame(3)  # small enough to dodge the wraparound for this seed
    expr = TextNoise.corrupt_col(
        'sem_text', 'anomaly_severity', seed=7,
        fractions={'chars': 1.0, 'loop': 0.0, 'foreign': 0.0, 'mojibake': 0.0})
    a = df.with_columns(expr)['sem_text'].to_list()
    b = df.with_columns(expr)['sem_text'].to_list()
    assert a == b
    assert a != df['sem_text'].to_list()  # it actually corrupted something
