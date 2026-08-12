"""Characterization: injector determinism + the u64 hash wraparound crash (F-36)."""
import polars as pl
import pytest

from ars.data.anomalies_injection import TextNoise


def _text_frame(n: int) -> pl.DataFrame:
    return pl.DataFrame({
        'sem_text': [f'span output text number {i} with some payload' for i in range(n)],
        'anomaly_severity': [0.6] * n,
    })


def test_text_noise_handles_large_frames():
    """F-36 FIXED: hash%length no longer wraps negative; 500-row frames corrupt
    cleanly (before the fix this raised InvalidOperationError on ~5% of rows)."""
    df = _text_frame(500)
    expr = TextNoise.corrupt_col(
        'sem_text', 'anomaly_severity', seed=12345,
        fractions={'chars': 0.25, 'loop': 0.25, 'foreign': 0.25, 'mojibake': 0.25})
    out = df.with_columns(expr)
    assert out.height == 500
    changed = (out['sem_text'] != df['sem_text']).sum()
    assert changed == 500  # every row was corrupted (all rows are victims here)


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
