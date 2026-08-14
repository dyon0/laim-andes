"""F-01 / F-10 tests: real semantic embeddings + embedder fingerprint pinning."""
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

STANDIN = Path('/tmp/models/USER-bge-m3-standin')


@pytest.fixture(scope='session')
def standin_embedder() -> Path:
    if not (STANDIN / 'config.json').exists() and not (STANDIN / 'modules.json').exists():
        r = subprocess.run(
            [sys.executable, 'baseline/build_standin_embedder.py', str(STANDIN)],
            capture_output=True, text=True, cwd=Path(__file__).parents[1])
        assert r.returncode == 0, r.stderr[-2000:]
    return STANDIN


@pytest.mark.slow
def test_span_embeddings_are_text_dependent(standin_embedder, fixture_spans, tmp_path):
    """F-01 FIXED: normal spans get REAL embeddings (legacy: constant zeros for
    every span, which made the semantic branch an injection oracle)."""
    from ars.configuration.c1__data import S1Config
    from ars.stages.s1__data import compute_semantic_embeddings
    cfg = S1Config(input_parquet_files=(), output_dir=tmp_path, output_prefix='t',
                   embedder_path=standin_embedder)
    df = fixture_spans.head(6).with_columns(
        pl.col('output_text').alias('sem_text'))
    out = compute_semantic_embeddings(df, cfg, 'sem_text', 'sem_vector')
    vecs = out['sem_vector'].to_list()
    assert len(vecs[0]) == 1024
    norms = [sum(v * v for v in vec) ** 0.5 for vec in vecs]
    assert all(n == pytest.approx(1.0, abs=1e-3) for n in norms)   # unit-norm, not zeros
    texts = df['sem_text'].to_list()
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i] != texts[j]:
                assert vecs[i] != vecs[j], (i, j)
            else:
                assert vecs[i] == pytest.approx(vecs[j])


@pytest.mark.slow
def test_duplicate_texts_get_identical_vectors(standin_embedder, tmp_path):
    """The dedup fast path must map repeated texts to bit-identical vectors
    and preserve row order."""
    from ars.configuration.c1__data import S1Config
    from ars.stages.s1__data import compute_semantic_embeddings
    cfg = S1Config(input_parquet_files=(), output_dir=tmp_path, output_prefix='t',
                   embedder_path=standin_embedder)
    df = pl.DataFrame({'sem_text': ['alpha', 'beta', 'alpha', 'gamma', 'beta', 'alpha']})
    out = compute_semantic_embeddings(df, cfg, 'sem_text', 'sem_vector')
    v = out['sem_vector'].to_list()
    assert v[0] == v[2] == v[5]      # all 'alpha' rows identical
    assert v[1] == v[4]              # both 'beta' rows identical
    assert v[0] != v[1] != v[3]      # distinct texts differ
    assert out.height == 6           # row order and count preserved


def test_embedding_device_plan():
    """Multi-GPU plan: 0 = all visible, N = first min(N, visible); CPU and
    single-GPU hosts always take the ordinary in-process path."""
    from ars.configuration.c1__data import S1Config
    from ars.stages.s1__data import embedding_device_plan

    def cfg(device, gpus=0):
        return S1Config(input_parquet_files=(), output_dir=Path('.'),
                        output_prefix='t', device=device, embedding_gpus=gpus)

    assert embedding_device_plan(cfg('cpu'), 8) == ('cpu',)
    assert embedding_device_plan(cfg('cuda'), 0) == ('cuda',)
    assert embedding_device_plan(cfg('cuda'), 1) == ('cuda',)
    assert embedding_device_plan(cfg('cuda', 0), 8) == tuple(f'cuda:{i}' for i in range(8))
    assert embedding_device_plan(cfg('cuda', 3), 8) == ('cuda:0', 'cuda:1', 'cuda:2')
    assert embedding_device_plan(cfg('cuda', 1), 8) == ('cuda',)
    assert embedding_device_plan(cfg('cuda', 5), 2) == ('cuda:0', 'cuda:1')   # capped


def test_balanced_slices_are_deterministic_and_cover():
    from ars.stages.s1__data import _balanced_slices
    assert _balanced_slices(10, 3) == ((0, 4), (4, 7), (7, 10))
    assert _balanced_slices(2, 8) == ((0, 1), (1, 2))     # never empty slices
    assert _balanced_slices(8, 8) == tuple((i, i + 1) for i in range(8))
    assert _balanced_slices(0, 4) == ()                   # no empty ranges, ever
    for n, k in ((41, 2), (100, 8), (7, 7), (1, 4)):
        b = _balanced_slices(n, k)
        assert b[0][0] == 0 and b[-1][1] == n
        assert all(x[1] == y[0] for x, y in zip(b, b[1:]))   # contiguous
        assert all(hi > lo for lo, hi in b)                  # non-empty


@pytest.mark.slow
def test_multi_device_encoding_matches_single(standin_embedder, tmp_path, monkeypatch):
    """The multi-device engine (per-device model replicas + threads — NEVER
    multiprocessing: the SberDS wrapper is not spawn-safe, a spawned worker
    re-executes the whole node) must produce the same vectors as one-replica
    encoding. Runs the real engine on two CPU replicas."""
    import numpy as np

    from ars.configuration.c1__data import S1Config
    from ars.stages import s1__data

    cfg = S1Config(input_parquet_files=(), output_dir=tmp_path, output_prefix='t',
                   embedder_path=standin_embedder, embedding_pool_chunk=7)
    texts = tuple(f'span text number {i} with payload {i * i}' for i in range(41))

    direct = np.asarray(s1__data.make_embedder(cfg)(texts))

    monkeypatch.setattr(s1__data, 'embedding_device_plan',
                        lambda _cfg, _n: ('cpu', 'cpu'))
    multi_embed = s1__data.make_embedder(cfg)
    assert multi_embed.progress_chunk == 7 * 2   # per-replica chunk x replicas
    multi = np.asarray(multi_embed(texts))
    small = np.asarray(multi_embed(texts[:3]))   # small-call path: one replica
    empty = np.asarray(multi_embed(()))          # empty input: (0, dim), no crash
    assert empty.shape == (0, 1024)
    multi_embed.close()
    multi_embed.close()   # idempotent

    assert multi.shape == direct.shape == (41, 1024)
    np.testing.assert_allclose(multi, direct, atol=1e-5)
    np.testing.assert_allclose(small, direct[:3], atol=1e-5)


def test_fingerprint_detects_model_swap(tmp_path):
    """F-10 FIXED: loading artifacts against a different embedder raises."""
    from ars.tools.utilities.fingerprint import model_fingerprint, verify_fingerprint
    a = tmp_path / 'model_a'; a.mkdir()
    (a / 'config.json').write_text('{"hidden": 128}')
    b = tmp_path / 'model_b'; b.mkdir()
    (b / 'config.json').write_text('{"hidden": 256}')

    fp_a = model_fingerprint(a)
    verify_fingerprint(a, fp_a)            # match: fine
    verify_fingerprint(a, None)            # legacy artifacts: skipped
    with pytest.raises(ValueError, match='отпечаток'):
        verify_fingerprint(b, fp_a)        # swap: refused


def test_fingerprint_is_stable(tmp_path):
    from ars.tools.utilities.fingerprint import model_fingerprint
    m = tmp_path / 'm'; m.mkdir()
    (m / 'config.json').write_text('{"x": 1}')
    (m / 'weights.bin').write_bytes(b'\x00' * 128)
    assert model_fingerprint(m) == model_fingerprint(m)
