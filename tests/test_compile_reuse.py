"""Compile reuse. XLA compiles a jitted function per input SHAPE, and on GPU
every compile costs tens of ms (one op) to seconds (a forward pass): a small
run is mostly compilation. A forward must compile once per run, not once per
split size (values equal up to float32 rounding), and model init must be one
program, not one per op (values bit-identical)."""
import jax
import jax.numpy as jp
import numpy as np
import pytest

from ars.models.m2__detector.architecture import (
    FMLP_AE, LSTM_AE, TRAIN, Branch, HyperParamsFMLPAE, HyperParamsLSTMAE,
    block_rows, over_blocks)
from ars.models.m2__detector.confidence import Calibrate

T, D = 7, 5
COMPILE = '/jax/core/compile/backend_compile_duration'


def _lstm(latent: int = 4) -> LSTM_AE:
    return LSTM_AE(HyperParamsLSTMAE(
        sz_features=D, sz_latent=latent, dropout_rate=0.1,
        layers_arch=(('bidirectional', 6), ('unidirectional', 5))))


def _fmlp() -> FMLP_AE:
    return FMLP_AE(HyperParamsFMLPAE(
        sz_latent_epi=4, sz_latent_sem=3, layers_arch=(('relu', 8), ('tanh', 2)),
        dropout_rate=0.1, use_batch_norm=True))


def _batch(n: int, seed: int) -> tuple[jax.Array, jax.Array]:
    key = jax.random.PRNGKey(seed)
    lens = jax.random.randint(jax.random.fold_in(key, 1), (n,), 1, T + 1)
    mask = jp.arange(T)[None, :] < lens[:, None]
    return jax.random.normal(key, (n, T, D), dtype=jp.float32) * mask[..., None], mask


def _same(a, b) -> None:
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _close(a, b) -> None:
    # XLA picks a matmul kernel by shape, so a row computed in a block of a
    # different size can differ in the last float32 bits (never more)
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=2e-5, atol=1e-6)


class _Compiles:
    def __enter__(self) -> '_Compiles':
        self.n = 0

        def on(event: str, duration: float, **_) -> None:
            if event == COMPILE:
                self.n += 1
        self._on = on
        jax.monitoring.register_event_duration_secs_listener(on)
        return self

    def __exit__(self, *exc) -> None:
        jax.monitoring.unregister_event_duration_listener(self._on)


def test_block_rows_buckets():
    assert [block_rows(n) for n in (0, 1, 3, 14, 27, 32, 68, 5000)] == [1, 1, 4, 16, 32, 32, 128, 1024]
    assert block_rows(68, cap=64) == 64


@pytest.mark.parametrize('model, dummies', [
    (_lstm(), (jp.ones((1, T, D), jp.float32), jp.full((1,), T, jp.int32))),
    (_fmlp(), (jp.ones((1, 4)), jp.ones((1, 3))))])
def test_train_state_is_one_program_with_the_same_values(model, dummies):
    rng = jax.random.PRNGKey(3)
    eager = model.init(rng, *dummies, training=True)        # op by op, as before
    with _Compiles() as c:
        state = TRAIN.make_train_state(rng, model, 1e-3, input_shape=(T, D) if isinstance(model, LSTM_AE) else None)
    assert c.n == 1
    _same(eager['params'], state.params)
    _same(eager.get('batch_stats', {}), state.batch_stats)


@pytest.mark.parametrize('n, rows', [(11, 4), (3, 8), (8, 8)])
def test_blocks_give_the_values_of_one_call(n, rows):
    model = _lstm()
    state = TRAIN.make_train_state(jax.random.PRNGKey(0), model, 1e-3, input_shape=(T, D))
    x, mask = _batch(n, n)
    lens = jp.sum(mask, axis=1).astype(jp.int32)
    _close(Branch.encode(model, state, x, mask, rows=rows),
           TRAIN.lstm_ae_encode_batch(state.params, model, x, lens))
    _close(over_blocks(lambda a, m: Calibrate.lstm_branch(model, state.params, a, m), rows, x, mask),
           Calibrate.lstm_branch(model, state.params, x, mask))
    sse, cnt = TRAIN.lstm_ae_sse_count(state.params, model, x, lens, mask)
    _close(Branch.lstm_recon_mse(model, state, x, mask, rows=rows), sse / cnt)
    cmb = TRAIN.make_train_state(jax.random.PRNGKey(1), _fmlp(), 1e-3)
    e, s = jax.random.normal(jax.random.PRNGKey(2), (n, 4)), jax.random.normal(jax.random.PRNGKey(3), (n, 3))
    _close(Branch.combined_forward(cmb, _fmlp(), e, s, rows),
           TRAIN.fmlp_ae_forward(cmb, _fmlp(), e, s, training=False))


def test_splits_of_a_run_share_one_compiled_forward():
    model = _lstm(latent=3)                  # a model no other test compiles
    state = TRAIN.make_train_state(jax.random.PRNGKey(0), model, 1e-3, input_shape=(T, D))
    before = TRAIN.lstm_ae_encode_batch._cache_size()
    rows = block_rows(13)
    for n in (13, 5, 9):                     # train / val / test of one run
        Branch.encode(model, state, *_batch(n, n), rows=rows)
    assert TRAIN.lstm_ae_encode_batch._cache_size() == before + 1
