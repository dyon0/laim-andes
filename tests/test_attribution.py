"""RCA-seam tests (M11): attribution surfaces are correct and localizing."""
import jax
import jax.numpy as jp
import pytest

from ars.models.m2__detector.architecture import HyperParamsLSTMAE, LSTM_AE
from ars.models.m2__detector.attribution import (
    counterfactual, per_feature_errors, per_span_errors, top_k)
from ars.models.m2__detector.train import Trainer

SEED = 7


@pytest.fixture(scope='module')
def trained_branch():
    """Tiny EPI AE overfit on clean data, so corrupted inputs stand out."""
    key = jax.random.PRNGKey(SEED)
    xs = jax.random.normal(key, (16, 6, 4), dtype=jp.float32) * 0.1
    mask = jp.ones((16, 6), dtype=bool)
    hp = HyperParamsLSTMAE(sz_features=4, sz_latent=3,
                           layers_arch=(('unidirectional', 8),))
    model = LSTM_AE(hp)
    state, _ = Trainer.train_lstm_ae(
        model=model, train_padded=xs, train_mask=mask,
        val_padded=xs[:4], val_mask=mask[:4],
        learning_rate=3e-3, num_epochs=30, batch_size=8, rng=key,
        input_shape=(6, 4), weight_decay=0.0, clip_grad=1.0,
        schedule_fn=None, patience=50, target_loss=None)
    return model, state, xs, mask


def test_per_span_errors_localize_corrupted_span(trained_branch):
    model, state, xs, mask = trained_branch
    corrupted = xs.at[:, 3, :].add(5.0)   # poison timestep 3 of every trace
    err = per_span_errors(model, state.params, corrupted, mask)
    assert err.shape == (16, 6)
    assert bool((jp.argmax(err, axis=1) == 3).mean() > 0.8)


def test_per_span_errors_zero_on_padding(trained_branch):
    model, state, xs, _ = trained_branch
    mask = jp.asarray([[True] * 3 + [False] * 3] * 16)
    err = per_span_errors(model, state.params, xs, mask)
    assert float(jp.abs(err[:, 3:]).max()) == 0.0


def test_per_feature_errors_localize_corrupted_feature(trained_branch):
    model, state, xs, mask = trained_branch
    corrupted = xs.at[:, :, 2].add(5.0)   # poison feature 2 everywhere
    err = per_feature_errors(model, state.params, corrupted, mask)
    assert err.shape == (16, 4)
    assert bool((jp.argmax(err, axis=1) == 2).mean() > 0.8)


def test_top_k_orders_descending():
    idx, vals = top_k(jp.asarray([[0.1, 3.0, 0.5, 2.0]]), k=2)
    assert idx[0].tolist() == [1, 3]
    assert vals[0].tolist() == pytest.approx([3.0, 2.0])


def test_counterfactual_reports_negative_delta_for_healing_edit(trained_branch):
    model, state, xs, mask = trained_branch
    corrupted = xs.at[:, :, 2].add(5.0)
    heal = lambda arr: arr.at[:, :, 2].add(-5.0)   # undo the corruption
    delta = counterfactual(model, state.params, corrupted, mask, heal)
    assert delta.shape == (16,)
    assert bool((delta < 0).all())   # healing reduces the error
