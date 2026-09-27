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


# ------------------------------------------------ explain(): the export surface

@pytest.fixture(scope='module')
def huber_branch():
    """Same setup with a Huber loss: the export must decompose THAT loss."""
    key = jax.random.PRNGKey(SEED + 1)
    xs = jax.random.normal(key, (16, 6, 4), dtype=jp.float32) * 0.1
    mask = jp.ones((16, 6), dtype=bool)
    hp = HyperParamsLSTMAE(sz_features=4, sz_latent=3, layers_arch=(('unidirectional', 8),),
                           loss_type='huber', huber_delta=0.5)
    model = LSTM_AE(hp)
    state, _ = Trainer.train_lstm_ae(
        model=model, train_padded=xs, train_mask=mask,
        val_padded=xs[:4], val_mask=mask[:4],
        learning_rate=3e-3, num_epochs=10, batch_size=8, rng=key,
        input_shape=(6, 4), weight_decay=0.0, clip_grad=1.0,
        schedule_fn=None, patience=50, target_loss=None)
    return model, state, xs, mask


@pytest.mark.parametrize('branch', ['trained_branch', 'huber_branch'])
def test_explain_decomposes_the_branch_error_exactly(branch, request):
    """Span errors sum to the detector's own branch error (Calibrate.lstm_branch),
    so shares are shares of the number that drove p_anomaly."""
    import numpy as np
    from ars.models.m2__detector.attribution import explain
    from ars.models.m2__detector.confidence import Calibrate
    model, state, xs, mask = request.getfixturevalue(branch)
    mask = mask.at[:5, 4:].set(False)
    out = explain(model, state.params, xs.at[:, 1, 2].add(3.0), mask, k_spans=6)
    ref, _ = Calibrate.lstm_branch(model, state.params, xs.at[:, 1, 2].add(3.0), mask)
    np.testing.assert_allclose(out['error'], np.asarray(ref), rtol=1e-5, atol=1e-7)
    finite = np.where(np.isfinite(out['span_share']), out['span_share'], 0.0)
    np.testing.assert_allclose(finite.sum(axis=1), 1.0, rtol=1e-4)


def test_explain_never_ranks_padding(trained_branch):
    import numpy as np
    from ars.models.m2__detector.attribution import explain
    model, state, xs, _ = trained_branch
    mask = jp.asarray([[True] * 2 + [False] * 4] * 16)
    out = explain(model, state.params, xs, mask, k_spans=5)
    valid = np.isfinite(out['span_err'])
    assert valid.sum(axis=1).tolist() == [2] * 16          # only the 2 real spans score
    assert set(out['span_idx'][valid].tolist()) <= {0, 1}


def test_explain_localizes_with_values_at_the_peak(trained_branch):
    """A poisoned (span 3, feature 2) cell: top span 3, top feature 2 peaking
    at span 3 with the poisoned value observed vs a near-normal expectation,
    and feature 2 driving span 3."""
    import numpy as np
    from ars.models.m2__detector.attribution import explain
    model, state, xs, mask = trained_branch
    poisoned = xs.at[:, 3, 2].add(5.0)
    out = explain(model, state.params, poisoned, mask, k_spans=3, k_feats=2, k_drivers=2)
    assert (out['span_idx'][:, 0] == 3).mean() > 0.8
    assert (out['feat_idx'][:, 0] == 2).mean() > 0.8
    hit = out['feat_idx'][:, 0] == 2
    assert (out['feat_peak'][hit, 0] == 3).all()
    np.testing.assert_allclose(out['feat_obs'][hit, 0], np.asarray(poisoned[hit, 3, 2]), rtol=1e-6)
    assert (np.abs(out['feat_exp'][hit, 0]) < 2.5).all()   # the model expects the normal level
    assert (out['drv_idx'][out['span_idx'][:, 0] == 3, 0, 0] == 2).mean() > 0.8


def test_explain_driver_mask_and_chunking(trained_branch):
    import numpy as np
    from ars.models.m2__detector.attribution import explain
    model, state, xs, mask = trained_branch
    poisoned = xs.at[:, 3, 2].add(5.0)
    eligible = np.asarray([True, True, False, True])
    out = explain(model, state.params, poisoned, mask, 3, 2, 3, driver_mask=eligible)
    assert 2 not in set(out['drv_idx'][np.isfinite(out['drv_err'])].tolist())
    small = explain(model, state.params, poisoned, mask, 3, 2, 3, driver_mask=eligible, chunk=5)
    for key in out:
        np.testing.assert_allclose(small[key], out[key], rtol=1e-5)
    none = explain(model, state.params, poisoned, mask, 3, 0, 2, driver_mask=np.zeros(4, dtype=bool))
    assert not np.isfinite(none['drv_err']).any()          # nothing eligible: no drivers


def test_combined_epi_share_splits_the_flagging_error():
    import numpy as np
    import optax
    from ars.models.m2__detector.architecture import FMLP_AE, HyperParamsFMLPAE, TrainState
    from ars.models.m2__detector.attribution import combined_epi_share
    model = FMLP_AE(HyperParamsFMLPAE(sz_latent_epi=3, sz_latent_sem=2, layers_arch=(('relu', 4),)))
    key = jax.random.PRNGKey(SEED)
    epi, sem = jax.random.normal(key, (6, 3)), jax.random.normal(jax.random.PRNGKey(1), (6, 2))
    params = model.init(key, epi, sem, False)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.identity(), batch_stats={})
    share = combined_epi_share(state, model, epi, sem)
    recon = np.asarray(model.apply({'params': params}, epi, sem, False))
    pw = (recon - np.concatenate((epi, sem), axis=1)) ** 2
    np.testing.assert_allclose(share, pw[:, :3].sum(1) / pw.sum(1), rtol=1e-5)
    assert ((share >= 0) & (share <= 1)).all()
