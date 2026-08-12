"""Characterization: pin s2 numerics (micro-train, threshold sweep, calibration).

Includes pins of KNOWN-BUGGY numerics (marked) that must flip when fixed.
"""
import jax
import jax.numpy as jp
import pytest

from ars.models.m2__detector.architecture import HyperParamsLSTMAE, LSTM_AE
from ars.models.m2__detector.confidence import Calibrate
from ars.models.m2__detector.train import Trainer
from ars.models.metrics import Loss
from ars.stages.s2__detector import Pad, select_threshold

SEED = 12345


@pytest.mark.characterization
@pytest.mark.slow
def test_micro_train_losses_pinned(golden):
    key = jax.random.PRNGKey(SEED)
    xs = jax.random.normal(key, (12, 6, 5), dtype=jp.float32)
    mask = jp.ones((12, 6), dtype=bool)
    hp = HyperParamsLSTMAE(sz_features=5, sz_latent=4,
                           layers_arch=(('unidirectional', 8),))
    _, losses = Trainer.train_lstm_ae(
        model=LSTM_AE(hp), train_padded=xs, train_mask=mask,
        val_padded=xs[:4], val_mask=mask[:4],
        learning_rate=1e-3, num_epochs=3, batch_size=4, rng=key,
        input_shape=(6, 5), weight_decay=0.0, clip_grad=1.0,
        schedule_fn=None, patience=10, target_loss=None)
    got = [round(float(l), 7) for l in losses]
    assert got == pytest.approx(golden['micro_train_losses'], rel=1e-5)


@pytest.mark.characterization
def test_threshold_sweep_pinned(golden):
    errors = jp.asarray([0.1, 0.2, 0.3, 0.9, 1.1, 1.3], dtype=jp.float32)
    labels = jp.asarray([0, 0, 0, 1, 1, 1], dtype=jp.int32)
    thr, metrics = select_threshold(errors, labels, 101, 'youden', 1e-8)
    assert float(thr) == pytest.approx(golden['micro_threshold'], abs=1e-6)
    assert float(metrics.youden) == pytest.approx(golden['micro_threshold_youden'], abs=1e-6)


def test_masked_loss_is_per_element():
    """F-23 FIXED: unit error gives loss 1.0 regardless of feature dim (was D)."""
    recon = jp.zeros((1, 2, 3))
    target = jp.ones((1, 2, 3))
    mask = jp.ones((1, 2), dtype=bool)
    assert float(Loss.masked(recon, target, mask, 'mse', 1.0, 1e-8)) == pytest.approx(1.0)
    # and masking works: padded steps contribute nothing
    mask2 = jp.asarray([[True, False]])
    assert float(Loss.masked(recon, target, mask2, 'mse', 1.0, 1e-8)) == pytest.approx(1.0)


@pytest.mark.characterization_bug  # F-03/F-04: zero-MAD calibration produces degenerate z-scores
def test_zero_mad_makes_robust_z_explode():
    errors = jp.zeros((16,), dtype=jp.float32)
    median, mad = Calibrate.robust_stats(errors)
    assert float(mad) == 0.0
    z = (jp.asarray(1.0) - median) / (1.4826 * mad + 1e-8)
    assert float(z) > 1e7  # a unit error maps to a ~1e8 z-score


def test_pad_split_empty_frame_yields_empty_tensors():
    """F-06 FIXED: an empty frame produces (0, max_len, dim) tensors (used to
    raise ValueError from jp.concatenate)."""
    import polars as pl
    empty = pl.DataFrame({'epi_sequence': []},
                         schema={'epi_sequence': pl.List(pl.List(pl.Float64))})
    padded, mask = Pad.split(empty, 'epi_sequence', dim=3, max_len=4, chunk=8)
    assert padded.shape == (0, 4, 3) and mask.shape == (0, 4)


@pytest.mark.characterization
def test_pad_split_shapes_and_mask():
    import polars as pl
    df = pl.DataFrame({'epi_sequence': [
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0]],
    ]})
    padded, mask = Pad.split(df, 'epi_sequence', dim=2, max_len=3, chunk=8)
    assert padded.shape == (2, 3, 2) and mask.shape == (2, 3)
    assert mask.tolist() == [[True, True, False], [True, False, False]]
    assert padded[0, 1].tolist() == [3.0, 4.0]
    assert padded[1, 1].tolist() == [0.0, 0.0]  # padded timestep is zeros
