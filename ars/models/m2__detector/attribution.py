"""RCA seams (gap M11): reconstruction-error attribution surfaces.

RCA itself is the team's R&D; this module provides the hooks it needs:
  * per_span_errors    — which timesteps (spans) of a trace reconstruct badly
  * per_feature_errors — which EPI features drive the error
  * counterfactual     — error delta under a caller-supplied input edit

All functions are pure over the already-trained branch models.
"""
from __future__ import annotations

from typing import Callable, Tuple

import jax as jx
import jax.numpy as jp
from flax.core import FrozenDict

from ars.models.m2__detector.architecture import LSTM_AE, Array, Params


@jx.jit(static_argnames=('model',))
def _pointwise(params: Params, model: LSTM_AE, padded: Array, mask: Array) -> Array:
    seq_lengths = jp.sum(mask, axis=1).astype(jp.int32)
    recon, _ = model.apply(FrozenDict({'params': params}), padded, seq_lengths,
                           training=False)
    return (recon - padded) ** 2 * mask[..., None]


def per_span_errors(model: LSTM_AE, params: Params, padded: Array, mask: Array) -> Array:
    """[N, T]: mean reconstruction error of each timestep (0 on padding)."""
    pw = _pointwise(params, model, padded, mask)
    return jp.mean(pw, axis=-1) * mask


def per_feature_errors(model: LSTM_AE, params: Params, padded: Array, mask: Array) -> Array:
    """[N, D]: per-feature error averaged over each trace's valid timesteps."""
    pw = _pointwise(params, model, padded, mask)
    steps = jp.maximum(jp.sum(mask, axis=1, keepdims=True), 1)
    return jp.sum(pw, axis=1) / steps


def top_k(values: Array, k: int) -> Tuple[Array, Array]:
    """(indices, values) of the k largest entries along the last axis."""
    k = min(k, values.shape[-1])
    vals, idx = jx.lax.top_k(values, k)
    return idx, vals


def counterfactual(
    model: LSTM_AE, params: Params, padded: Array, mask: Array,
    edit: Callable[[Array], Array],
) -> Array:
    """[N]: change of the per-trace error when `edit` transforms the inputs.

    `edit` receives the padded tensor and returns an edited tensor of the same
    shape (e.g. zero a feature column, clamp a span's duration). A negative
    delta means the edit makes the trace look MORE normal — evidence that the
    edited coordinates carried the anomaly.
    """
    base = per_span_errors(model, params, padded, mask).sum(axis=1)
    edited = per_span_errors(model, params, edit(padded), mask).sum(axis=1)
    return edited - base
