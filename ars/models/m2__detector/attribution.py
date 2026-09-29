"""RCA seams (gap M11): reconstruction-error attribution surfaces.

RCA itself is the team's R&D; this module provides the hooks it needs:
  * per_span_errors    — which timesteps (spans) of a trace reconstruct badly
  * per_feature_errors — which EPI features drive the error
  * counterfactual     — error delta under a caller-supplied input edit
  * explain            — one-pass export surface: padding-safe top spans, top
                         features with observed/expected values at their peak
                         span, and the features driving each top span
  * combined_epi_share — which branch the flagging (combined) error came from

All functions are pure over the already-trained branch models.
"""
from __future__ import annotations

from typing import Callable, Tuple

import jax as jx
import jax.numpy as jp
import numpy as np
from flax.core import FrozenDict

from ars.models.m2__detector.architecture import (
    FMLP_AE, LSTM_AE, Array, Branch, Params, TrainState, block_rows, row_blocks)
from ars.models.metrics import Loss

# traces per jitted call: bounds the transient [chunk, T, D] error tensors
# (SEM branch: D = 1024 — 256 traces x 200 steps is ~200 MB per tensor)
EXPLAIN_CHUNK = 256


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


@jx.jit(static_argnames=('model', 'k_spans', 'k_feats', 'k_drivers'))
def _explain_chunk(params: Params, model: LSTM_AE, padded: Array, mask: Array, driver_mask: Array,
                   k_spans: int, k_feats: int, k_drivers: int) -> dict[str, Array]:
    seq_lengths = jp.sum(mask, axis=1).astype(jp.int32)
    recon, _ = model.apply(FrozenDict({'params': params}), padded, seq_lengths,
                           training=False)
    # the branch's own training loss, so the span errors decompose the
    # detector's e_epi / e_sem exactly (huber branches included)
    pw = Loss.pointwise(recon, padded, model.hp.loss_type, model.hp.huber_delta) * mask[..., None]
    steps = jp.maximum(jp.sum(mask, axis=1), 1).astype(pw.dtype)
    tiny = jp.asarray(1e-12, dtype=pw.dtype)
    span_err = jp.mean(pw, axis=-1)
    span_sum = jp.sum(span_err, axis=1)
    # padding can never win a top-k slot: it scores -inf and is dropped later
    span_val, span_idx = jx.lax.top_k(jp.where(mask, span_err, -jp.inf), k_spans)
    out = {'error': span_sum / steps, 'span_idx': span_idx, 'span_err': span_val,
           'span_share': span_val / (span_sum[:, None] + tiny)}
    if k_feats:
        feat_err = jp.sum(pw, axis=1) / steps[:, None]
        feat_val, feat_idx = jx.lax.top_k(feat_err, k_feats)
        per_step = lambda arr: jp.take_along_axis(arr, feat_idx[:, None, :], axis=2)   # [N, T, kf]
        peak = jp.argmax(jp.where(mask[..., None], per_step(pw), -jp.inf), axis=1)     # [N, kf]
        at_peak = lambda arr: jp.take_along_axis(per_step(arr), peak[:, None, :], axis=1)[:, 0, :]
        out |= {'feat_idx': feat_idx, 'feat_err': feat_val,
                'feat_share': feat_val / (jp.sum(feat_err, axis=1, keepdims=True) + tiny),
                'feat_peak': peak, 'feat_obs': at_peak(padded), 'feat_exp': at_peak(recon)}
    if k_drivers:
        at_span = lambda arr: jp.take_along_axis(arr, span_idx[:, :, None], axis=1)    # [N, ks, D]
        eligible = jp.where(driver_mask[None, None, :], at_span(pw), -jp.inf)
        drv_val, drv_idx = jx.lax.top_k(eligible, k_drivers)
        out |= {'drv_idx': drv_idx, 'drv_err': drv_val,
                'drv_obs': jp.take_along_axis(at_span(padded), drv_idx, axis=2),
                'drv_exp': jp.take_along_axis(at_span(recon), drv_idx, axis=2)}
    return out


def explain(model: LSTM_AE, params: Params, padded: Array, mask: Array,
            k_spans: int, k_feats: int = 0, k_drivers: int = 0,
            driver_mask: None | np.ndarray = None, chunk: int = EXPLAIN_CHUNK,
            rows: None | int = None) -> dict[str, np.ndarray]:
    """Attribution of one branch's reconstruction error, as numpy arrays.

    Always: `error` [N] (equals the detector's branch error), `span_idx`,
    `span_err`, `span_share` [N, ks] — the ks worst spans; slots beyond a
    trace's length score -inf (callers drop them). With k_feats: `feat_idx`,
    `feat_err`, `feat_share`, `feat_peak` (span where the feature deviates
    most), `feat_obs` / `feat_exp` (normalized input vs reconstruction at that
    span) [N, kf]. With k_drivers: `drv_idx`, `drv_err`, `drv_obs`, `drv_exp`
    [N, ks, kd] — the features driving each top span, chosen among
    `driver_mask` [D] (default: all); slots with no eligible feature score -inf.
    rows: traces per jitted call — the training run's row count
    (S2Meta.infer_rows) when it is within `chunk`, so the forward reuses shapes
    the process already compiled; else n's own bucket, at most `chunk`.
    """
    n, t, d = padded.shape
    ks, kf, kd = min(k_spans, t), min(k_feats, d), min(k_drivers, d)
    eligible = jp.asarray(np.ones(d, dtype=bool) if driver_mask is None else driver_mask, dtype=bool)
    # fixed-size blocks: one compile per model, not one per input size; the
    # padding is cut on the host (row_blocks in architecture.py)
    rows = rows if rows and rows <= chunk else block_rows(n, chunk)
    parts = tuple(
        {key: v[:k] for key, v in jx.device_get(
            _explain_chunk(params, model, x, m, eligible, ks, kf, kd)).items()}
        for (x, m), k in row_blocks((padded, mask), rows))
    if not parts:
        return {}
    return {key: np.concatenate(tuple(p[key] for p in parts)) for key in parts[0]}


def combined_epi_share(state: TrainState, model: FMLP_AE, epi_lat: Array, sem_lat: Array,
                       rows: None | int = None) -> np.ndarray:
    """[N]: fraction of the combined (flagging) error carried by the EPI
    latent block of the FMLP reconstruction; 1 - share is the SEM block's."""
    recon = Branch.combined_forward(state, model, epi_lat, sem_lat, rows)
    target = jp.concatenate((epi_lat, sem_lat), axis=-1)
    pw = np.asarray(Loss.pointwise(recon, target, model.hp.loss_type, model.hp.huber_delta))
    split = epi_lat.shape[-1]
    return pw[:, :split].sum(axis=1) / np.maximum(pw.sum(axis=1), 1e-12)
