"""Full evaluation surface (gap M9): everything the audit found missing.

Overall + per-anomaly-type P/R/FPR/F1/ROC-AUC/PR-AUC, threshold sweep,
recall @ FPR<=target, prevalence-adjusted PPV, calibration quality (ECE, Brier,
reliability), latency percentiles. Pure functions over arrays + one orchestrator
over trained artifacts.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np


def binary_metrics(y: np.ndarray, yhat: np.ndarray, score: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    fpr = fp / (fp + tn) if fp + tn else float('nan')
    f1 = (2 * prec * rec / (prec + rec)
          if (tp + fp) and (tp + fn) and (prec + rec) > 0 else float('nan'))
    try:
        roc = float(roc_auc_score(y, score))
    except ValueError:
        roc = float('nan')
    try:
        pr = float(average_precision_score(y, score))
    except ValueError:
        pr = float('nan')
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': prec,
            'recall': rec, 'fpr': fpr, 'f1': f1, 'roc_auc': roc, 'pr_auc': pr}


def recall_at_fpr(y: np.ndarray, score: np.ndarray, target_fpr: float) -> dict:
    """Highest recall achievable with FPR <= target; returns the threshold too."""
    thresholds = np.unique(score)[::-1]
    best = {'recall': 0.0, 'fpr': 0.0, 'threshold': float('inf')}
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {**best, 'recall': float('nan'), 'note': 'degenerate labels'}
    for t in thresholds:
        yhat = (score >= t).astype(int)
        fp = int(((yhat == 1) & (y == 0)).sum())
        fpr = fp / n_neg
        if fpr <= target_fpr:
            rec = int(((yhat == 1) & (y == 1)).sum()) / n_pos
            if rec > best['recall']:
                best = {'recall': rec, 'fpr': fpr, 'threshold': float(t)}
        else:
            break  # thresholds are descending; FPR only grows from here
    return best


def prevalence_adjusted_ppv(recall: float, fpr: float, prevalence: float) -> float:
    """PPV if deployed at `prevalence` with this recall/FPR operating point."""
    tp = recall * prevalence
    fp = fpr * (1.0 - prevalence)
    return tp / (tp + fp) if (tp + fp) > 0 else float('nan')


def ece_brier(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, w = float(p[m].mean()), float(y[m].mean()), float(m.mean())
        ece += w * abs(conf - acc)
        rows.append({'bin': b, 'mean_p': conf, 'frac_anomalous': acc, 'weight': w})
    return {'ece': float(ece), 'brier': float(np.mean((p - y) ** 2)),
            'reliability': rows}


def evaluate_split(y: np.ndarray, score: np.ndarray, p_anomaly: np.ndarray,
                   threshold: float, anomaly_types: np.ndarray | None,
                   target_fpr: float, assumed_prevalence: float,
                   calibration_bins: int = 10) -> dict:
    yhat = (score > threshold).astype(int)
    overall = binary_metrics(y, yhat, score)
    at_fpr = recall_at_fpr(y, score, target_fpr)
    out: dict[str, Any] = {
        'n': int(len(y)),
        'n_anomalous': int((y == 1).sum()),
        'threshold': float(threshold),
        'overall': overall,
        f'recall_at_fpr_{target_fpr}': at_fpr,
        'ppv_at_assumed_prevalence': {
            'prevalence': assumed_prevalence,
            'at_selected_threshold': prevalence_adjusted_ppv(
                overall['recall'], overall['fpr'], assumed_prevalence),
            'at_fpr_target': prevalence_adjusted_ppv(
                at_fpr['recall'], at_fpr['fpr'], assumed_prevalence)
            if not np.isnan(at_fpr.get('recall', float('nan'))) else float('nan'),
        },
        'calibration': ece_brier(y.astype(float), p_anomaly, calibration_bins),
        'per_anomaly_type': {},
    }
    if anomaly_types is not None:
        normal_mask = y == 0
        types = sorted({t for t in anomaly_types.tolist()
                        if t and str(t).lower() not in ('nonanomaly', 'unknown', 'none')})
        for t in types:
            t_mask = anomaly_types == t
            m = t_mask | normal_mask
            out['per_anomaly_type'][t] = {
                'n_anomalous': int(t_mask.sum()),
                **binary_metrics(y[m], yhat[m], score[m]),
                f'recall_at_fpr_{target_fpr}': recall_at_fpr(y[m], score[m], target_fpr),
            }
    return out


def measure_latency(predict_one, n_items: int, reps: int, warmup: int = 3) -> dict:
    for i in range(min(warmup, n_items)):
        predict_one(i)
    times = []
    for i in range(reps):
        t0 = time.perf_counter()
        predict_one(i % n_items)
        times.append((time.perf_counter() - t0) * 1000)
    return {'p50_ms': float(np.percentile(times, 50)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'mean_ms': float(np.mean(times)), 'reps': reps}
