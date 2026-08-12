"""Phase 0 post-hoc baseline evaluation (read-only).

Loads the artifacts written by a `run_baseline.py` run (the newest
/tmp/mas-monitor#* directory unless --run-dir is given) and computes the
metrics the legacy pipeline does NOT produce on its own:

  * reconstruction error per branch (EPI / SEM / Combined) per split
  * the threshold + Youden sweep it came from (recomputed on VAL)
  * Precision / Recall / FPR / F1 / ROC-AUC / PR-AUC overall AND per anomaly type
  * calibration quality of p_anomaly: ECE (10 bins), Brier, reliability table
  * per-trace inference latency p50 / p95 / p99 (batch=1, after warmup)

No file under ars/ is modified; only public loaders/predictors are called.
Output: baseline/baseline_metrics.json (merged with run_result.json).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl


def newest_run_dir() -> Path:
    runs = sorted(Path('/tmp').glob('mas-monitor#*'), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError('no /tmp/mas-monitor#* run directory found')
    return runs[-1]


def ece_brier(labels: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc, w = float(p[m].mean()), float(labels[m].mean()), float(m.mean())
        ece += w * abs(conf - acc)
        rows.append({'bin': b, 'mean_p': conf, 'frac_anomalous': acc, 'weight': w})
    return {'ece': ece, 'brier': float(np.mean((p - labels) ** 2)), 'reliability': rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, default=None)
    ap.add_argument('--run-result', type=Path, required=True,
                    help='run_result.json from run_baseline.py')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--latency-reps', type=int, default=50)
    args = ap.parse_args()

    run_dir = args.run_dir or newest_run_dir()
    train_dir = run_dir / 'traces_train'
    models_dir = train_dir / 'models'

    meta_json = next(train_dir.glob('*_meta.json'))
    s1_meta_raw = json.loads(meta_json.read_text())
    prefix = meta_json.name.replace('_meta.json', '')

    import jax.numpy as jp
    from sklearn.metrics import average_precision_score, roc_auc_score

    from ars.data.stages_meta import S1Meta, S2Meta
    from ars.models.m2__detector.confidence import Predict
    from ars.stages.s2__detector import Pad, load_models_for_inference, _threshold_sweep
    from ars.tools.utilities.miscellaneous import FileIO

    s1_meta = S1Meta(**s1_meta_raw)
    best_info = FileIO.json_read(models_dir / 'best' / 'best_info.json')
    exp_name = best_info['best_experiment']
    exp_dir = models_dir / exp_name
    combined = FileIO.pickle_read(exp_dir / 'combined_model.pkl')
    results = FileIO.json_read(exp_dir / 'results.json')

    s2_meta = S2Meta(
        output_dir=str(models_dir), experiment_dir=str(exp_dir),
        best_experiment=exp_name, select_metric=best_info['metric'],
        best_metric_value=float(best_info['value']),
        max_len=0, epi_dim=s1_meta.epi_dim,
        sem_dim=s1_meta.semantic_vectors['sem_sem_vector'],
        epi_sz_latent=int(combined['config'].sz_latent_epi),
        sem_sz_latent=int(combined['config'].sz_latent_sem),
        seq_pad_chunk=8192,
        best_threshold=float(combined['best_threshold']),
        normalize_latent=bool(combined['normalize_latent']),
        epi_latent_mean=combined['epi_latent_mean'], epi_latent_std=combined['epi_latent_std'],
        sem_latent_mean=combined['sem_latent_mean'], sem_latent_std=combined['sem_latent_std'],
        test_metrics=results['test_metrics'], calibration=combined['calibration'])

    splits = {
        s: pl.read_parquet(train_dir / f'{prefix}_{s}.parquet')
        for s in ('train', 'val', 'test')
    }
    max_len = max(
        int(df.select(pl.col('epi_sequence').list.len().max()).item()) for df in splits.values())
    s2_meta = type(s2_meta)(**{**s2_meta.__dict__, 'max_len': max_len})
    meta, models = load_models_for_inference(s2_meta)
    meta = type(meta)(**{**meta.__dict__, 'max_len': max_len})

    def tensors(df: pl.DataFrame):
        epi, epi_m = Pad.split(df, 'epi_sequence', s1_meta.epi_dim, max_len, 8192)
        sem, sem_m = Pad.split(df, 'sem_sequence_sem_vector', s2_meta.sem_dim, max_len, 8192)
        return epi, epi_m, sem, sem_m

    per_split = {}
    outs = {}
    for name, df in splits.items():
        epi, epi_m, sem, sem_m = tensors(df)
        out = Predict.batch(meta, models, epi, epi_m, sem, sem_m)
        outs[name] = (df, out)
        per_split[name] = {
            'n_traces': df.height,
            'epi_recon_error_mean': float(jp.mean(out.e_epi)),
            'sem_recon_error_mean': float(jp.mean(out.e_sem)),
            'combined_recon_error_mean': float(jp.mean(out.e_comb)),
        }

    # threshold + Youden sweep recomputed on VAL (mixed)
    val_df, val_out = outs['val']
    val_labels = val_df['is_anomaly'].fill_null(0).to_numpy().astype(np.int32)
    thr_arr, sweep = _threshold_sweep(
        val_out.e_comb, jp.asarray(val_labels), 512, jp.asarray(1e-8, dtype=jp.float32))
    youden_curve = {
        'thresholds': [float(x) for x in thr_arr],
        'youden': [float(x) for x in sweep.youden],
    }

    # TEST metrics overall and per anomaly type
    test_df, test_out = outs['test']
    y = test_df['is_anomaly'].fill_null(0).to_numpy().astype(np.int32)
    e = np.asarray(test_out.e_comb)
    p = np.asarray(test_out.p_anomaly)
    yhat = (e > s2_meta.best_threshold).astype(np.int32)

    def binary_metrics(y_, yhat_, score_) -> dict:
        tp = int(((yhat_ == 1) & (y_ == 1)).sum()); fp = int(((yhat_ == 1) & (y_ == 0)).sum())
        fn = int(((yhat_ == 0) & (y_ == 1)).sum()); tn = int(((yhat_ == 0) & (y_ == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else float('nan')
        fpr = fp / (fp + tn) if fp + tn else float('nan')
        f1 = 2 * prec * rec / (prec + rec) if prec + rec and not (np.isnan(prec) or np.isnan(rec)) else float('nan')
        try: roc = float(roc_auc_score(y_, score_))
        except ValueError: roc = float('nan')
        try: pr = float(average_precision_score(y_, score_))
        except ValueError: pr = float('nan')
        return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': prec, 'recall': rec,
                'fpr': fpr, 'f1': f1, 'roc_auc': roc, 'pr_auc': pr}

    overall = binary_metrics(y, yhat, e)
    per_type = {}
    types = [t for t in test_df['anomaly_type'].unique().to_list()
             if t and t.lower() not in ('nonanomaly', 'unknown')]
    normal_mask = y == 0
    for t in sorted(types):
        t_mask = (test_df['anomaly_type'] == t).to_numpy()
        m = t_mask | normal_mask     # this type's anomalies vs all normals
        per_type[t] = {'n_anomalous': int(t_mask.sum()),
                       **binary_metrics(y[m], yhat[m], e[m])}

    calibration_quality = ece_brier(y.astype(float), p)

    # latency, batch=1, warmed up
    epi, epi_m, sem, sem_m = tensors(test_df)
    for _ in range(3):
        Predict.batch(meta, models, epi[:1], epi_m[:1], sem[:1], sem_m[:1])
    times = []
    n = epi.shape[0]
    for i in range(args.latency_reps):
        j = i % n
        t0 = time.perf_counter()
        out = Predict.batch(meta, models, epi[j:j+1], epi_m[j:j+1], sem[j:j+1], sem_m[j:j+1])
        _ = float(out.p_anomaly[0])   # force sync
        times.append((time.perf_counter() - t0) * 1000)
    lat = {'p50_ms': float(np.percentile(times, 50)),
           'p95_ms': float(np.percentile(times, 95)),
           'p99_ms': float(np.percentile(times, 99)),
           'reps': args.latency_reps, 'batch': 1, 'device': 'cpu'}

    run_result = json.loads(args.run_result.read_text())
    payload = {
        **run_result,
        'run_dir': str(run_dir),
        'best_experiment': exp_name,
        'selected_threshold': float(s2_meta.best_threshold),
        'threshold_selected_on': 'val (mixed) — but experiment picked by TEST metric (see finding)',
        'reconstruction_errors': per_split,
        'youden_curve_val': youden_curve,
        'test_overall': overall,
        'test_per_anomaly_type': per_type,
        'calibration_quality_test': calibration_quality,
        'latency_per_trace': lat,
        'experiment_results_raw': results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print('EVAL_DONE', args.out)


if __name__ == '__main__':
    main()
