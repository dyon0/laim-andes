"""Regenerate golden characterization artifacts (tests/golden/golden.json).

Run ONLY when a behavior change is intended; the diff must be explained in the
same commit (see PLAN.md metric-movement policy).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import polars as pl  # noqa: E402

SEED = 12345


def fixture_spans() -> pl.DataFrame:
    df = pl.read_parquet(REPO / 'data' / 'traces_1k_sample.parquet')
    ids = df.select('trace_id').unique().sort('trace_id').head(20)
    return df.join(ids, on='trace_id', how='semi').sort('trace_id', 'start_time_ns')


def summarize_matrix(df: pl.DataFrame, cols: list[str]) -> dict:
    stats = {}
    for c in cols:
        s = df[c].cast(pl.Float64)
        stats[c] = {
            'mean': round(float(s.mean()), 9) if s.mean() is not None else None,
            'std': round(float(s.std() or 0.0), 9),
            'nulls': int(s.null_count()),
        }
    return stats


def main() -> None:
    from ars.configuration.c1__data import S1Config
    from ars.data.features import FeaturePatterns, FeaturesSpan, RawSchema
    from ars.stages.s1__data import (
        calculate_features, fill_missing_values, build_traces,
        stratified_split, normalize_epi_features, _set_seeds)
    from ars.data.anomalies_injection import inject_anomalies, InjectionConfig
    from ars.specification.spec import DataObject

    import tempfile
    out_dir = Path(tempfile.mkdtemp(prefix='golden_'))
    cfg = S1Config(
        input_parquet_files=(), output_dir=out_dir, output_prefix='golden',
        seed_random=SEED, seed_polars=SEED, seed_torch=SEED,
        seed_split=SEED, seed_synth=SEED, seed_llm=SEED)
    _set_seeds(cfg, None)

    spans = fixture_spans()
    # the pipeline adds label stubs before feature computation (load_spans)
    spans = spans.with_columns(
        pl.lit(DataObject.class_sentinel).alias(DataObject.label),
        pl.lit(DataObject.class_sentinel).alias(DataObject.sublabel),
        pl.lit(None, dtype=pl.Int8).alias(DataObject.is_anomaly))

    golden: dict = {'seed': SEED, 'n_fixture_traces': spans['trace_id'].n_unique()}

    # --- 1. raw feature construction (FeaturesSpan.make_features) ---
    feats = FeaturesSpan().make_features(spans.clone(), FeaturePatterns())
    local_names = sorted(
        fd.name for fd in vars(FeaturesSpan()).values()
        if getattr(fd, 'include_in_sequence', False) and hasattr(fd, 'name'))
    golden['local_feature_names'] = local_names
    numeric_locals = [n for n in local_names if n in feats.columns
                      and feats.schema[n].is_numeric()]
    golden['local_feature_stats'] = summarize_matrix(feats, numeric_locals)
    golden['duration_diff_is_all_zero'] = bool(
        feats.select((pl.col('duration_diff') == 0).all()).item())

    # --- 2. full s1 feature stage (selection + fill) ---
    spans_f, epi_names = calculate_features(spans.clone(), cfg, FeaturePatterns(), RawSchema())
    golden['epi_feature_names'] = list(epi_names)
    golden['epi_dim'] = len(epi_names)

    # --- 3. injection labels (embedder=None degraded path, deterministic) ---
    injected = inject_anomalies(
        spans_f, InjectionConfig(sem_cols=(), text_col=None), embedder=None)
    labels = (injected.group_by('trace_id')
              .agg(pl.col('anomaly_type').drop_nulls().first())
              .sort('trace_id'))
    golden['injection_labels'] = dict(labels.iter_rows())

    injected = injected.with_columns(
        (pl.col(DataObject.sublabel) != 'NonAnomaly').cast(pl.Int8).alias(DataObject.is_anomaly))

    # --- 4. traces, split membership, normalization params ---
    traces = build_traces(injected, (), cfg)
    train, val, test = stratified_split(traces, DataObject.sublabel, cfg)
    golden['split_membership'] = {
        name: sorted(df['trace_id'].to_list())
        for name, df in (('train', train), ('val', val), ('test', test))}
    (train_n, _val_n, _test_n), norm_params = normalize_epi_features(train, val, test, cfg)
    golden['normalization'] = {
        'method': norm_params['method'],
        'shift': [round(float(x), 9) for x in norm_params['shift']],
        'scale': [round(float(x), 9) for x in norm_params['scale']],
    }
    golden['normalization_min_scale'] = round(float(min(norm_params['scale'])), 9)

    # --- 5. detector micro-train on deterministic tensors ---
    import jax
    import jax.numpy as jp
    from ars.models.m2__detector.architecture import LSTM_AE, HyperParamsLSTMAE
    from ars.models.m2__detector.train import Trainer
    from ars.stages.s2__detector import select_threshold

    key = jax.random.PRNGKey(SEED)
    xs = jax.random.normal(key, (12, 6, 5), dtype=jp.float32)
    mask = jp.ones((12, 6), dtype=bool)
    hp = HyperParamsLSTMAE(sz_features=5, sz_latent=4,
                           layers_arch=(('unidirectional', 8),))
    state, losses = Trainer.train_lstm_ae(
        model=LSTM_AE(hp), train_padded=xs, train_mask=mask,
        val_padded=xs[:4], val_mask=mask[:4],
        learning_rate=1e-3, num_epochs=3, batch_size=4, rng=key,
        input_shape=(6, 5), weight_decay=0.0, clip_grad=1.0,
        schedule_fn=None, patience=10, target_loss=None)
    golden['micro_train_losses'] = [round(float(l), 7) for l in losses]

    errors = jp.asarray([0.1, 0.2, 0.3, 0.9, 1.1, 1.3], dtype=jp.float32)
    labels_arr = jp.asarray([0, 0, 0, 1, 1, 1], dtype=jp.int32)
    thr, metrics = select_threshold(errors, labels_arr, 101, 'youden', 1e-8)
    golden['micro_threshold'] = round(float(thr), 7)
    golden['micro_threshold_youden'] = round(float(metrics.youden), 7)

    out = Path(__file__).parent / 'golden.json'
    out.write_text(json.dumps(golden, indent=2, ensure_ascii=False))
    print('golden written:', out)


if __name__ == '__main__':
    main()
