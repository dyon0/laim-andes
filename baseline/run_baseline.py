"""Phase 0 baseline runner: executes the UNMODIFIED legacy pipeline end to end.

Usage:
    .venv/bin/python baseline/run_baseline.py --corpus real  --out baseline/runs/real_a
    .venv/bin/python baseline/run_baseline.py --corpus synth --out baseline/runs/synth_a

Notes recorded in AUDIT_00_baseline.md:
  * huggingface.co is blocked by the environment network policy, so the real
    deepvk/USER-bge-m3 cannot be used. A deterministic random-weight stand-in
    with identical interface (SentenceTransformer dir, 1024-dim, L2-normalized)
    is used instead. This does NOT change the code path: the legacy s1 stage
    stubs all normal-span embeddings to constant zeros anyway (s1__data.py:505);
    the embedder is only exercised by the anomaly injector.
  * No source file under ars/ is modified by this runner.
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EMBEDDER = '/tmp/models/USER-bge-m3-standin'
SAMPLE = Path(__file__).resolve().parents[1] / 'data' / 'traces_1k_sample.parquet'


def split_real(work: Path, train_frac: float, seed: int) -> tuple[str, str]:
    """Split the real sample into train/test files at TRACE boundaries.

    The legacy ars/main.py __main__ splits by raw row index, which cuts traces
    in half; we deliberately split by trace_id here so the baseline measures the
    pipeline, not a data-preparation accident.
    """
    import polars as pl
    df = pl.read_parquet(SAMPLE)
    traces = df.select('trace_id').unique(maintain_order=True).sort('trace_id')
    n_train = int(traces.height * train_frac)
    shuffled = traces.sample(fraction=1.0, shuffle=True, seed=seed)
    train_ids = shuffled.head(n_train)
    test_ids = shuffled.tail(traces.height - n_train)
    p_train = work / 'train_spans.parquet'
    p_test = work / 'test_spans.parquet'
    df.join(train_ids, on='trace_id', how='semi').write_parquet(p_train)
    df.join(test_ids, on='trace_id', how='semi').write_parquet(p_test)
    return str(p_train), str(p_test)


def synth_corpus(work: Path, target_spans: int, seed: int) -> tuple[str, str]:
    """Generate a corpus with the repo's own synthesizer (deterministic)."""
    import polars as pl
    from ars.data.synthesis import synthesize, GenConfig
    frame = synthesize(GenConfig(target_spans=target_spans, seed=seed)).collect()
    traces = frame.select('trace_id').unique().sort('trace_id')
    shuffled = traces.sample(fraction=1.0, shuffle=True, seed=seed)
    n_train = int(traces.height * 0.8)
    p_train = work / 'train_spans.parquet'
    p_test = work / 'test_spans.parquet'
    frame.join(shuffled.head(n_train), on='trace_id', how='semi').write_parquet(p_train)
    frame.join(shuffled.tail(traces.height - n_train), on='trace_id', how='semi').write_parquet(p_test)
    return str(p_train), str(p_test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', choices=('real', 'synth'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--synth-spans', type=int, default=20000)
    ap.add_argument('--synth-seed', type=int, default=20250601)
    ap.add_argument('--train-frac', type=float, default=0.8)
    ap.add_argument('--recast', default='true')
    args = ap.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.corpus == 'real':
        p_train, p_test = split_real(out, args.train_frac, args.seed)
    else:
        p_train, p_test = synth_corpus(out, args.synth_spans, args.synth_seed)

    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                            text=True, cwd=Path(__file__).parents[1]).stdout.strip()
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                            capture_output=True, text=True).stdout

    t0 = time.time()
    from ars.main import main as run_all
    result = run_all(
        p_train, p_test, EMBEDDER,
        output_prefix='baseline',
        device='',                       # cpu
        recast=args.recast,
        disable_progress=True,
        track_peak_memory=False,
        s1_seed_random=args.seed, s1_seed_polars=args.seed, s1_seed_torch=args.seed,
        s1_seed_split=args.seed, s1_seed_synth=args.seed, s1_seed_llm=args.seed,
        s2_seed=args.seed,
        s3_seed=args.seed,
    )
    wall = time.time() - t0

    payload = {
        'corpus': args.corpus,
        'seed': args.seed,
        'commit': commit,
        'python': sys.version,
        'platform': platform.platform(),
        'gpu': 'none (CPU-only container)',
        'embedder': EMBEDDER + ' (random-weight stand-in, HF blocked)',
        'train_file': p_train,
        'test_file': p_test,
        'wall_seconds_total': wall,
        'detector_metrics_holdout': result['detector_metrics_holdout'],
        'classifier_metrics_holdout': result['classifier_metrics_holdout'],
        'n_detected_anomalies_on_test_file': int(len(result['anomaly_traces'])),
    }
    (out / 'run_result.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    (out / 'pip_freeze.txt').write_text(freeze)
    print('BASELINE_RUN_DONE', json.dumps(payload['detector_metrics_holdout'], default=str)[:500])


if __name__ == '__main__':
    main()
