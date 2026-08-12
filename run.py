#!/usr/bin/env python
"""Single entry point for the LAIM anomaly-detection pipeline.

    run.py synth     --config configs/default.toml          # generate synthetic spans
    run.py validate  --spans data/traces_1k_sample.parquet  # data-contract check
    run.py prepare   --config ...                            # s1 only
    run.py train     --config ...                            # s1 + s2 (+ s3)
    run.py eval      --config ...                            # s1 + s2 (+ s3) + full eval report
    run.py infer     --model-dir runs/<run> --spans new.parquet
    run.py all       --config ...                            # prepare→train→eval[→infer]

Any config value can be overridden: `--set detector.epochs=100 --set runtime.seed=7`.
Training and inference are separate: `train`/`eval`/`all` produce a run directory
with models + manifest + metrics report; `infer` consumes one.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=(
        'synth', 'validate', 'prepare', 'train', 'eval', 'infer', 'all'))
    ap.add_argument('--config', help='TOML config file')
    ap.add_argument('--set', dest='overrides', action='append', default=[],
                    metavar='SECTION.KEY=VALUE', help='config override (repeatable)')
    ap.add_argument('--spans', help='spans parquet (validate / infer / all-infer)')
    ap.add_argument('--model-dir', help='trained run directory (infer)')
    args = ap.parse_args()

    from laim.config import load_config
    from laim.pipeline import run

    cfg = load_config(args.config, args.overrides)
    result = run(cfg, args.command, spans=args.spans, model_dir=args.model_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()
