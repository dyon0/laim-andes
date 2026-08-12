from    pathlib                 import Path

from    ars.data.stages_meta    import S1Meta, S2Meta


def collect_holdout_metrics(s2_meta: S2Meta, s3_meta = None) -> tuple[dict, dict]: #todo: типизировать все dict до конца
    detector    = {**dict(s2_meta.test_metrics), **dict(s2_meta.calibration)}
    classifier  = dict(getattr(s3_meta, 'test_metrics', {}) or {})

    return detector, classifier


def collect_reports(s1_meta: S1Meta, s2_meta: S2Meta, s3_meta = None) -> dict: #todo: типизировать все dict до конца
    read = lambda p: Path(p).read_text(encoding = 'utf-8') if Path(p).exists() else ''

    return {
        'data'      : read(Path(s1_meta.output_dir) / 'data_report.html'),
        'detector'  : read(Path(s2_meta.output_dir) / 'summary_report.html'),
        'classifier': read(Path(s3_meta.output_dir) / 'classifier_report.html') if s3_meta is not None else ''}