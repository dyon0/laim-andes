from    typing                  import Tuple

import  polars                  as pl

from    ars.data.stages_meta    import S1Meta, S2Meta, S3Meta


def analyze_anomalies(data: pl.LazyFrame, meta: Tuple[S1Meta, S2Meta, S3Meta]) -> pl.LazyFrame:
    '''RCA seam (gap M11): formats the detector's attribution surface into a
    human-readable per-trace summary. The actual root-cause analysis (LMA per
    the LumiMAS paper) plugs in here; until then the report names the worst
    spans and the EPI features driving the reconstruction error.'''
    s1_meta, _s2_meta, _s3_meta = meta
    cols = data.collect_schema().names()
    if 'rca_top_span_indices' not in cols:
        return data.with_columns(rca_report_str = pl.lit(''))

    feature_names = list(s1_meta.epi_features)

    def _summarize(row: dict) -> str:
        spans = ', '.join(
            f'span#{i} (err={e:.4f})'
            for i, e in zip(row['rca_top_span_indices'], row['rca_top_span_errors']))
        feats = ', '.join(
            f'{feature_names[i] if i < len(feature_names) else f"f{i}"} (err={e:.4f})'
            for i, e in zip(row['rca_top_feature_indices'], row['rca_top_feature_errors']))
        return f'worst spans: {spans}; driving features: {feats}'

    return data.with_columns(
        pl.struct('rca_top_span_indices', 'rca_top_span_errors',
                  'rca_top_feature_indices', 'rca_top_feature_errors')
          .map_elements(_summarize, return_dtype = pl.String)
          .alias('rca_report_str'))
