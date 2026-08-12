from    typing                  import Tuple

import  polars                  as pl

from    ars.data.stages_meta    import S1Meta, S2Meta, S3Meta


def analyze_anomalies(data: pl.LazyFrame, meta: Tuple[S1Meta, S2Meta, S3Meta]) -> pl.LazyFrame:
    return data.with_columns(rca_report_str = pl.lit('-dummy-'))