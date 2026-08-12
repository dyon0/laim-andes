import  ars.configuration.c0__env_setup
from    ars.configuration.c0__env_setup     import Runtime

from    typing                              import ClassVar
from    dataclasses                         import dataclass
from    functools                           import partial
from    itertools                           import starmap

from    argparse                            import ArgumentParser
from    pathlib                             import PurePath, Path
from    datetime                            import datetime
from    random                              import choices
from    json                                import dumps
from    sys                                 import exit
from    zipfile                             import ZipFile

import  polars                              as pl

from    ars.configuration.c0__device        import Device
from    ars.stages.s1__data                 import process_train_data, prepare_test_data
from    ars.stages.s2__detector             import train_detector, detect_anomalies
from    ars.stages.s3__classifier           import train_classifier, classify_anomalies
from    ars.stages.s4__rca                  import analyze_anomalies
from    ars.stages.s5__reports              import collect_holdout_metrics, collect_reports
from    ars.specification.common_core       import Sentinel
from    ars.specification.spec              import recast as do_recast
from    ars.tools.abstraction.composition   import identity
from    ars.tools.tui.tui                   import ColorSchemeDataScience, sprint, redirect_native_stderr
from    ars.tools.tui.tui_data              import ColorSchemeRetroWave


@dataclass(frozen = True)
class Anomalies:
    clock   : ClassVar[str]             = '%Y-%m-%dT%H:%M:%SZ'
    blanks  : ClassVar[tuple[str, ...]] = ('anomaly_type', 'business_description', 'user_query', 'agent_response', 'tech_details', 'rca_results', '_comment')

    @staticmethod
    def bounds(lf: pl.LazyFrame) -> pl.DataFrame:
        return (lf
                  .group_by('trace_id')
                  .agg(
                      pl.col('start_time_ns').min().alias('_t0'),
                      pl.col('end_time_ns').max().alias('_t1'))
                  .collect())

    @staticmethod
    def enrich(detected: pl.DataFrame, bounds: pl.DataFrame) -> pl.DataFrame:
        timed   = (detected
                     .join(bounds, on = 'trace_id', how = 'left')
                     .with_columns(
                         pl.col('_t0').cast(pl.Datetime(time_unit = 'ns')).dt.strftime(Anomalies.clock).alias('starttime'),
                         pl.col('_t1').cast(pl.Datetime(time_unit = 'ns')).dt.strftime(Anomalies.clock).alias('endtime'),
                         (pl.col('detector_confidence') * 100).round().cast(pl.Int64).alias('confidence'))
                     .drop('_t0', '_t1'))
        absent  = filter(lambda n: n not in timed.columns, Anomalies.blanks)

        return timed.with_columns(map(lambda n: pl.lit('').alias(n), absent))

    @staticmethod
    def records(frame: pl.DataFrame) -> list:
        field   = lambda r, n: r[n] if n in r else ''
        row     = lambda r: {
            '_comment'              : field(r, '_comment'),
            'trace_id'              : r['trace_id'],
            'starttime'             : r['starttime'],
            'endtime'               : r['endtime'],
            'anomaly_type'          : field(r, 'anomaly_type'),
            'confidence'            : int(r['confidence']),
            'business_description'  : field(r, 'business_description'),
            'user_query'            : field(r, 'user_query'),
            'agent_response'        : field(r, 'agent_response'),
            'tech_details'          : field(r, 'tech_details'),
            'rca_results'           : field(r, 'rca_results')}

        return list(map(row, frame.to_dicts()))


def extract_query_response(spans: pl.LazyFrame, detected: pl.DataFrame) -> pl.LazyFrame:
    keyed = spans.join(detected.lazy().select('trace_id').unique(), on = 'trace_id', how = 'semi').sort('start_time_ns')
    
    return (keyed
        .group_by('trace_id', maintain_order = True)
        .agg(
            pl.col('input_text')
              .filter(
                  (pl.col('aef_kind') == 'input_request')       &
                  (pl.col('input_text').is_not_null())          &
                  (pl.col('input_text') != Sentinel.unknown)    &
                  (pl.col('input_text') != ''))
              .first()
              .alias('user_query'),
            pl.col('output_text')
              .filter(
                  (pl.col('aef_kind') == 'output_request')      &
                  (pl.col('output_text').is_not_null())         &
                  (pl.col('output_text') != Sentinel.unknown)   &
                  (pl.col('output_text') != ''))
              .last()
              .alias('agent_response'))
        .with_columns(
            pl.col('user_query').fill_null('<запрос пользователя не обнаружен>'),
            pl.col('agent_response').fill_null('<ответ агента не обнаружен>')))


def main(
    path_traces_train   : str,
    path_traces_test    : str,
    path_embedder       : str,

    read_extensions     : bool = True, # вынести в descriptor, remove_pandas убрать

    output_prefix       : str  = 'demo',
    dont_colorize_log   : bool = True,

    device              : str  = '',
    recast              : bool = False,

    disable_progress    : bool  = False,
    progress_every      : float = 0.0,
    track_peak_memory   : bool  = True,

    **ui
) -> dict: #todo: типизировать dict[str, тип_сумма_значений]
    s1_int      = frozenset({'embedding_batch_size','embedding_max_length','llm_sem_max_new_tokens','llm_sem_batch_size',
        'output_max_collection_len','seed_random','seed_polars','seed_torch','seed_split','seed_synth','seed_llm'})
    s1_float    = frozenset({'min_fill_rate','max_static_rate','max_correlation','llm_sem_temperature','norm_train_ratio',
        'norm_val_ratio','anom_val_ratio','anom_test_ratio','eps_normalization','eps_divide','duration_scale_to_sec'})
    s1_bool     = frozenset({'use_meta_sem','use_llm_sem','llm_sem_use_stub','llm_sem_do_sample','winsorize_epi','export_features'})
    s2_int      = frozenset({'n_thresholds','inference_normal_count','inference_anomalous_count','seed','encode_chunk','seq_pad_chunk'})
    s2_float    = frozenset({'eps'})

    def overrides(ui: dict, prefix: str, ints: frozenset, floats: frozenset, bools: frozenset) -> dict:
        raw     = starmap(lambda k, v: (k[len(prefix):], v), filter(lambda kv: kv[0].startswith(prefix), ui.items()))
        cast    = lambda n, v: (
            int(v)                                                      if n in ints
            else float(v)                                               if n in floats
            else str(v).strip().lower() in ('true', '1', 'yes', 'on')   if n in bools
            else v)
        return dict(starmap(lambda n, v: (n, cast(n, v)), raw))
    
    cs = ColorSchemeDataScience() if dont_colorize_log else ColorSchemeRetroWave()

    unit    = Device.of(device)
    rec     = str(recast).strip().lower() in ('true', '1', 'yes', 'on')

    Runtime.apply(
        track_peak       = str(track_peak_memory).strip().lower() in ('true', '1', 'yes', 'on'),
        disable_progress = str(disable_progress).strip().lower()  in ('true', '1', 'yes', 'on'),
        progress_every   = float(progress_every or 0.0))

    unit.force()

    sprint('создание каталога запуска...', style_code = cs.info, end = '\t')
    try:
        run_id = ''.join(choices('abcdefghijklmnopqrstuvwxyz0123456789', k = 8))

        Path((
            root_path := PurePath(f'/tmp/mas-monitor#{(run_time := datetime.now()):%Y-%m-%d_%H:%M:%S}')
        )).mkdir(parents = False, exist_ok = False)
    except:
        sprint('не удалось создать системный каталог. завершение', style_code = cs.error)
        exit(1)    
    sprint(root_path.as_posix(), style_code = cs.success, end = '')
    sprint(f' | {run_time}', style_code = cs.info)

    s1_over = {**overrides(ui, 's1_', s1_int, s1_float, s1_bool), 'device': unit.torch}
    s2_over = overrides(ui, 's2_', s2_int, s2_float, frozenset())
    
    def extract_zip(zip_path: PurePath, extract_to: None | PurePath = None) -> PurePath:
        if not Path(zip_path).is_file(): raise FileNotFoundError(f'Архив не найден: {zip_path.as_posix()}')

        if extract_to is None: extract_to = zip_path.with_suffix('')

        Path(extract_to).mkdir(parents = True, exist_ok = True)

        with ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)

        return extract_to
    
    def resolve_model_path(raw_path: str | PurePath) -> PurePath:
        raw_path = PurePath(raw_path)

        if read_extensions:
            if not (raw_path.suffix.lower() == '.zip'): return raw_path
        
        Path(tmp_dir := PurePath('/tmp/data/embedders/')).mkdir(parents = True, exist_ok = True)
        
        return extract_zip(raw_path, extract_to = tmp_dir) / 'USER-bge-m3'
    
    model_path = resolve_model_path(path_embedder)
    
    _c, s1_meta = process_train_data(
        #path_path_traces_train,
        PurePath(path_traces_train),
        model_path,
        root_path, output_prefix, run_id, recast = rec, overrides = s1_over)
    lf_test     = prepare_test_data(
        _c,
        #path_path_traces_test,
        PurePath(path_traces_test),
        s1_meta,
        root_path, output_prefix)

    sprint(f'native-логи C++ => {(
        debug_log_path := root_path / 'debug.log').as_posix()}',
        style_code = cs.debug)

    with redirect_native_stderr(debug_log_path):
        s2_meta = train_detector(s1_meta, overrides = s2_over)
        #s3_meta = train_classifier(s2_meta)

        s3_over = overrides(ui, 's3_', frozenset({'seed'}), frozenset({'eps'}), frozenset())
        s3_meta = train_classifier(s1_meta, s2_meta, overrides = s3_over)
    
    detect      = partial(detect_anomalies,   s2_meta = s2_meta)
    classify    = partial(classify_anomalies, s3_meta = s3_meta)
    analyze     = partial(analyze_anomalies,  meta = (s1_meta, s2_meta, s3_meta))

    inference_pipeline = identity >> detect >> classify >> analyze
    inference_pipeline.compile()

    with redirect_native_stderr(debug_log_path):
        detected = inference_pipeline(lf_test).collect(engine = unit.engine)

    lf_spans_test   = pl.scan_parquet(PurePath(path_traces_test).as_posix())
    lf_spans_test   = do_recast(lf_spans_test) if rec else lf_spans_test
    query_resp_df   = extract_query_response(lf_spans_test, detected).collect(engine = unit.engine)
    detected        = detected.join(query_resp_df, on = 'trace_id', how = 'left')
    enriched        = Anomalies.enrich(detected, Anomalies.bounds(lf_spans_test))
    anomaly_traces  = enriched.to_pandas()
    test_anomalies  = dumps({'anomalies': Anomalies.records(enriched)}, ensure_ascii = False)

    detector_metrics_holdout, classifier_metrics_holdout    = collect_holdout_metrics(s2_meta, s3_meta)
    html_reports                                            = collect_reports(s1_meta, s2_meta, s3_meta)

    return {
        'detector_metrics_holdout'   : detector_metrics_holdout,
        'classifier_metrics_holdout' : classifier_metrics_holdout,
        'anomaly_traces'             : anomaly_traces,
        'test_anomalies'             : test_anomalies,
        'html_reports'               : html_reports}


if __name__ == '__main__':
    parser      = ArgumentParser(
        description = 'система распознавания аномалий в мультиагентных системах')
    args        = parser.parse_args()
    
    data        = pl.scan_parquet(
        '/mnt/data/traces/raw/synthetic__trip_planner.parquet').with_row_index('__i')

    tr_train    = data.filter(pl.col('__i').is_between(0,      79000)).drop('__i')
    tr_test     = data.filter(pl.col('__i').is_between(80000, 100000)).drop('__i')

    tr_train.collect().write_parquet(path_tr_train := PurePath('/tmp/tr_train.paquet').as_posix())
    tr_test.collect().write_parquet(path_tr_test := PurePath('/tmp/tr_test.paquet').as_posix())
    
    results_dict = main(
        path_tr_train,
        path_tr_test,
        '/mnt/data/models/embedder/encodechka.zip')
    #print(results_dict['test_anomalies'][-5:])

    #with open('/mnt/ta.txt', mode = 'w') as f: f.write(results_dict['test_anomalies'])
