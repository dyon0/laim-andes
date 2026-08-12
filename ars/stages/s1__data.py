import  ars.configuration.c0__env_setup

from    typing                          import Literal, FrozenSet, Tuple, Dict, Callable
from    dataclasses                     import asdict, replace
from    itertools                       import starmap, chain, filterfalse, islice

from    argparse                        import ArgumentParser
from    operator                        import itemgetter
from    pathlib                         import PurePath, Path
from    random                          import seed
from    json                            import dump, dumps
from    gc                              import collect

import  polars                          as pl
import  polars.selectors                as ps

import  jax.numpy                       as jp

import  torch                           as tr

from    sentence_transformers           import SentenceTransformer

from    ars.configuration.c1__data      import S1Config
from    ars.specification.spec          import DataObject, Recast
from    ars.data.stages_meta            import S1Meta
from    ars.data.features               import RawSchema, FeaturePatterns, FeatureDefinition, FeaturesSpan
from    ars.tools.performance.perf      import benchmark, inject_color_scheme
from    ars.tools.tui.tui               import sprint, ColorSchemeDataScience, Progress
from    ars.tools.tui.tui_data          import ColorSchemeDataScienceSakura
from    ars.tools.visualisations        import viz
from    ars.data.anomalies_injection    import inject_anomalies, InjectionConfig
from    ars.tools.tui.tui               import redirect_native_stderr




def fill_missing_values(df: pl.DataFrame, cfg: S1Config) -> pl.DataFrame:
    filled_cols     = set(chain.from_iterable(cfg.fill_values.values()))
    
    numeric_to_fill = (ps.numeric() - ps.by_name(tuple(filled_cols) + (DataObject.is_anomaly,), require_all = False))
    default_expr    = numeric_to_fill.as_expr().fill_null(cfg.default_fill_value)
    
    is_stat_key     = lambda kv: next(iter(kv)) in ('mean', 'median')
    stat_fill_cols  = filter(is_stat_key, cfg.fill_values.items())
    const_fill_cols = filterfalse(is_stat_key, cfg.fill_values.items())
    
    const_exprs     = starmap(lambda val, cols: pl.col(*cols).fill_null(val), const_fill_cols)
    
    def _build_stat_for_column(col: str, stat_name: str, group_col: None | str) -> pl.Expr:
        match stat_name:
            case 'mean':    base = pl.col(col).mean()
            case 'median':  base = pl.col(col).median()
            case _:         raise NotImplementedError(f'неподдерживаемый тип агрегации: {stat_name}')

        if group_col: base = base.over(group_col)

        return pl.col(col).fill_null(base)
    
    stat_exprs = chain.from_iterable(
        starmap(
            lambda stat_name, cols: map(
                lambda c: _build_stat_for_column(c, stat_name, cfg.fill_group_col.get(c)),
                cols),
            stat_fill_cols))
    
    return df.with_columns(default_expr, *const_exprs, *stat_exprs)


@benchmark('загрузка данных')
def load_spans(cfg: S1Config, raw_schema: RawSchema) -> pl.DataFrame:
    '''загружает данные спанов согласно схеме и конфигурации'''

    MyColorScheme = cfg.output_color_scheme
    MyColorScheme.print_section('ЗАГРУЗКА ДАННЫХ')
    
    sprint(f'загружается файлов: {len(cfg.input_parquet_files)}', style_code = MyColorScheme.perf_metric)    
    def _load(df_file: PurePath) -> pl.DataFrame:
        sprint(f'\t\t{(f := df_file.as_posix())}', style_code = MyColorScheme.debug)
        
        return pl.scan_parquet(f).collect()

    # сырые спаны от aef
    df_asis = pl.concat(map(_load, cfg.input_parquet_files), how = 'vertical')
    df_asis = Recast.overlay(df_asis.lazy(), df_asis.schema).collect() if cfg.recast else df_asis
    MyColorScheme.print_metric('всего сырых записей', df_asis.height)
    
    # если на инференс пришли данные без меток -- ставим заглушки
    if DataObject.label not in df_asis.columns:
        df_asis = df_asis.with_columns(pl.lit(DataObject.class_sentinel).alias(DataObject.label))
    if DataObject.sublabel not in df_asis.columns:
        df_asis = df_asis.with_columns(pl.lit(DataObject.class_sentinel).alias(DataObject.sublabel))
    
    # удаляем колонки, которые не удалось собрать на стадии aef
    df = df_asis.select(pl.exclude(raw_schema.drop_mask))

    # вычисляем метки спанов и упорядочиваем
    label_col       = pl.col(FeaturePatterns.label_col)
    sublabel_col    = pl.col(FeaturePatterns.sublabel_col)
    label_expr      = pl.coalesce(
        pl.when(sublabel_col.is_not_null()).then(sublabel_col.ne('NonAnomaly').cast(pl.Int8)),
        pl.when(label_col.is_not_null()).then(label_col.eq('true').cast(pl.Int8)),)
    label_expr      = (
        pl.when(pl.any_horizontal(
            pl.col(FeaturePatterns.label_col).eq(DataObject.class_sentinel),
            pl.col(FeaturePatterns.sublabel_col).eq(DataObject.class_sentinel),))
        .then(None)
        .otherwise(label_expr))
        
    obj_keys, sort_directions = zip(*FeaturePatterns.objects_order)
    df = df.with_columns(label_expr.alias(DataObject.is_anomaly)).sort(obj_keys, descending = sort_directions)

    # смотрим общую информацию
    MyColorScheme.print_subsection('Состав данных')
    vc_kind = (df.select(pl.col(DataObject.kind).value_counts())
                    .unnest(DataObject.kind)
                    .sort('count', descending = True))
    _ = tuple(starmap(lambda kind, cnt: MyColorScheme.print_metric(
        f'спанов типа \'{kind}\'', f'{cnt} ({cnt / df.height * 100:.1f}%)'),
        zip(
            vc_kind.select(DataObject.kind).to_series().to_list(),
            vc_kind.select('count').to_series().to_list())))
    
    vc_label = (df.select(pl.col(DataObject.is_anomaly).value_counts())
                    .unnest(DataObject.is_anomaly)
                    .sort('count', descending = True))
    
    real_labels = vc_label.filter(pl.col(DataObject.is_anomaly).is_not_null())
    dummy_count = vc_label.filter(pl.col(DataObject.is_anomaly).is_null())['count'].sum()

    if real_labels.height > 0:
        real_anom = real_labels.filter(pl.col(DataObject.is_anomaly) == 1).select('count').sum().item()
        if real_anom == 0:
            sprint('в данных отсутствуют аномалии', style_code = MyColorScheme.warning)
        _ = tuple(starmap(lambda lbl, cnt: MyColorScheme.print_metric(
            f'метка "{'аномалия' if lbl else 'норма'}"',
            f'{cnt} ({cnt / df.height * 100:.1f}%)'),
            zip(real_labels[DataObject.is_anomaly].to_list(), real_labels['count'].to_list())))
    else:
        sprint('реальные метки отсутствуют', style_code = MyColorScheme.warning)

    if dummy_count > 0:
        MyColorScheme.print_metric(
            'заглушки (без меток)',
            f'{dummy_count} ({dummy_count / df.height * 100:.1f}%)')
    
    if FeaturePatterns.sublabel_col in df.columns:
        MyColorScheme.print_subsection('Виды аномалий')
        anom_stats = (df
            .filter(pl.col(FeaturePatterns.sublabel_col) != DataObject.class_sentinel)
            .group_by(sublabel_col)
            .agg(pl.len().alias('count'))
            .sort(('count', sublabel_col), descending = (True, False)))
        _ = list(starmap(lambda an_type, cnt: MyColorScheme.print_metric(
            f'{an_type or "NonAnomaly"}',
            f'{cnt} ({cnt / df.height * 100:.1f}%)'),
            zip(
                anom_stats.select(sublabel_col).to_series().to_list(),
                anom_stats.select('count').to_series().to_list())))
    else: sprint('в данных отсутствуют виды аномалий', style_code = MyColorScheme.warning)
    
    sprint(f'\nуникальных трасс   (trace_id): {df.n_unique('trace_id')}', style_code = MyColorScheme.good_metric)
    sprint(f'уникальных агентов (agent_id): {df.n_unique('agent_id')}', style_code = MyColorScheme.good_metric)
    
    sprint('длительность спана:', df
            .select(((pl.col('end_time_ns') - pl.col('start_time_ns'))
                        .truediv(pl.lit(cfg.duration_scale_to_sec))).alias('dur_sec'))
            .select(
                pl.col('dur_sec').median().round(4).alias('median'),
                pl.col('dur_sec').mean().round(4).alias('mean'),
                pl.col('dur_sec').quantile(0.95).round(4).alias('q95'))
            .select(
                pl.format(
                    'медиана: {} сек, среднее: {} сек, 95-й перцентиль: {} сек',
                    pl.col('median'),
                    pl.col('mean'),
                    pl.col('q95')
                )).item(), style_code = MyColorScheme.info)

    return df #.head(100)


@benchmark('отбор признаков')
def select_features(df: pl.DataFrame, cfg: S1Config, protected: FrozenSet[str],
                    stats_df: None | pl.DataFrame = None) -> pl.DataFrame:
    '''отбор (числовых) признаков; статистики отбора считаются на stats_df
    (train-normal спаны, F-11), а прореживаются колонки всего df'''

    MyColorScheme = cfg.output_color_scheme
    MyColorScheme.print_section('ОТБОР ПРИЗНАКОВ')

    stats_df = df if stats_df is None else stats_df
    candidate_cols = sorted(
        df.drop(pl.selectors.by_name(protected, require_all = False), ~pl.selectors.numeric()).columns)

    sprint(f'Кандидатов для очистки: {(n_initial := len(candidate_cols))} '
           f'(статистики на {stats_df.height} спанах)', style_code = MyColorScheme.info)
    if n_initial == 0: return df

    fill_exprs      = map(
        lambda c: (pl.lit(1.0).sub(pl.col(c).null_count().truediv(pl.lit(stats_df.height)))).alias(f'{c}_fill'),
        candidate_cols)
    
    static_exprs    = map(
        lambda c: (pl.col(c).value_counts(sort = True).head(1).struct.field('count').first()
                        .truediv(pl.lit(stats_df.height))).fill_null(0.0).alias(f'{c}_static'),
        candidate_cols)

    df_fill     = stats_df.select(fill_exprs)
    df_static   = stats_df.select(static_exprs)

    fill_long   = df_fill.unpivot().rename({'variable': 'column', 'value': 'fill_rate'})
    static_long = df_static.unpivot().rename({'variable': 'column', 'value': 'static_rate'})

    fill_long   = fill_long.with_columns(pl.col('column').str.replace('_fill', ''))
    static_long = static_long.with_columns(pl.col('column').str.replace('_static', ''))

    quality     = fill_long.join(static_long, on = 'column', how = 'inner')

    passed      = quality.filter(
        (pl.col('fill_rate')    >= pl.lit(cfg.min_fill_rate)) &
        (pl.col('static_rate')  <= pl.lit(cfg.max_static_rate)))
    
    cols_after  = sorted(passed['column'].to_list())
    sprint(
        f'После фильтрации заполненности/статичности: {(n_after := len(cols_after))} (удалено {n_initial - n_after})',
        style_code = MyColorScheme.info)
    
    corr_df = stats_df.select(cols_after).corr()

    _viz_dir = cfg.output_dir / 'visualizations'
    #viz.save_correlation(corr_df, tuple(cols_after), _viz_dir, 'feature_correlation', 'Корреляции отобранных признаков')

    corr_df_with_index = corr_df.with_columns(pl.Series('feature', cols_after))
    corr_long   = (
        corr_df_with_index
        .unpivot(index = 'feature', variable_name = 'neighbor', value_name = 'correlation')
        .filter(pl.col('feature') < pl.col('neighbor'))
        .filter(pl.col('correlation').abs() > pl.lit(cfg.max_correlation))
        .with_columns(pl.col('correlation').abs().alias('abs_corr')))
    
    strong_sums = (
        pl.concat((
            corr_long.select(pl.col('feature').alias('col'),    pl.col('abs_corr')),
            corr_long.select(pl.col('neighbor').alias('col'),   pl.col('abs_corr'))))
        .group_by('col')
        .agg(pl.col('abs_corr').sum().alias('strong_corr_sum')))
    
    pairs_with_strength = (corr_long
        .join(strong_sums, left_on = 'feature', right_on = 'col', how = 'left')
        .rename({'strong_corr_sum': 'sum1'})
        .join(strong_sums.select(
                pl.col('col').alias('neighbor'),
                pl.col('strong_corr_sum').alias('sum2')),
            on = 'neighbor', how = 'left'))
    
    to_drop = (pairs_with_strength
        .select(
            pl.when(
                (pl.col('sum1') > pl.col('sum2')) |
                ((pl.col('sum1') == pl.col('sum2')) & (pl.col('feature') > pl.col('neighbor'))))
            .then(pl.col('feature'))
            .otherwise(pl.col('neighbor'))
            .alias('to_drop'))
        .to_series()
        .unique()
        .to_list())

    cols_final = sorted(filterfalse(lambda c: c in to_drop, cols_after))

    sprint(
        f'После удаления корреляций: {(n_final := len(cols_final))} признаков (удалено {n_after - n_final})',
        style_code = MyColorScheme.info)
    
    mean_fill = (stats_df.select(
        pl.col(cols_final).null_count().truediv(pl.lit(stats_df.height)))
        .mean_horizontal()
        .item()
    ) if cols_final else None
    sprint(
        f'Средняя заполненность после очистки: {round(1.0 - mean_fill, 2) if mean_fill is not None else "нет признаков"}',
        style_code = MyColorScheme.info)
    
    return df.select(pl.selectors.by_name(protected, require_all = False) | pl.selectors.by_name(cols_final))


@benchmark('вычисление epi-признаков')
def calculate_features(
    df              : pl.DataFrame,
    cfg             : S1Config,
    feature_cfg     : FeaturePatterns,
    raw_schema      : RawSchema,
    selection_ids   : None | pl.DataFrame = None,
) -> Tuple[pl.DataFrame, Tuple[str, ...]]:
    MyColorScheme = cfg.output_color_scheme
    MyColorScheme.print_section('ВЫЧИСЛЕНИЕ EPI-ПРИЗНАКОВ')

    # 1. генерация признаков и их агрегаций
    features_obj    = FeaturesSpan()
    spans_enriched  = features_obj.make_features(df, feature_cfg)
    
    # метки и все нерелевантные epi колонки
    marks       = frozenset(chain(
        (feature_cfg.label_col, feature_cfg.sublabel_col),
        starmap(lambda obj, ord: obj, feature_cfg.objects_order)
    )) | {DataObject.is_anomaly}
    protected   = (
        marks |
        frozenset(raw_schema.schema.keys()) |
        frozenset({'sem_text', 'agent_prompt'}))
    
    # 2. общая статистика
    all_feature_defs    = frozenset(filter(
        lambda v: isinstance(v, FeatureDefinition),
        vars(features_obj).values()))
    
    local_names         = frozenset(map(
        lambda fd: fd.name,
        filter(lambda fd: fd.include_in_sequence, all_feature_defs)))
    not_aggs            = frozenset(chain(local_names, marks, raw_schema.schema.keys()))
    agg_names           = frozenset(filterfalse(
        lambda c: c in not_aggs,
        spans_enriched.columns))
    
    features_iter   = frozenset(chain(local_names, agg_names))
    len_features    = sum(map(lambda _: 1, features_iter))
    len_locals      = sum(map(lambda _: 1, local_names))
    len_aggs        = sum(map(lambda _: 1, agg_names))

    sprint(f'Всего признаков: {len_features}', style_code = MyColorScheme.info)
    MyColorScheme.print_inline_tuple(
        f'локальные и динамические признаки ({len_locals})', tuple(local_names))
    MyColorScheme.print_inline_tuple(
        f'глобальные статистики ({len_aggs})',
        tuple(islice(
                agg_names,
                min(cfg.output_max_collection_len if cfg.output_max_collection_len is not None else float('inf'), #type: ignore
                    len_aggs))) + (('...',) if cfg.output_max_collection_len is not None else tuple()))
    
    # 3. статистика пропусков
    missing_info = (
        spans_enriched
        .select(pl.col(features_iter))
        .null_count()
        .unpivot()
        .rename({'variable': 'col', 'value': 'nulls'})
        .filter(pl.col('nulls') > 0)
        .with_columns((pl.col('nulls')
            .truediv(pl.lit(spans_enriched.height)).mul(100)
            .round(2)).alias('pct')))
    total_missing = missing_info['nulls'].sum() if missing_info.height > 0 else 0

    if total_missing > 0:
        #missings_df = missing_info.select(
        #    pl.col('col').alias('Признак'),
        #    pl.col('nulls').alias('Пропусков'),
        #    pl.col('pct').alias('Процент'),
        #    pl.lit(str(cfg.default_fill_value)).alias('Заполнено'))
        #MyColorScheme.print_missings_table('Пропуски в признаках', missings_df)
        MyColorScheme.print_metric('Всего пропущенных значений', total_missing)
        fill_rate = (1.0 - total_missing / (len_features * spans_enriched.height)) * 100
        MyColorScheme.print_metric('Процент заполненности данных', f'{fill_rate:.2f}%')
    else:
        MyColorScheme.print_metric('Пропусков', 'нет')
        MyColorScheme.print_metric('Процент заполненности данных', '100.00%')
    
    if missing_info.height > 0:
        _viz_dir = cfg.output_dir / 'visualizations'
        _ = viz.save_bars(missing_info, 'col', 'pct', _viz_dir, 'features_missingness', 'Пропуски признаков, %')
    
    # 4. удаление (около)пустых, (около)констант, сильно коррелированных, неинформативных колонок
    #    (F-11: статистики отбора — только на train-normal трассах, если заданы)
    stats_scope = (spans_enriched.join(selection_ids, on = DataObject.trace_id, how = 'semi')
                   if selection_ids is not None else None)
    refined = select_features(spans_enriched, cfg, protected, stats_df = stats_scope)
    
    # 5. определение итогового набора EPI-признаков (все числовые колонки, исключая служебные и сырые)
    epi_candidate_cols = (
        refined.select(pl.selectors.numeric().exclude(marks | raw_schema.schema.keys()))
        .columns)
    if not epi_candidate_cols:
        raise ValueError(
            'Не осталось числовых признаков после очистки. '
            'Уменьшите строгость порогов (min_fill_rate, max_static_rate, max_correlation).')
    
    # 6. заполнение пропусков в оставшихся колонках
    filled = fill_missing_values(refined, cfg)
    
    # 7. фильтрация признаков с аномально большими значениями
    max_abs_threshold = cfg.max_abs_feature  # F-11: и этот фильтр — на train-normal статистиках
    abs_scope = (filled.join(selection_ids, on = DataObject.trace_id, how = 'semi')
                 if selection_ids is not None else filled)
    suspicious = (abs_scope
        .select(pl.col(epi_candidate_cols).abs().max())
        .unpivot()
        .filter(pl.col('value') > pl.lit(max_abs_threshold))
        .select('variable')
        .to_series()
        .to_list())
    epi_feature_names = filterfalse(lambda c: c in suspicious, epi_candidate_cols)
    _ = sprint(f'Удалены астрономические признаки: {suspicious}', style_code = MyColorScheme.debug) if suspicious else None
    
    # 8. формирование epi_sequence и финальная статистика
    epi_feature_names       = tuple(sorted(epi_feature_names))
    len_epi_feature_names   = len(epi_feature_names)
    result = filled.with_columns(
        pl.concat_list(epi_feature_names).alias('epi_vector'))
    
    stats = (
        ('Всего признаков сгенерировано',   len_features),
        ('Финальное число EPI-признаков',   len_epi_feature_names),
        ('Признаков подвергнуто log1p',     sum(map(
            lambda _: 1,
            filter(lambda fd: fd.log_transform and (fd.name in epi_feature_names), all_feature_defs))))
    )
    stats_df = pl.DataFrame(stats, orient = 'row', schema = ('Параметр', 'Значение'))
    MyColorScheme.print_stats_table(
        'Статистика EPI-признаков',
        stats_df.with_columns(pl.col('Значение').cast(pl.Utf8)))
    MyColorScheme.print_metric('Размерность EPI', result.select(pl.col('epi_vector').list.len().unique()).item())
    sprint(f'итоговый набор epi: {epi_feature_names}', style_code = MyColorScheme.debug)

    return result, epi_feature_names


@benchmark('вычисление sem-признаков')
def compute_semantic_embeddings(
    df       : pl.DataFrame,
    cfg      : S1Config,
    text_col : str,
    out_col  : str,
    embed    : 'None | Callable[[tuple[str, ...]], jp.ndarray]' = None,
) -> pl.DataFrame:
    """F-01: РЕАЛЬНЫЕ эмбеддинги для каждого спана (легаси возвращал нулевые
    заглушки, из-за чего семантическая ветвь детектировала факт инъекции, а не
    семантику). embed — общая функция с инъектором (make_embedder), чтобы
    нормальные и аномальные спаны кодировались одной моделью."""
    MyColorScheme = cfg.output_color_scheme
    MyColorScheme.print_section(f'ЭМБЕДДИНГИ: {text_col} -> {out_col}')
    sprint(f'Модель: {cfg.embedder_path.name}, устройство: {cfg.device}', style_code = MyColorScheme.info)

    texts = df[text_col].cast(pl.Utf8).fill_null('').to_list()
    MyColorScheme.print_metric('Всего текстов', len(texts))

    if embed is None:
        embed = make_embedder(cfg)

    chunk       = max(1, cfg.embedding_batch_size) * 32
    parts       = tuple(map(
        lambda i: embed(tuple(texts[i:i + chunk])),
        range(0, len(texts), chunk)))
    import numpy as _np
    embeddings  = _np.concatenate(tuple(map(_np.asarray, parts)), axis = 0) if parts else _np.zeros((0, 1024))

    if len(embeddings) != df.height:
        raise ValueError('количество эмбеддингов не совпадает с числом спанов')

    df = df.with_columns(pl.Series(out_col, embeddings.tolist()))
    MyColorScheme.print_metric(f'Размерность SEM[{out_col}]', embeddings.shape[1] if len(embeddings) else 0)

    return df




def make_embedder(cfg: S1Config) -> Callable[[tuple[str, ...]], jp.ndarray]:
    model = SentenceTransformer(
        cfg.embedder_path.as_posix(),
        device              = cfg.device,
        local_files_only    = True
    )

    def embed(texts: tuple[str, ...]) -> jp.ndarray:
        emb = model.encode(
            texts,
            batch_size              = cfg.embedding_batch_size,
            normalize_embeddings    = True,
            convert_to_tensor       = False
        )

        return jp.asarray(emb)

    return embed


@benchmark('сборка трасс')
def build_traces(df: pl.DataFrame, sem_vec_names: Tuple[str, ...], cfg: S1Config) -> pl.DataFrame:
    MyColorScheme = cfg.output_color_scheme
    MyColorScheme.print_section('ГРУППИРОВКА В ТРАССЫ')
    
    agg_exprs = (
        pl.col('epi_vector').alias('epi_sequence'),
        pl.col(sem_vec_names).name.prefix('sem_sequence_'),
        pl.col(DataObject.is_anomaly).max().alias(DataObject.is_anomaly),
        pl.col(DataObject.sublabel).drop_nulls().first().alias(DataObject.sublabel),
    )
    obj_keys, sort_directions = zip(*FeaturePatterns.objects_order)
    traces = (df
        .sort(obj_keys, descending = sort_directions)
        .group_by('trace_id', 'agent_id').agg(agg_exprs).sort('trace_id'))
    
    MyColorScheme.print_metric('Получено трасс', traces.height)
    seq_lens = traces['epi_sequence'].list.len()

    assert seq_lens.min() > 0, f'обнаружены пустые последовательности!' #type: ignore

    MyColorScheme.print_metric(
        'Длина последовательности (спанов)',
        f'мин={seq_lens.min()}, макс={seq_lens.max()}, медиана={seq_lens.median():.1f}, среднее={seq_lens.mean():.1f}')
    

    extract_vec_len = pl.element().list.len().unique().item()
    get_unique_lens = lambda c: pl.col(c).list.agg(extract_vec_len).unique().item().name.prefix('dim_')
    epi_dims = traces.select(get_unique_lens('epi_sequence'))
    sem_dims = traces.select(get_unique_lens('^sem_sequence_.+$'))
    MyColorScheme.print_stats_table(
        'Размерности векторов',
        pl.concat((epi_dims, sem_dims), how = 'horizontal').transpose(
            include_header = True, header_name = 'Параметр', column_names = ('Значение',)))
    
    """
    MyColorScheme.print_subsection('Распределение меток в трассах')
    label_table = (traces
        .select(pl.col(DataObject.is_anomaly).value_counts())
        .unnest(DataObject.is_anomaly)
        .sort(DataObject.is_anomaly)
        .with_columns(
            pl.col('count').truediv(pl.lit(traces.height)).mul(100).round(1).alias('доля'),
            pl.when(pl.col(DataObject.is_anomaly) == 0).then(pl.lit('норма')).otherwise(pl.lit('аномалия')).alias('метка'))
        .select(
            pl.col('метка'),
            pl.col('count').alias('количество'),
            pl.col('доля')))
    _ = tuple(map(
        lambda r: MyColorScheme.print_metric(r[0], f'{r[1]} ({r[2]:>4} %)'),
        label_table.iter_rows()))

    if DataObject.sublabel in traces.columns:
        MyColorScheme.print_subsection('Типы аномалий в трассах')
        anom_table = (traces
            .select(pl.col(DataObject.sublabel).value_counts())
            .unnest(DataObject.sublabel)
            .sort(('count', DataObject.sublabel), descending = (True, False))
            .with_columns(
                pl.col('count').truediv(pl.lit(traces.height)).mul(100).round(1).alias('доля'),
                pl.col(DataObject.sublabel).fill_null(pl.lit('NonAnomaly')).alias('тип'))
            .select(
                pl.col('тип'),
                pl.col('count').alias('количество'),
                pl.col('доля')))
        _ = tuple(map(
            lambda r: MyColorScheme.print_metric(r[0], f'{r[1]} ({r[2]:>4} %)'),
            anom_table.iter_rows()))
    """

    MyColorScheme.print_subsection('Распределение меток в трассах')

    label_cnt = (traces
        .select(pl.col(DataObject.is_anomaly).value_counts())
        .unnest(DataObject.is_anomaly)
        .sort(DataObject.is_anomaly, nulls_last = True))

    real_labels = label_cnt.filter(pl.col(DataObject.is_anomaly).is_not_null())
    dummy_count = label_cnt.filter(pl.col(DataObject.is_anomaly).is_null()).select('count').sum().item()

    if real_labels.height > 0:
        _ = tuple(starmap(
            lambda lbl, cnt: MyColorScheme.print_metric(
                f'метка "{'аномалия' if lbl else 'норма'}"',
                f'{cnt} ({cnt / traces.height * 100:.1f}%)'),
            zip(real_labels[DataObject.is_anomaly].to_list(), real_labels['count'].to_list())))
    else: sprint('реальные метки (норма/аномалия) отсутствуют', style_code = MyColorScheme.warning)

    if dummy_count > 0:
        MyColorScheme.print_metric(
            'заглушки (без меток)',
            f'{dummy_count} ({dummy_count / traces.height * 100:.1f}%)')

    if DataObject.sublabel in traces.columns:
        MyColorScheme.print_subsection('Типы аномалий в трассах')

        anom_table = (traces
            .filter(pl.col(DataObject.sublabel) != DataObject.class_sentinel)
            .select(pl.col(DataObject.sublabel).value_counts())
            .unnest(DataObject.sublabel)
            .sort(('count', DataObject.sublabel), descending = (True, False))
            .with_columns(
                (pl.col('count') / traces.height * 100).round(1).alias('доля'),
                pl.col(DataObject.sublabel).fill_null('NonAnomaly').alias('тип'))
            .select('тип', 'count', 'доля'))

        if anom_table.height > 0:
            _ = tuple(starmap(
                lambda an_type, cnt, доля: MyColorScheme.print_metric(
                    an_type,
                    f'{cnt} ({доля:>4} %)'),
                zip(anom_table['тип'].to_list(),
                    anom_table['count'].to_list(),
                    anom_table['доля'].to_list())))
        else: sprint('типы аномалий отсутствуют (возможно только заглушки)', style_code = MyColorScheme.warning)
    
    return traces



def split_trace_ids(units: pl.DataFrame, strata_col: str, cfg: S1Config):
    """Deterministic trace-level split (F-08/F-11): `units` has one row per
    trace_id with its (planned or actual) anomaly class. Returns id-frames
    (norm_train, norm_val, norm_test, anom_val, anom_test). Because injection
    labels are a pure hash of trace_id, this can run BEFORE features/injection
    and will agree with the post-injection split."""
    is_normal   = (
        pl.col(strata_col).is_null() |
        pl.col(strata_col).cast(pl.Utf8).str.to_lowercase().is_in(('', 'nonanomaly')))
    units       = units.sort(DataObject.trace_id)
    norm_units  = units.filter(is_normal)
    anom_units  = units.filter(~is_normal)

    norm_shuffled   = norm_units.sample(fraction = 1.0, shuffle = True, seed = cfg.seed_split)
    n_norm          = norm_shuffled.height
    n_train, n_val  = tuple(map(
        lambda r: int(n_norm * r),
        (cfg.norm_train_ratio, cfg.norm_val_ratio)))
    n_test          = n_norm - n_train - n_val
    assert n_train + n_val + n_test == n_norm, 'некорректный сплит нормальных объектов'

    ids             = lambda part: part.select(DataObject.trace_id)
    norm_train_ids  = ids(norm_shuffled.slice(0, n_train))
    norm_val_ids    = ids(norm_shuffled.slice(n_train,  n_val))
    norm_test_ids   = ids(norm_shuffled.slice(n_train + n_val, n_test))

    def _split_anom_class(class_name: str):
        group_shuffled  = (anom_units
            .filter(pl.col(strata_col) == class_name)
            .sample(fraction = 1.0, shuffle = True, seed = cfg.seed_split))
        n_val_anom      = int(group_shuffled.height * cfg.anom_val_ratio)
        return (ids(group_shuffled.slice(0, n_val_anom)),
                ids(group_shuffled.slice(n_val_anom)))

    class_names     = sorted(anom_units[strata_col].unique().to_list())
    anom_pairs      = tuple(map(_split_anom_class, class_names))
    empty_ids       = units.head(0).select(DataObject.trace_id)
    anom_val_ids    = pl.concat(tuple(map(itemgetter(0), anom_pairs))) if anom_pairs else empty_ids
    anom_test_ids   = pl.concat(tuple(map(itemgetter(1), anom_pairs))) if anom_pairs else empty_ids

    return norm_train_ids, norm_val_ids, norm_test_ids, anom_val_ids, anom_test_ids


@benchmark('разделение выборок')
def stratified_split(df: pl.DataFrame, strata_col: str, cfg: S1Config):
    MyColorScheme = cfg.output_color_scheme
    MyColorScheme.print_section('РАЗБИЕНИЕ НА ВЫБОРКИ')
    sprint(f'Стратификация по колонке: {strata_col}', style_code = MyColorScheme.info)
    sprint('Нормальные трассы >> случайно в train/val/test по norm_*_ratio', style_code = MyColorScheme.debug)
    sprint('Аномальные трассы >> только в val/test стратифицированно по anom_*_ratio', style_code = MyColorScheme.debug)
    
    df = df.filter(pl.col(strata_col) != DataObject.class_sentinel)

    units = (df
        .group_by(DataObject.trace_id)
        .agg(pl.col(strata_col).drop_nulls().first().alias(strata_col))
        .sort(DataObject.trace_id))
    norm_train_ids, norm_val_ids, norm_test_ids, anom_val_ids, anom_test_ids = split_trace_ids(units, strata_col, cfg)

    member  = lambda id_frame: df.join(id_frame, on = DataObject.trace_id, how = 'semi')
    train   = member(norm_train_ids)
    val     = pl.concat((member(norm_val_ids),  member(anom_val_ids)))
    test    = pl.concat((member(norm_test_ids), member(anom_test_ids)))

    MyColorScheme.print_metric('Train (обучение реконструкции, только нормальные)', train.height)
    MyColorScheme.print_metric('Val   (подбор порога)\t\t\t\t', val.height)
    MyColorScheme.print_metric('Test  (выбор модели)\t\t\t\t', test.height)

    def _describe_split(name: str, subdf: pl.DataFrame):
        norm    = subdf.filter(pl.col(DataObject.is_anomaly) == 0).height
        anom    = subdf.height - norm
        ratio   = round(anom / subdf.height * 100 if subdf.height > 0 else 0, 1)
        ds      = len(str(max(norm, anom))) + 1
        sprint(
            f'\t{name}\t: норма ={norm:>{ds}}, аномалия ={anom:>{ds}} (anom ratio={ratio:>5}%)',
            style_code = MyColorScheme.info)

    _describe_split('Train', train)
    _describe_split('Val',   val)
    _describe_split('Test',  test)

    _viz_dir = cfg.output_dir / 'visualizations'
    _pairs   = (('Train', train), ('Val', val), ('Test', test))
    _balance = pl.concat(map(
        lambda nd: nd[1].select(
                            pl.lit(nd[0]).alias('split'),
                            pl.when(pl.col(DataObject.is_anomaly) == 0).then(pl.lit('норма')).otherwise(pl.lit('аномалия')).alias('class'))
                        .group_by('split', 'class').agg(pl.len().alias('count')),
        _pairs))
    viz.save_grouped_bars(_balance, 'split', 'count', 'class', _viz_dir, 'split_balance', 'Баланс классов по выборкам')

    return train, val, test


def _apply_norm(
    df: pl.DataFrame, epi_dim: int, shift_arr: pl.Expr, scale_arr: pl.Expr,
    z_clip: float = 0.0,
) -> pl.DataFrame:
    # F-05: normalized values are clipped to [-z_clip, z_clip] (scaling is
    # monotone, so this equals clipping raw values at shift ± z_clip*scale).
    # Applied identically at train/val/test/inference; z_clip <= 0 preserves the
    # legacy unbounded behavior.
    normed = pl.element().list.to_array(epi_dim).sub(shift_arr).truediv(scale_arr).arr.to_list()
    if z_clip > 0.0:
        normed = normed.list.eval(pl.element().clip(-z_clip, z_clip))
    return df.with_columns(pl.col('epi_sequence').list.eval(normed)).sort('trace_id')


@benchmark('нормализация epi-признаков')
def normalize_epi_features(
    train_df    : pl.DataFrame,
    val_df      : pl.DataFrame,
    test_df     : pl.DataFrame,
    cfg         : S1Config,
) -> Tuple[
    Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame],
    Dict[str, Literal['zscore', 'robust'] | bool | float | Tuple[float, float]],
]:
    MyColorScheme = cfg.output_color_scheme
    MyColorScheme.print_section('НОРМАЛИЗАЦИЯ EPI-ПРИЗНАКОВ')
    
    train_norm = train_df.filter(pl.col(DataObject.is_anomaly) == 0)
    if train_norm.height == 0:
        raise ValueError('В тренировочной выборке нет нормальных трасс')
    MyColorScheme.print_metric('Нормальных трасс для расчёта параметров', train_norm.height)
    
    extract_vec_len = pl.element().list.len().unique().item()
    get_unique_lens = lambda c: pl.col(c).list.agg(extract_vec_len).unique().item().name.prefix('dim_')
    epi_dim = train_norm.select(get_unique_lens('epi_sequence')).item()
    
    flat = train_norm.select(pl.col('epi_sequence')
            .explode()
            .list.to_array(epi_dim)
            .arr.to_struct()
            .struct.unnest())
    flat = flat.select(sorted(flat.columns, key = lambda c: int(c.rsplit('_', 1)[-1])))
    
    if cfg.winsorize_epi:
        flat = flat.with_columns(
            pl.all().clip(
                pl.all().quantile(cfg.winsorize_limits[0]),
                pl.all().quantile(cfg.winsorize_limits[1])))
    
    match cfg.epi_normalization:
        case 'robust':
            shift_expr = pl.all().median()
            # todo: вынести значения квантилей в конфиг
            scale_expr = pl.all().quantile(0.75) - pl.all().quantile(0.25)
        case 'zscore':
            shift_expr = pl.all().mean()
            scale_expr = pl.all().std(ddof = 1)
        case _:
            raise NotImplementedError(f'неподдерживаемый метод нормализации: {cfg.epi_normalization}')

    # F-05: a scale barely above eps_normalization produced |z| ~ 40+ on
    # quantile-degenerate features; floor the scale outright. A near-zero
    # spread means the feature is near-constant on normal data — deviations
    # are still visible, just not explosively amplified.
    floor_value = 1.0 if cfg.scale_floor <= 0.0 else cfg.scale_floor
    scale_expr  = (
        pl.when(scale_expr.abs() < max(cfg.eps_normalization, cfg.scale_floor))
            .then(scale_expr.mul(0.0).add(floor_value))   # keeps per-column names
            .otherwise(scale_expr))
    shift_row   = flat.select(shift_expr).row(0)
    scale_row   = flat.select(scale_expr).row(0)
    shift_arr   = pl.lit(shift_row, dtype = pl.Array(pl.Float64, epi_dim))
    scale_arr   = pl.lit(scale_row, dtype = pl.Array(pl.Float64, epi_dim))

    train_norm_out  = _apply_norm(train_df, epi_dim, shift_arr, scale_arr, cfg.norm_z_clip)
    val_norm_out    = _apply_norm(val_df,   epi_dim, shift_arr, scale_arr, cfg.norm_z_clip)
    test_norm_out   = _apply_norm(test_df,  epi_dim, shift_arr, scale_arr, cfg.norm_z_clip)

    norm_params = {
        'method'            : cfg.epi_normalization,
        'shift'             : shift_row,
        'scale'             : scale_row,
        'winsorize'         : cfg.winsorize_epi,
        'winsorize_limits'  : cfg.winsorize_limits if cfg.winsorize_epi else None,
        'scale_floor'       : cfg.scale_floor,
        'z_clip'            : cfg.norm_z_clip,
    }

    _viz_dir = cfg.output_dir / 'visualizations'
    _dims    = map(lambda i: f'd{i}', range(epi_dim))
    _norm    = pl.DataFrame({'dim': _dims, 'shift': shift_row, 'scale': scale_row})
    viz.save_grouped_bars(
        _norm.unpivot(index = 'dim', on = ['shift', 'scale'], variable_name = 'param', value_name = 'value'),
        'dim', 'value', 'param', _viz_dir, 'norm_params', 'Параметры нормализации по измерениям')

    return (train_norm_out, val_norm_out, test_norm_out), norm_params


@benchmark('сохранение датасетов')
def save_datasets(
    epi_features    : Tuple[str, ...],
    train_df        : pl.DataFrame,
    val_df          : pl.DataFrame,
    test_df         : pl.DataFrame,
    norm_params     : Dict['str', Literal['zscore', 'robust'] | bool | float | Tuple[float, float]],
    cfg             : S1Config,
    run_id          : str,
) -> S1Meta:
    MyColorScheme = cfg.output_color_scheme

    def _build_filename(name: str) -> str:
        return f'{cfg.output_prefix}_{run_id}_{name}' if run_id else f'{cfg.output_prefix}_{name}'
    
    Path(cfg.output_dir).mkdir(parents = True, exist_ok = True)
    
    datasets = (('train', train_df), ('val', val_df), ('test', test_df))
    _ = tuple(starmap(
        lambda name, df: (
            df.write_parquet(
                (path := cfg.output_dir / _build_filename(f'{name}.parquet')).as_posix()),
            MyColorScheme.print_metric(name.capitalize(), path.as_posix()),),
        datasets))

    sample_epi      = train_df['epi_sequence'][0][0]
    epi_dim         = len(sample_epi)
    seq_cols        = tuple(filter(lambda c: c.startswith('sem_sequence_'), train_df.columns))
    sem_vecs        = dict(map(
        lambda c: (c.replace('_sequence', ''), len(train_df[c][0][0])),
        seq_cols))
    anomaly_types   = (pl.concat((val_df.select(DataObject.sublabel), test_df.select(DataObject.sublabel)))
        .filter(
            pl.col(DataObject.sublabel).is_not_null() &
            (pl.col(DataObject.sublabel).str.to_lowercase() != 'nonanomaly'))
        .unique()
        .to_series()
        .to_list())
    
    def _label_count(df: pl.DataFrame, label: int) -> int:
        return df.filter(pl.col(DataObject.is_anomaly) == label).height

    train_normal = _label_count(train_df, 0)
    train_anom   = _label_count(train_df, 1)
    val_normal   = _label_count(val_df,   0)
    val_anom     = _label_count(val_df,   1)
    test_normal  = _label_count(test_df,  0)
    test_anom    = _label_count(test_df,  1)
    
    meta = S1Meta(
        prefix              = cfg.output_prefix,
        run_id              = run_id,
        output_dir          = cfg.output_dir.as_posix(),
        raw_files           = tuple(map(PurePath.as_posix, cfg.input_parquet_files)),
        seed_random         = cfg.seed_random,
        seed_polars         = cfg.seed_polars,
        seed_torch          = cfg.seed_torch,
        seed_split          = cfg.seed_split,
        seed_synth          = cfg.seed_synth,
        seed_llm            = cfg.seed_llm,
        embedding_model     = cfg.embedder_path.as_posix(),
        embedding_fingerprint = _embedder_fingerprint_or_none(cfg),
        epi_dim             = epi_dim,
        epi_features        = epi_features,
        epi_normalization   = norm_params,
        anomaly_types       = tuple(sorted(anomaly_types)),
        semantic_vectors    = sem_vecs,
        #synth_anom_injected = {
        #    'enabled'       : cfg.synth_anom_inject,
        #    'val_ratio'     : cfg.synth_anom_val_ratio,
        #    'test_ratio'    : cfg.synth_anom_test_ratio,
        #    'basis'         : cfg.synth_anom_stats_basis,
        #    'counts'        : dict(map(
        #        lambda m: (m, getattr(cfg, f'synth_anom_{m}_count')),
        #        ('outlier', 'mutate', 'extreme'))),
        #},
        split_config = {
            'norm_train_ratio'  : cfg.norm_train_ratio,
            'norm_val_ratio'    : cfg.norm_val_ratio,
            #'norm_test_ratio'   : cfg.norm_test_ratio,
            'anom_val_ratio'    : cfg.anom_val_ratio,
            'anom_test_ratio'   : cfg.anom_test_ratio,
        },
        train_samples       = train_df.height,
        val_samples         = val_df.height,
        test_samples        = test_df.height,
        train_normal_count  = train_normal,
        train_anomaly_count = train_anom,
        val_normal_count    = val_normal,
        val_anomaly_count   = val_anom,
        test_normal_count   = test_normal,
        test_anomaly_count  = test_anom)
    
    meta_path = cfg.output_dir / _build_filename('meta.json')
    with open(meta_path, 'w') as f:
        dump(asdict(meta), f, indent = 4, ensure_ascii = False)
    MyColorScheme.print_debug(dumps(asdict(meta), indent = 4, ensure_ascii = False))
    
    if cfg.samples_dir is not None:
        Path(cfg.samples_dir).mkdir(parents = True, exist_ok = True)
        _ = tuple(starmap(
            lambda name, df: (dump(
                df.head(10).to_dicts(),
                f := open(
                    cfg.samples_dir / #type: ignore
                    _build_filename(f'{name}_sample.json'),
                    'w', encoding = 'utf-8'),
                indent          = 4,
                ensure_ascii    = False), f.close()),
            datasets))

    return meta


def _embedder_fingerprint_or_none(cfg: S1Config) -> None | str:
    from ars.tools.utilities.fingerprint import model_fingerprint
    try:
        return model_fingerprint(cfg.embedder_path)
    except FileNotFoundError:
        return None


def _set_seeds(cfg: S1Config, mcs: None | ColorSchemeDataScience) -> None:
    sprint(f'random seed <general> : {cfg.seed_random}',    style_code = mcs.debug if mcs else '')
    seed(cfg.seed_random)

    sprint(f'random seed <torch>   : {cfg.seed_torch}',     style_code = mcs.debug if mcs else '')
    tr.manual_seed(cfg.seed_torch)
    if tr.cuda.is_available(): tr.cuda.manual_seed_all(cfg.seed_torch)
    
    sprint(f'random seed <polars>  : {cfg.seed_polars}',    style_code = mcs.debug if mcs else '')
    pl.set_random_seed(cfg.seed_polars)
    
    return


@benchmark('этап 1: подготовка данных')
def main(
    cfg     : S1Config,
    run_id  : None | str = None,
) -> Dict[str, pl.DataFrame | S1Meta]:
    mcs = cfg.output_color_scheme
    inject_color_scheme(globals(), mcs)

    _set_seeds(cfg, mcs)

    feature_cfg                     = FeaturePatterns()
    raw_schema                      = RawSchema()
    
    if mcs:
        mcs.print_section('ЭТАП 1: ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ ДЕТЕКТОРА')
    else:
        print('ЭТАП 1: ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ ДЕТЕКТОРА')

    spans                           = load_spans(cfg, raw_schema)
    spans_raw                       = spans.clone()

    # F-11: the trace-level split is knowable BEFORE features (injection labels
    # are a pure hash of trace_id), so feature-selection statistics can be
    # restricted to train-normal traces instead of peeking at val/test.
    if cfg.inject_anomalies:
        from ars.data.anomalies_injection import planned_trace_labels
        planned = planned_trace_labels(
            spans, InjectionConfig(sem_cols = ('sem_vector',), text_col = 'sem_text'))
        planned = planned.rename({'anomaly_type': DataObject.sublabel}) \
            if DataObject.sublabel != 'anomaly_type' else planned
    else:
        planned = (spans
            .group_by(DataObject.trace_id)
            .agg(pl.col(DataObject.sublabel).drop_nulls().first())
            .sort(DataObject.trace_id))
    planned_units       = planned.filter(pl.col(DataObject.sublabel) != DataObject.class_sentinel)
    planned_train_ids   = split_trace_ids(planned_units, DataObject.sublabel, cfg)[0]

    spans, epi_feature_names        = calculate_features(
        spans, cfg, feature_cfg, raw_schema, selection_ids = planned_train_ids)
    
    # этапы генерации альтернативных семантик и синтетики ...
    
    shared_embedder                 = make_embedder(cfg)
    spans                           = compute_semantic_embeddings(
        spans, cfg, 'sem_text', 'sem_vector', embed = shared_embedder)

    if cfg.export_features:
        spans.write_parquet((cfg.output_dir / 'spans_features.parquet').as_posix())
    
    if cfg.inject_anomalies:
        @benchmark('инъекция аномалий')
        def _inject(spans_w_features: pl.DataFrame) -> pl.DataFrame:
            mcs.print_section(f'ИНЪЕКЦИЯ АНОМАЛИЙ')

            with redirect_native_stderr(PurePath('/tmp/debug.log')):
                spans_w_anoms = inject_anomalies(
                    spans_w_features,
                    InjectionConfig(sem_cols = ('sem_vector',), text_col = 'sem_text'),
                    embedder = shared_embedder)
            spans_w_anoms = spans_w_anoms.with_columns(
                (pl.col(DataObject.sublabel) != 'NonAnomaly').cast(pl.Int8).alias(DataObject.is_anomaly))            
            #spans_w_anoms = fill_missing_values(spans_w_anoms, cfg)

            return spans_w_anoms
        
        spans = _inject(spans)

    traces                          = build_traces(spans, ('sem_vector',), cfg)
    
    strata                          = DataObject.sublabel if DataObject.sublabel in traces.columns else DataObject.is_anomaly
    train, val, test                = stratified_split(traces, strata, cfg)

    if cfg.inject_anomalies:
        # the planned (pre-feature) split must agree with the actual one;
        # a mismatch means injector label assignment drifted (hard error)
        actual_train    = set(train.get_column(DataObject.trace_id).unique().to_list())
        planned_train   = set(planned_train_ids.get_column(DataObject.trace_id).to_list())
        if actual_train != planned_train:
            raise RuntimeError(
                'плановый train-сплит не совпал с фактическим: '
                f'{len(actual_train ^ planned_train)} расхождений — '
                'проверьте согласованность seed/plan инъекции')
    
    (train, val, test), norm_params = normalize_epi_features(train, val, test, cfg)
    
    meta                            = save_datasets(
        epi_feature_names, train, val, test, norm_params, cfg, run_id)

    if mcs: mcs.print_section('ДАННЫЕ ПОДГОТОВЛЕНЫ')
    else:   print('ДАННЫЕ ПОДГОТОВЛЕНЫ')
    
    _viz_dir = cfg.output_dir / 'visualizations'
    viz.write_report(
        cfg.output_dir / 'data_report.html',
        'Этап 1 — отчёт подготовки данных',
        f'{cfg.output_prefix} · нормализация: {cfg.epi_normalization}',
        (
            viz.block_cards('Сводка', (
                ('Сырых спанов',    str(spans_raw.height)),
                ('Трасс',           str(traces.height)),
                ('EPI-признаков',   str(len(epi_feature_names))),
                ('Train/Val/Test',  f'{train.height} / {val.height} / {test.height}'))),
            viz.block_image('Пропуски признаков',           _viz_dir / 'features_missingness.png'),
            viz.block_image('Корреляции отобранных',        _viz_dir / 'feature_correlation.png'),
            viz.block_image('Баланс классов по выборкам',   _viz_dir / 'split_balance.png'),
            viz.block_image('Параметры нормализации',       _viz_dir / 'norm_params.png')))

    return {
        'spans_raw':    spans_raw,
        'train':        train,
        'val':          val,
        'test':         test,
        'meta':         meta
    }


def process_train_data(
    path_traces_train_aef   : PurePath,
    path_embedding_model    : PurePath,
    root_path               : PurePath,
    output_prefix           : str,
    run_id                  : str,
    recast                  : bool = False,
    overrides               : None | dict = None,
) -> Tuple[S1Config, S1Meta]:
    Path(train_dir := root_path / 'traces_train').mkdir(parents = True, exist_ok = True)

    cfg_train = S1Config(
        input_parquet_files = (path_traces_train_aef,),
        output_dir          = train_dir,
        output_prefix       = output_prefix,
        #embedding_cache     = path_embedding_model,
        embedder_path       = path_embedding_model,
        output_color_scheme = ColorSchemeDataScienceSakura(),
        recast              = recast,
        **(overrides or {})
    )

    inject_color_scheme(globals(), cfg_train.output_color_scheme)
    
    return cfg_train, main(cfg_train, run_id = run_id)['meta']


def prepare_test_data(
    cfg_train               : S1Config,
    path_traces_test_aef    : PurePath,
    s1_meta                 : S1Meta,
    root_path               : PurePath,
    output_prefix           : str,
) -> pl.LazyFrame:
    Path(test_dir := root_path / 'traces_test').mkdir(parents = True, exist_ok = True)
    
    cfg_test = replace(cfg_train,
        input_parquet_files   = (path_traces_test_aef,),
        output_dir            = test_dir,
        output_prefix         = output_prefix,)
    
    mcs = cfg_test.output_color_scheme
    inject_color_scheme(globals(), mcs)

    #_set_seeds(cfg_test, mcs)
    
    # F-10: the serving embedder must be the one the artifacts were built with
    from ars.tools.utilities.fingerprint import verify_fingerprint
    verify_fingerprint(cfg_test.embedder_path, getattr(s1_meta, 'embedding_fingerprint', None))

    raw_schema          = RawSchema()
    feature_patterns    = FeaturePatterns()
    features_span       = FeaturesSpan()
    
    spans               = load_spans(cfg_test, raw_schema)
    
    spans_enriched      = features_span.make_features(spans, feature_patterns)
    
    spans_filled        = fill_missing_values(spans_enriched, cfg_test)

    spans_epi           = spans_filled.with_columns(
        pl.concat_list(s1_meta.epi_features).alias('epi_vector')
    )
    
    spans_sem           = compute_semantic_embeddings(
        spans_epi, cfg_test, text_col = 'sem_text', out_col = 'sem_vector'
    )
    
    traces              = build_traces(spans_sem, ('sem_vector',), cfg_test).drop(
        DataObject.is_anomaly, DataObject.sublabel)

    norm_params         = s1_meta.epi_normalization
    epi_dim             = s1_meta.epi_dim
    shift_arr           = pl.lit(norm_params['shift'], dtype = pl.Array(pl.Float64, epi_dim))
    scale_arr           = pl.lit(norm_params['scale'], dtype = pl.Array(pl.Float64, epi_dim))
    z_clip              = float(norm_params.get('z_clip') or 0.0)

    traces_normalized   = _apply_norm(traces, epi_dim, shift_arr, scale_arr, z_clip)

    return traces_normalized.lazy()


if __name__ == '__main__':
    parser = ArgumentParser(
        description = 'система распознавания аномалий в мультиагентных системах | stage 1: подготовка данных'
    )
    args = parser.parse_args()

    cfg = S1Config(
        input_parquet_files = (
            PurePath('/mnt/data/traces/raw/parquet/2k_generated_traces_fraud_agent.parquet'),
            #PurePath('/mnt/data/traces/raw/parquet/ipoteka_new_p1.parquet'),
            #PurePath('/mnt/data/traces/raw/parquet/ipoteka_new_p2.parquet'),
            #PurePath('/mnt/data/traces/raw/parquet/ipoteka_old_p1.parquet'),
            #PurePath('/mnt/data/traces/raw/parquet/ipoteka_old_p2.parquet'),
        ),
        output_dir          = (od := PurePath('/mnt/data/traces/processed/detector/fraud_agent')),
        output_prefix       = 'fa_',
        #epi_normalization   = 'zscore',
        samples_dir         = od / 'artifacts1',
        output_color_scheme = ColorSchemeDataScienceSakura()
    )
    
    _ = tuple(map(
        lambda obj: setattr(obj, '__cs_call', cfg.output_color_scheme),
        filter(lambda obj: callable(obj) and hasattr(obj, '__benchmarked'), globals().values())))
    
    sprint(main(cfg), style_code = cfg.output_color_scheme.info)
