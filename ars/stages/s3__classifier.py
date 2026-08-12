import  ars.configuration.c0__env_setup

from    typing                                          import Tuple, ClassVar, cast
from    dataclasses                                     import dataclass, asdict
from    itertools                                       import starmap
from    functools                                       import reduce

from    pathlib                                         import PurePath, Path

import  polars                                          as pl

import  jax                                             as jx
import  jax.numpy                                       as jp

from    ars.configuration.c3__classifier                import S3Config
from    ars.specification.spec                          import DataObject
from    ars.data.stages_meta                            import S1Meta, S2Meta, S3Meta
from    ars.stages.s2__detector                         import load_models_for_inference, Pad

from    ars.models.m2__detector.architecture            import Branch, Models, InferenceMeta, Array
from    ars.models.m2__detector.confidence              import Predict
from    ars.models.m3__classifier.architecture          import Array
from    ars.models.m3__classifier.stacking              import Stack, StackState
from    ars.configuration.experiments.e3__classifier    import EXPERIMENTS, Experiment
from    ars.tools.utilities.miscellaneous               import FileIO
from    ars.tools.performance.perf                      import benchmark, measure_during_call, inject_color_scheme
from    ars.tools.tui.tui                               import print_table, print_best_summary, ColorSchemeDataScience, Progress
from    ars.tools.tui.tui_data                          import ColorSchemeDataScienceCold
from    ars.tools.visualisations                        import viz


@dataclass(frozen = True)
class Signals:
    e_epi       : Array
    e_sem       : Array
    e_comb      : Array
    p_anomaly   : Array
    confidence  : Array

    def slice(self, lo: int, hi: int) -> 'Signals':
        return Signals(self.e_epi[lo:hi], self.e_sem[lo:hi], self.e_comb[lo:hi], self.p_anomaly[lo:hi], self.confidence[lo:hi])


@dataclass(frozen = True)
class Split:
    xs  : Array
    ys  : Array


@dataclass(frozen = True)
class FeatureData:
    class_names : Tuple[str, ...]
    feature_dim : int
    train       : Split
    val         : Split
    test        : Split


@dataclass(frozen = True)
class FrameSplits:
    train   : pl.DataFrame
    val     : pl.DataFrame
    test    : pl.DataFrame


@dataclass(frozen = True)
class Features:
    signals : ClassVar[Tuple[str, ...]] = ('e_epi', 'e_sem', 'e_comb', 'p_anomaly', 'confidence')
    derived : ClassVar[int]             = 4

    @staticmethod
    def metaparams(s2: S2Meta) -> Tuple[float, ...]:
        cal = s2.calibration
        return tuple(map(float, (
            s2.best_threshold, s2.best_metric_value, s2.epi_sz_latent, s2.sem_sz_latent, s2.max_len,
            cal.get('comb_temperature', 0.0), cal.get('aux_z_anomaly', 0.0),
            cal.get('w_epi', 0.0), cal.get('w_sem', 0.0), cal.get('w_comb', 0.0), cal.get('bias', 0.0),
            cal.get('epi_median', 0.0), cal.get('epi_mad', 0.0),
            cal.get('sem_median', 0.0), cal.get('sem_mad', 0.0),
            cal.get('comb_median', 0.0), cal.get('comb_mad', 0.0))))

    @staticmethod
    def robust(s2: S2Meta, e_epi: Array, e_sem: Array, e_comb: Array) -> Array:
        cal = s2.calibration
        rz  = lambda e, med, mad: (e - cal.get(med, 0.0)) / (1.4826 * cal.get(mad, 1.0) + 1e-8)
        return jp.stack((
            rz(e_epi,  'epi_median',  'epi_mad'),
            rz(e_sem,  'sem_median',  'sem_mad'),
            rz(e_comb, 'comb_median', 'comb_mad'),
            e_comb - s2.best_threshold), axis = 1)

    @staticmethod
    def assemble(s2: S2Meta, z_epi: Array, z_sem: Array, sig: Signals) -> Array:
        signals = jp.stack((sig.e_epi, sig.e_sem, sig.e_comb, sig.p_anomaly, sig.confidence), axis = 1)
        robust  = Features.robust(s2, sig.e_epi, sig.e_sem, sig.e_comb)
        flat    = jp.asarray(Features.metaparams(s2))
        meta    = jp.broadcast_to(flat, (z_epi.shape[0], flat.shape[0]))
        return jp.concatenate((z_epi, z_sem, signals, robust, meta), axis = 1)

    @staticmethod
    def dim(s2: S2Meta) -> int:
        return s2.epi_sz_latent + s2.sem_sz_latent + len(Features.signals) + Features.derived + len(Features.metaparams(s2))

    @staticmethod
    def layout(s2: S2Meta) -> dict:
        return {
            'z_epi':        s2.epi_sz_latent,
            'z_sem':        s2.sem_sz_latent,
            'signals':      len(Features.signals),
            'derived':      Features.derived,
            'metaparams':   len(Features.metaparams(s2))}


@dataclass(frozen = True)
class Encode:
    @staticmethod
    def labels(df: pl.DataFrame, class_names: Tuple[str, ...]) -> Array:
        mapping = dict(zip(class_names, range(len(class_names))))
        return jp.asarray(tuple(map(lambda s: mapping.get(s, 0), df[DataObject.sublabel].to_list())), dtype = jp.int32)

    @staticmethod
    def latents_and_signals(meta: InferenceMeta, models: Models, df: pl.DataFrame, epi_dim: int, sem_dim: int, max_len: int, chunk: int) -> Tuple[Array, Array, Signals]:
        epi_pad, epi_mask   = Pad.split(df, 'epi_sequence',            epi_dim, max_len, chunk)
        sem_pad, sem_mask   = Pad.split(df, 'sem_sequence_sem_vector', sem_dim, max_len, chunk)
        out                 = Predict.batch(meta, models, epi_pad, epi_mask, sem_pad, sem_mask)
        epi_lat             = Branch.encode(models.epi_model, models.epi_state, epi_pad, epi_mask)
        sem_lat             = Branch.encode(models.sem_model, models.sem_state, sem_pad, sem_mask)
        z_epi               = (epi_lat - meta.epi_latent_mean) / meta.epi_latent_std if meta.normalize_latent else epi_lat
        z_sem               = (sem_lat - meta.sem_latent_mean) / meta.sem_latent_std if meta.normalize_latent else sem_lat
        return z_epi, z_sem, Signals(out.e_epi, out.e_sem, out.e_comb, out.p_anomaly, out.confidence)


@dataclass(frozen = True)
class Data:
    @staticmethod
    def pool(data_dir: Path, prefix: str, anomaly_types: Tuple[str, ...]) -> pl.DataFrame:
        frames = tuple(map(
            lambda suffix: pl.scan_parquet((data_dir / f'{prefix}_{suffix}.parquet').as_posix()),
            ('train', 'val', 'test')))
        return pl.concat(frames, how = 'vertical_relaxed').filter(
            pl.col(DataObject.sublabel).is_in(list(anomaly_types))).collect()

    @staticmethod
    def stratified(df: pl.DataFrame, label_col: str, seed: int, train_frac: float, val_frac: float) -> FrameSplits:
        ranked  = df.sample(fraction = 1.0, shuffle = True, seed = seed).with_columns(
            (pl.int_range(pl.len()).over(label_col) / pl.len().over(label_col)).alias('_q'))
        pick    = lambda lo, hi: ranked.filter((pl.col('_q') >= lo) & (pl.col('_q') < hi)).drop('_q')
        return FrameSplits(
            train   = pick(0.0,                   train_frac),
            val     = pick(train_frac,            train_frac + val_frac),
            test    = pick(train_frac + val_frac, 1.01))


@benchmark('сборка признаков s3')
def prepare_features(cfg: S3Config) -> FeatureData:
    mcs = cfg.output_color_scheme
    mcs.print_section('СБОРКА ПРИЗНАКОВ КЛАССИФИКАТОРА')
    prefix          = cfg._prefix_for()
    data_dir        = Path(cfg.s1_meta.output_dir)
    class_names     = cfg.s1_meta.anomaly_types
    meta, models    = load_models_for_inference(cfg.s2_meta)
    max_len         = cfg.s2_meta.max_len
    pooled          = Data.pool(data_dir, prefix, class_names)

    # F-06: a class with too few members produces empty/degenerate splits, which
    # used to crash Pad.split (and silently break early stopping). Fail early
    # with an actionable message instead.
    counts  = dict(pooled.group_by(DataObject.sublabel).len().iter_rows())
    starved = dict(filter(lambda kv: kv[1] < cfg.min_per_class,
                          ((c, counts.get(c, 0)) for c in class_names)))
    if starved:
        raise ValueError(
            f'классификатор: недостаточно примеров на класс (минимум {cfg.min_per_class}): '
            f'{starved}. Увеличьте корпус/долю инъекций или отключите классификатор '
            f'(classifier.enabled=false).')

    splits          = Data.stratified(pooled, DataObject.sublabel, cfg.seed, cfg.train_frac, cfg.val_frac)

    def featurize(df: pl.DataFrame) -> Split:
        z_epi, z_sem, sig = Encode.latents_and_signals(
            meta, models, df, cfg.s2_meta.epi_dim, cfg.s2_meta.sem_dim, max_len, cfg.s2_meta.seq_pad_chunk)
        return Split(Features.assemble(cfg.s2_meta, z_epi, z_sem, sig), Encode.labels(df, class_names))

    train, val, test = featurize(splits.train), featurize(splits.val), featurize(splits.test)
    mcs.print_metric('Подклассы (источник истины)', ', '.join(class_names))
    mcs.print_metric('Классов аномалий',        len(class_names))
    mcs.print_metric('Размерность признаков',   Features.dim(cfg.s2_meta))
    mcs.print_metric('Размечено аномалий',      pooled.height)
    mcs.print_metric('train',   f'{train.xs.shape[0]} трасс')
    mcs.print_metric('val',     f'{val.xs.shape[0]} трасс')
    mcs.print_metric('test',    f'{test.xs.shape[0]} трасс')
    return FeatureData(class_names, int(Features.dim(cfg.s2_meta)), train, val, test)


@benchmark('эксперимент классификатора')
def run_experiment(exp_cls: type[Experiment], data: FeatureData, cfg: S3Config) -> dict:
    mcs     = cfg.output_color_scheme
    exp     = exp_cls()
    classes = len(data.class_names)
    mcs.print_section(f'Эксперимент классификатора: {exp.name}')
    mcs.print_metric('Базовые модели',  '+'.join(exp.bases))
    mcs.print_metric('Мета-солвер',     exp.meta_solver)
    mcs.print_metric('Фолдов OOF',      exp.n_folds)
    mcs.print_metric('Признаков',       int(data.feature_dim))
    mcs.print_metric('train/val/test',  f'{data.train.xs.shape[0]}/{data.val.xs.shape[0]}/{data.test.xs.shape[0]}')
    stack, elapsed, _cpu, _gpu  = measure_during_call(
        Stack.fit, (jx.random.PRNGKey(cfg.seed), data.train.xs, data.train.ys, classes, exp, (data.val.xs, data.val.ys)), {})
    test_metrics                = Stack.evaluate(stack, data.test.xs, data.test.ys, cfg.eps)
    val_metrics                 = Stack.evaluate(stack, data.val.xs,  data.val.ys,  cfg.eps)
    experiment_dir              = Path(cfg.output_dir) / exp.name
    experiment_dir.mkdir(parents = True, exist_ok = True)
    FileIO.pickle_write(experiment_dir / 'stack.pkl', stack)
    _ = tuple(starmap(lambda k, v: mcs.print_metric(f'Test {k}', f'{v:.4f}'), asdict(test_metrics).items()))
    return {
        'experiment':   exp.name,
        'test_metrics': asdict(test_metrics),
        'val_metrics':  asdict(val_metrics),
        'bases':        exp.bases,
        'meta_solver':  exp.meta_solver,
        'n_folds':      exp.n_folds,
        'time':         float(elapsed)}


def _summary_row(res: dict) -> dict:
    metrics = res['test_metrics']
    return {
        'Experiment':   res['experiment'],
        'Accuracy':     float(metrics['accuracy']),
        'Macro-F1':     float(metrics['f1']),
        'Macro-Recall': float(metrics['recall']),
        'Bases':        '+'.join(res['bases']),
        'Meta':         res['meta_solver'],
        'Folds':        res['n_folds'],
        'Time (s)':     float(res['time'])}


def _print_summary(mcs: ColorSchemeDataScience, all_results: Tuple[dict, ...]) -> None:
    if not all_results: return
    mcs.print_section('ИТОГОВАЯ СВОДКА ПО ЭКСПЕРИМЕНТАМ КЛАССИФИКАТОРА')
    print_table(
        pl.DataFrame(map(_summary_row, all_results)),
        cols            = ('Experiment', 'Accuracy', 'Macro-F1', 'Macro-Recall', 'Bases', 'Meta', 'Folds', 'Time (s)'),
        highlight_cols  = {'Accuracy': 'max', 'Macro-F1': 'max', 'Macro-Recall': 'max'},
        float_fmt       = '.4f', scheme = mcs)


def _pick_best(all_results: Tuple[dict, ...], select_metric: str) -> dict:
    return reduce(
        lambda a, b: a if a['test_metrics'][select_metric] >= b['test_metrics'][select_metric] else b,
        all_results)


def _write_report(cfg: S3Config, data: FeatureData, all_results: Tuple[dict, ...], best: dict) -> None:
    layout      = Features.layout(cfg.s2_meta)
    summary_df  = pl.DataFrame(tuple(map(_summary_row, all_results)))
    layout_df   = pl.DataFrame({'компонент': tuple(layout.keys()), 'размер': tuple(layout.values())})
    viz.write_report(
        Path(cfg.output_dir) / 'classifier_report.html',
        'Этап 3 — отчёт классификатора типов аномалий',
        f'{cfg.output_prefix} · победитель: {best['experiment']} · {cfg.select_metric} = {best['test_metrics'][cfg.select_metric]:.4f}',
        (
            viz.block_cards('Конфигурация победителя', (
                ('Эксперимент',          str(best['experiment'])),
                ('Базовые модели',       '+'.join(best['bases'])),
                ('Мета-солвер',          str(best['meta_solver'])),
                ('Фолдов OOF',           str(best['n_folds'])),
                ('Классов аномалий',     str(len(data.class_names))),
                ('Размерность признаков', str(int(data.feature_dim))))),
            viz.block_table('Метрики по экспериментам (test)',          summary_df),
            viz.block_table('Состав метапризнаков на сигналах детектора', layout_df),
            viz.block_cards('Пайплайн метапризнаков', (
                ('Классы (источник истины)', ', '.join(data.class_names)),
                ('Латенты',                  'z_epi (EPI-AE) + z_sem (SEM-AE) детектора'),
                ('Сигналы детектора',        ', '.join(Features.signals)),
                ('Производные',              'robust-z ошибок epi/sem/comb + зазор до порога'),
                ('Метапараметры',            f'{len(Features.metaparams(cfg.s2_meta))} из s2 (broadcast по строкам)'),
                ('Стекинг',                  f'{best['n_folds']}-fold OOF → мета-солвер {best['meta_solver']}')))))


@benchmark('обучение экспериментов s3')
def train_experiments(cfg: S3Config) -> Tuple[FeatureData, Tuple[dict, ...]]:
    mcs         = cfg.output_color_scheme
    data        = prepare_features(cfg)
    bar         = Progress.bar(EXPERIMENTS, desc = 'эксперименты s3', unit = 'exp', **mcs.tqdm_kwargs())
    all_results = tuple(map(lambda exp_cls: run_experiment(exp_cls, data, cfg), bar))
    _print_summary(mcs, all_results)
    best        = _pick_best(all_results, cfg.select_metric)
    best_dir    = Path(cfg.output_dir) / 'best'
    best_dir.mkdir(parents = True, exist_ok = True)
    FileIO.json_write(best_dir / 'best_info.json', {
        'best_experiment':  best['experiment'],
        'metric':           cfg.select_metric,
        'value':            float(best['test_metrics'][cfg.select_metric]),
        'test_metrics':     best['test_metrics']})
    print_best_summary(mcs, best)
    _write_report(cfg, data, all_results, best)
    mcs.print_metric('HTML-отчёт классификатора', (Path(cfg.output_dir) / 'classifier_report.html').as_posix())
    return data, all_results


def _build_s3_meta(cfg: S3Config, data: FeatureData, all_results: Tuple[dict, ...]) -> S3Meta:
    best = _pick_best(all_results, cfg.select_metric)
    return S3Meta(
        output_dir          = Path(cfg.output_dir).as_posix(),
        experiment_dir      = (Path(cfg.output_dir) / best['experiment']).as_posix(),
        best_experiment     = best['experiment'],
        select_metric       = cfg.select_metric,
        best_metric_value   = float(best['test_metrics'][cfg.select_metric]),
        n_classes           = len(data.class_names),
        class_names         = tuple(data.class_names),
        feature_dim         = int(data.feature_dim),
        feature_layout      = Features.layout(cfg.s2_meta),
        base_kinds          = tuple(best['bases']),
        meta_solver         = best['meta_solver'],
        n_folds             = int(best['n_folds']),
        test_metrics        = dict(starmap(lambda k, v: (k, float(v)), best['test_metrics'].items())),
        metaparams          = Features.metaparams(cfg.s2_meta),
        s2_meta             = cfg.s2_meta)


def train_classifier(s1_meta: S1Meta, s2_meta: S2Meta, overrides: None | dict = None) -> S3Meta:
    data_dir    = PurePath(s1_meta.output_dir)
    cfg         = S3Config(
        s1_meta             = s1_meta,
        s2_meta             = s2_meta,
        output_dir          = data_dir / 'classifier',
        output_prefix       = s1_meta.prefix,
        run_id              = s1_meta.run_id,
        output_color_scheme = ColorSchemeDataScienceCold(),
        **(overrides or {}))
    Path(cfg.output_dir).mkdir(parents = True, exist_ok = True)
    inject_color_scheme(globals(), cfg.output_color_scheme)
    data, all_results = train_experiments(cfg)
    return _build_s3_meta(cfg, data, all_results)


def classify_anomalies(data: pl.LazyFrame, s3_meta: S3Meta) -> pl.LazyFrame:
    df = data.collect()
    if df.height == 0:
        return df.with_columns(
            pl.lit('', dtype = pl.String).alias(DataObject.sublabel),
            pl.lit(0.0, dtype = pl.Float64).alias('classifier_confidence')).lazy()
    stack   = cast(StackState, FileIO.pickle_read(PurePath(s3_meta.experiment_dir) / 'stack.pkl'))
    z_epi   = jp.asarray(df['detector_z_epi'].to_list(), dtype = jp.float32)
    z_sem   = jp.asarray(df['detector_z_sem'].to_list(), dtype = jp.float32)
    sig     = Signals(
        e_epi       = jp.asarray(df['detector_e_epi'].to_list(),                 dtype = jp.float32),
        e_sem       = jp.asarray(df['detector_e_sem'].to_list(),                 dtype = jp.float32),
        e_comb      = jp.asarray(df['detector_reconstruction_error'].to_list(),  dtype = jp.float32),
        p_anomaly   = jp.asarray(df['detector_p_anomaly'].to_list(),             dtype = jp.float32),
        confidence  = jp.asarray(df['detector_confidence'].to_list(),            dtype = jp.float32))
    bounds  = tuple(map(lambda s: (s, min(s + 8192, df.height)), range(0, df.height, 8192)))
    proba   = jp.concatenate(tuple(map(
        lambda lh: Stack.proba(stack, Features.assemble(
            s3_meta.s2_meta, z_epi[lh[0]:lh[1]], z_sem[lh[0]:lh[1]], sig.slice(lh[0], lh[1]))),
        bounds)), axis = 0)
    pred    = jp.argmax(proba, axis = 1)
    conf    = jp.max(proba, axis = 1)
    return df.with_columns(
        pl.Series(DataObject.sublabel, list(map(lambda i: s3_meta.class_names[i], pred.tolist()))),
        pl.Series('classifier_confidence', conf.tolist())).lazy()