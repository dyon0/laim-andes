import  ars.configuration.c0__env_setup

from    typing                                      import Tuple, Callable, Iterable
from    dataclasses                                 import dataclass, asdict, replace
from    itertools                                   import starmap, chain
from    functools                                   import reduce, partial

from    argparse                                    import ArgumentParser
from    operator                                    import itemgetter
from    pathlib                                     import PurePath, Path
from    random                                      import Random

import  polars                                      as pl

import  jax                                         as jx
import  jax.numpy                                   as jp

import  optax                                       as ox

from    sklearn.metrics                             import average_precision_score, roc_auc_score

from    ars.configuration.c2__detector              import S2Config
from    ars.specification.spec                      import DataObject
from    ars.data.stages_meta                        import S1Meta, S2Meta
from    ars.models.metrics                          import MetricName, MetricSet, Binary, Confusion, Operators
from    ars.models.m2__detector.architecture        import (
    LSTM_AE, FMLP_AE,
    TrainState, HyperParamsLSTMAE, HyperParamsFMLPAE,
    Branch, Calibration, Models, InferenceMeta, PreparedData, Array, Params)
from    ars.models.m2__detector.train               import Trainer
from    ars.models.m2__detector.confidence          import Calibrate, Predict
from    ars.configuration.experiments.e2__detector  import EXPERIMENTS, Experiment
from    ars.tools.utilities.miscellaneous           import FileIO
from    ars.tools.performance.perf                  import benchmark, inject_color_scheme
from    ars.tools.tui.tui                           import sprint, print_table, ColorSchemeDataScience, capture_epoch_progress, print_summary_table, print_config_table, print_best_summary, Progress
from    ars.tools.tui.tui_data                      import ColorSchemeDataScienceCold
from    ars.tools.visualisations.viz                import (
    save_loss_plot, save_error_distribution, save_roc, save_threshold_sweep, save_confusion_preds,
    generate_experiment_report, generate_summary_report)


@dataclass(frozen = True)
class TracePrediction:
    reconstruction_error    : float
    p_anomaly               : float
    confidence              : float
    is_anomaly              : bool


def _maybe_tolist(arr: None | Array) -> None | Tuple[float, ...]:
    return tuple(arr.tolist()) if arr is not None else None


@dataclass(frozen = True)
class Pad:
    '''векторное извлечение и паддинг трасс в тензоры [N, max_len, dim] — чанками, без по-трассного python-цикла'''

    @staticmethod
    def max_len(frames: Iterable[pl.DataFrame], col: str) -> int:
        return reduce(max, map(lambda d: d.select(pl.col(col).list.len().max()).item() or 1, frames), 1)

    @staticmethod
    def split(df: pl.DataFrame, col: str, dim: int, max_len: int, chunk: int) -> Tuple[Array, Array]:
        if df.height == 0:  # F-06: empty input must yield empty tensors, not a concat crash
            return (jp.zeros((0, max_len, dim), dtype = jp.float32),
                    jp.zeros((0, max_len),      dtype = bool))
        def block(start: int) -> Tuple[Array, Array]:
            rows = min(chunk, df.height - start)
            flat = (df.slice(start, chunk).select(col)
                .with_row_index('_li')
                .explode(col)
                .with_columns(pl.int_range(0, pl.len()).over('_li').alias('_step'))
                .filter(pl.col('_step') < max_len))
            vecs = jp.asarray(flat.get_column(col).list.to_array(dim).to_numpy(), dtype = jp.float32)
            li   = jp.asarray(flat.get_column('_li').to_numpy())
            step = jp.asarray(flat.get_column('_step').to_numpy())
            return (jp.zeros((rows, max_len, dim), dtype = jp.float32).at[li, step].set(vecs),
                    jp.zeros((rows, max_len),      dtype = bool       ).at[li, step].set(True))
        parts = tuple(map(block, range(0, df.height, chunk)))
        return (jp.concatenate(tuple(map(itemgetter(0), parts)), axis = 0),
                jp.concatenate(tuple(map(itemgetter(1), parts)), axis = 0))


def calculate_metrics(labels: Array, predictions: Array, eps: float) -> MetricSet:
    return Binary.evaluate(labels, predictions, eps)


@partial(jx.jit, static_argnames = ('n_thresholds',))
def _threshold_sweep(errors: Array, labels: Array, n_thresholds: int, eps: Array) -> Tuple[Array, MetricSet]:
    thresholds  = jp.linspace(errors.min(), errors.max(), n_thresholds)
    pos_label   = (labels == 1).astype(jp.float32)
    neg_label   = (labels == 0).astype(jp.float32)

    def _step(carry, threshold):
        preds   = (errors > threshold).astype(jp.float32)
        tp      = jp.sum(preds         * pos_label)
        fp      = jp.sum(preds         * neg_label)
        fn      = jp.sum((1.0 - preds) * pos_label)
        tn      = jp.sum((1.0 - preds) * neg_label)
        return carry, jp.stack((tp, fp, fn, tn))

    _, confusion            = jx.lax.scan(_step, jp.zeros((), dtype = jp.float32), thresholds)
    tp_a, fp_a, fn_a, tn_a  = confusion[:, 0], confusion[:, 1], confusion[:, 2], confusion[:, 3]
    return thresholds, Confusion.metrics(tp_a, tn_a, fp_a, fn_a, eps)


def select_threshold(errors: Array, labels: Array, n_thresholds: int, target_metric: MetricName, eps: float) -> Tuple[float, MetricSet]:
    thresholds, sweep   = _threshold_sweep(errors, labels, n_thresholds, jp.asarray(eps, dtype = jp.float32))
    best_idx            = int(jp.argmax(Operators.named(sweep, target_metric)))
    best_threshold      = float(thresholds[best_idx])
    best_metrics        = Operators.to_float(jx.tree_util.tree_map(lambda column: column[best_idx], sweep))
    return best_threshold, best_metrics


def _train_lstm_with_progress(
        exp_cfg         : Experiment,
        mcs             : ColorSchemeDataScience,
        label           : str,
        model           : LSTM_AE,
        train_padded    : Array,
        train_mask      : Array,
        val_padded      : Array,
        val_mask        : Array,
        input_shape     : Tuple[int, ...],
        rng             : jx.Array,
        schedule_fn     : None | Callable,
        target_loss     : None | float,
) -> Tuple[TrainState, Tuple[float, ...]]:
    with Progress.bar(total = exp_cfg.epochs, desc = label, unit = 'epoch', **mcs.tqdm_kwargs()) as bar:
        with capture_epoch_progress(bar):
            state, losses = Trainer.train_lstm_ae(
                model           = model, train_padded = train_padded, train_mask = train_mask,
                val_padded      = val_padded, val_mask = val_mask,
                learning_rate   = exp_cfg.learning_rate, num_epochs = exp_cfg.epochs, batch_size = exp_cfg.batch_size,
                rng             = rng, input_shape = input_shape, weight_decay = exp_cfg.weight_decay, clip_grad = exp_cfg.clip_grad,
                schedule_fn     = schedule_fn, patience = exp_cfg.patience, target_loss = target_loss)
    return state, tuple(map(float, losses))


def _train_fmlp_with_progress(
        exp_cfg     : Experiment,
        mcs         : ColorSchemeDataScience,
        label       : str,
        model       : FMLP_AE,
        train_epi   : Array,
        train_sem   : Array,
        val_epi     : Array,
        val_sem     : Array,
        rng         : jx.Array,
        schedule_fn : None | Callable,
        target_loss : None | float,
) -> Tuple[TrainState, Tuple[float, ...]]:
    with Progress.bar(total = exp_cfg.epochs, desc = label, unit = 'epoch', **mcs.tqdm_kwargs()) as bar:
        with capture_epoch_progress(bar):
            state, losses = Trainer.train_fmlp_ae(
                model           = model, train_epi = train_epi, train_sem = train_sem, val_epi = val_epi, val_sem = val_sem,
                learning_rate   = exp_cfg.learning_rate, num_epochs = exp_cfg.epochs, batch_size = exp_cfg.batch_size,
                rng             = rng, weight_decay = exp_cfg.weight_decay, clip_grad = exp_cfg.clip_grad,
                schedule_fn     = schedule_fn, patience = exp_cfg.patience, target_loss = target_loss)
    return state, tuple(map(float, losses))


@benchmark
def load_prepared_data(cfg: S2Config) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    '''загружает train/val/test parquet-артефакты, сохранённые этапом s1_data'''

    mcs = cfg.output_color_scheme
    mcs.print_section('ЗАГРУЗКА ДАННЫХ')

    prefix      = cfg._prefix_for()
    data_dir    = Path(cfg.s1_meta.output_dir)

    cols        = ('epi_sequence', 'sem_sequence_sem_vector', DataObject.is_anomaly)
    load        = lambda split: pl.scan_parquet((data_dir / f'{prefix}_{split}.parquet').as_posix()).select(cols).collect()
    train_df, val_df, test_df = load('train'), load('val'), load('test')

    mcs.print_metric('Train samples', train_df.height)
    mcs.print_metric('Val samples',   val_df.height)
    mcs.print_metric('Test samples',  test_df.height)

    return train_df, val_df, test_df


@benchmark
def prepare_arrays(train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame, s1_meta: S1Meta, cfg: S2Config) -> PreparedData:
    '''переводит датафреймы в jax-тензоры, паддит, фильтрует train/val от аномалий'''

    mcs = cfg.output_color_scheme
    mcs.print_section('ПОДГОТОВКА ДАННЫХ')

    epi_dim = s1_meta.epi_dim
    sem_dim = s1_meta.semantic_vectors['sem_sem_vector']

    chunk   = cfg.seq_pad_chunk

    mcs.print_subsection('Извлечение и паддинг последовательностей')
    train_lbls  = tuple(train_df[DataObject.is_anomaly].to_list())
    val_lbls    = tuple(val_df[DataObject.is_anomaly].to_list())
    test_lbls   = tuple(test_df[DataObject.is_anomaly].to_list())

    mcs.print_metric('train', f'{train_df.height} трасс')
    mcs.print_metric('val',   f'{val_df.height} трасс')
    mcs.print_metric('test',  f'{test_df.height} трасс')

    mcs.print_subsection('Параметры тензоров')
    mcs.print_metric('EPI размерность', epi_dim)
    mcs.print_metric('SEM размерность', sem_dim)

    max_len = Pad.max_len((train_df, val_df, test_df), 'epi_sequence')
    mcs.print_metric('Максимальная длина (в спанах)', max_len)

    train_epi_pad, train_epi_mask   = Pad.split(train_df, 'epi_sequence', epi_dim, max_len, chunk)
    val_epi_pad,   val_epi_mask     = Pad.split(val_df,   'epi_sequence', epi_dim, max_len, chunk)
    test_epi_pad,  test_epi_mask    = Pad.split(test_df,  'epi_sequence', epi_dim, max_len, chunk)
    train_sem_pad, train_sem_mask   = Pad.split(train_df, 'sem_sequence_sem_vector', sem_dim, max_len, chunk)
    val_sem_pad,   val_sem_mask     = Pad.split(val_df,   'sem_sequence_sem_vector', sem_dim, max_len, chunk)
    test_sem_pad,  test_sem_mask    = Pad.split(test_df,  'sem_sequence_sem_vector', sem_dim, max_len, chunk)

    train_lbls_arr      = jp.asarray(train_lbls, dtype = jp.int32)
    train_normal_mask   = train_lbls_arr == 0
    train_lbls_norm     = tuple(filter(lambda lbl: lbl == 0, train_lbls))
    mcs.print_metric('Train (только нормальные)', len(train_lbls_norm))

    val_lbls_arr    = jp.asarray(val_lbls, dtype = jp.int32)
    val_normal_mask = val_lbls_arr == 0
    val_lbls_normal = tuple(filter(lambda lbl: lbl == 0, val_lbls))
    mcs.print_metric('Val normal (для early stopping)', len(val_lbls_normal))
    mcs.print_metric('Val mixed (для подбора порога)', len(val_lbls))

    return PreparedData(
        epi_dim             = epi_dim,
        sem_dim             = sem_dim,
        max_len             = max_len,
        train_epi_pad       = train_epi_pad[train_normal_mask],
        train_epi_mask      = train_epi_mask[train_normal_mask],
        train_sem_pad       = train_sem_pad[train_normal_mask],
        train_sem_mask      = train_sem_mask[train_normal_mask],
        train_labels        = train_lbls_norm,
        val_epi_pad_normal  = val_epi_pad[val_normal_mask],
        val_epi_mask_normal = val_epi_mask[val_normal_mask],
        val_sem_pad_normal  = val_sem_pad[val_normal_mask],
        val_sem_mask_normal = val_sem_mask[val_normal_mask],
        val_epi_pad_mixed   = val_epi_pad,
        val_epi_mask_mixed  = val_epi_mask,
        val_sem_pad_mixed   = val_sem_pad,
        val_sem_mask_mixed  = val_sem_mask,
        val_labels_mixed    = val_lbls,
        test_epi_pad        = test_epi_pad,
        test_epi_mask       = test_epi_mask,
        test_sem_pad        = test_sem_pad,
        test_sem_mask       = test_sem_mask,
        test_labels         = test_lbls)


def _exp_schedule(exp_cfg: Experiment, num_train: int, target_losses: None | Tuple[float, ...]) -> None | Callable:
    if target_losses is not None: return None
    decay_steps = exp_cfg.epochs * max(1, num_train // exp_cfg.batch_size)
    return ox.exponential_decay(
        init_value = exp_cfg.learning_rate, transition_steps = max(1, decay_steps // 2), decay_rate = 0.5)


@benchmark('обучение EPI LSTM-AE')
def _train_epi_branch(cfg: S2Config, exp_cfg: Experiment, data: PreparedData, rng: jx.Array, schedule_fn: None | Callable, target_loss: None | float) -> Tuple[LSTM_AE, TrainState, Tuple[float, ...], HyperParamsLSTMAE]:
    mcs = cfg.output_color_scheme
    mcs.print_subsection('Обучение EPI LSTM-AE')
    epi_loss_type, epi_huber    = exp_cfg.get_epi_loss_params()
    hp                          = HyperParamsLSTMAE(
        sz_features     = data.epi_dim, sz_latent = exp_cfg.epi_sz_latent, layers_arch = exp_cfg.epi_layers_arch,
        dropout_rate    = exp_cfg.dropout_rate, decoder_type = exp_cfg.epi_decoder_type, loss_type = epi_loss_type, huber_delta = epi_huber)
    model                       = LSTM_AE(hp)
    state, losses               = _train_lstm_with_progress(
        exp_cfg         = exp_cfg, mcs = mcs, label = 'EPI', model = model,
        train_padded    = data.train_epi_pad, train_mask = data.train_epi_mask,
        val_padded      = data.val_epi_pad_normal, val_mask = data.val_epi_mask_normal,
        input_shape     = (data.max_len, data.epi_dim), rng = rng, schedule_fn = schedule_fn, target_loss = target_loss)
    return model, state, losses, hp


@benchmark('обучение SEM LSTM-AE')
def _train_sem_branch(cfg: S2Config, exp_cfg: Experiment, data: PreparedData, rng: jx.Array, schedule_fn: None | Callable, target_loss: None | float) -> Tuple[LSTM_AE, TrainState, Tuple[float, ...], HyperParamsLSTMAE]:
    mcs = cfg.output_color_scheme
    mcs.print_subsection('Обучение SEM LSTM-AE')
    sem_loss_type, sem_huber    = exp_cfg.get_sem_loss_params()
    hp                          = HyperParamsLSTMAE(
        sz_features     = data.sem_dim, sz_latent = exp_cfg.sem_sz_latent, layers_arch = exp_cfg.sem_layers_arch,
        dropout_rate    = exp_cfg.dropout_rate, decoder_type = exp_cfg.sem_decoder_type, loss_type = sem_loss_type, huber_delta = sem_huber)
    model                       = LSTM_AE(hp)
    state, losses               = _train_lstm_with_progress(
        exp_cfg         = exp_cfg, mcs = mcs, label = 'SEM', model = model,
        train_padded    = data.train_sem_pad, train_mask = data.train_sem_mask,
        val_padded      = data.val_sem_pad_normal, val_mask = data.val_sem_mask_normal,
        input_shape     = (data.max_len, data.sem_dim), rng = rng, schedule_fn = schedule_fn, target_loss = target_loss)
    return model, state, losses, hp


@benchmark('вычисление латентов LSTM-AE')
def _compute_branch_latents(cfg: S2Config, epi_model: LSTM_AE, epi_state: TrainState, sem_model: LSTM_AE, sem_state: TrainState, data: PreparedData) -> Tuple[Array, Array, Array, Array, Array, Array]:
    mcs = cfg.output_color_scheme
    mcs.print_subsection('Вычисление латентных векторов')
    return (
        Branch.encode(epi_model, epi_state, data.train_epi_pad,      data.train_epi_mask,      chunk = cfg.encode_chunk),
        Branch.encode(sem_model, sem_state, data.train_sem_pad,      data.train_sem_mask,      chunk = cfg.encode_chunk),
        Branch.encode(epi_model, epi_state, data.val_epi_pad_normal, data.val_epi_mask_normal, chunk = cfg.encode_chunk),
        Branch.encode(sem_model, sem_state, data.val_sem_pad_normal, data.val_sem_mask_normal, chunk = cfg.encode_chunk),
        Branch.encode(epi_model, epi_state, data.val_epi_pad_mixed,  data.val_epi_mask_mixed,  chunk = cfg.encode_chunk),
        Branch.encode(sem_model, sem_state, data.val_sem_pad_mixed,  data.val_sem_mask_mixed,  chunk = cfg.encode_chunk))


def _normalize_latent_arrays(train_lat: Array, others: Tuple[Array, ...], eps: float) -> Tuple[Array, Array, Tuple[Array, ...]]:
    mean    = jp.mean(train_lat, axis = 0, keepdims = True)
    std     = jp.std (train_lat, axis = 0, keepdims = True) + eps
    normed  = tuple(map(lambda x: (x - mean) / std, chain((train_lat,), others)))
    return mean, std, normed


@benchmark('обучение CMB FMLP-AE')
def _train_combined_branch(cfg: S2Config, exp_cfg: Experiment, train_epi_lat: Array, train_sem_lat: Array, val_epi_lat: Array, val_sem_lat: Array, rng: jx.Array, schedule_fn: None | Callable, target_loss: None | float) -> Tuple[FMLP_AE, TrainState, Tuple[float, ...], HyperParamsFMLPAE]:
    mcs = cfg.output_color_scheme
    mcs.print_subsection('Обучение CMB FMLP-AE')
    comb_loss_type, comb_huber  = exp_cfg.get_combined_loss_params()
    hp                          = HyperParamsFMLPAE(
        sz_latent_epi   = exp_cfg.epi_sz_latent, sz_latent_sem = exp_cfg.sem_sz_latent, layers_arch = exp_cfg.combined_layers_arch,
        dropout_rate    = exp_cfg.dropout_rate, use_batch_norm = exp_cfg.use_batch_norm, loss_type = comb_loss_type, huber_delta = comb_huber)
    model                       = FMLP_AE(hp)
    state, losses               = _train_fmlp_with_progress(
        exp_cfg     = exp_cfg, mcs = mcs, label = 'COMBINED', model = model,
        train_epi   = train_epi_lat, train_sem = train_sem_lat, val_epi = val_epi_lat, val_sem = val_sem_lat,
        rng         = rng, schedule_fn = schedule_fn, target_loss = target_loss)
    return model, state, losses, hp


def _save_branch_pickle(path: PurePath, params: Params, hp: HyperParamsLSTMAE, losses: Tuple[float, ...]) -> None:
    FileIO.pickle_write(path, {'params': params, 'config': hp, 'losses': losses})


def _save_combined_pickle(
        path                : PurePath,
        state               : TrainState,
        hp                  : HyperParamsFMLPAE,
        best_threshold      : float,
        test_metrics        : dict,
        normalize_latent    : bool,
        epi_latent_mean     : None | Tuple[float, ...],
        epi_latent_std      : None | Tuple[float, ...],
        sem_latent_mean     : None | Tuple[float, ...],
        sem_latent_std      : None | Tuple[float, ...],
        losses              : Tuple[float, ...],
        calibration         : Calibration,
) -> None:
    FileIO.pickle_write(path, {
        'params'            : state.params,
        'batch_stats'       : state.batch_stats,
        'config'            : hp,
        'best_threshold'    : best_threshold,
        'test_metrics'      : test_metrics,
        'normalize_latent'  : normalize_latent,
        'epi_latent_mean'   : epi_latent_mean,
        'epi_latent_std'    : epi_latent_std,
        'sem_latent_mean'   : sem_latent_mean,
        'sem_latent_std'    : sem_latent_std,
        'losses'            : losses,
        'calibration'       : asdict(calibration)})


@benchmark
def run_experiment(exp_cls: type[Experiment], data: PreparedData, cfg: S2Config) -> dict:
    '''полный пайплайн обучения детектора по одному эксперименту'''
    mcs             = cfg.output_color_scheme
    exp_cfg         = exp_cls()
    if cfg.epochs   is not None: exp_cfg = replace(exp_cfg, epochs   = cfg.epochs)
    if cfg.patience is not None: exp_cfg = replace(exp_cfg, patience = cfg.patience)
    experiment_name = exp_cfg.name
    experiment_dir  = Path(cfg.output_dir) / experiment_name
    experiment_dir.mkdir(parents = True, exist_ok = True)
    viz_dir = Path(cfg.output_dir) / 'visualizations' / experiment_name
    viz_dir.mkdir(parents = True, exist_ok = True)
    mcs.print_section(f'Запуск эксперимента: {experiment_name}')

    target_losses                       = exp_cfg.target_losses
    target_epi, target_sem, target_comb = target_losses if target_losses is not None else (None, None, None)
    rng                                 = jx.random.PRNGKey(cfg.seed)
    rng_epi, rng_sem, rng_combined      = jx.random.split(rng, 3)
    num_train                           = int(data.train_epi_pad.shape[0])
    schedule                            = _exp_schedule(exp_cfg, num_train, target_losses)

    epi_model, epi_state, epi_losses, hp_epi    = _train_epi_branch(cfg, exp_cfg, data, rng_epi, schedule, target_epi)
    t_epi                                       = _train_epi_branch.elapsed
    epi_train_mse                               = Branch.lstm_recon_mse(epi_model, epi_state, data.train_epi_pad,      data.train_epi_mask,      chunk = cfg.encode_chunk)
    epi_val_mse                                 = Branch.lstm_recon_mse(epi_model, epi_state, data.val_epi_pad_normal, data.val_epi_mask_normal, chunk = cfg.encode_chunk)
    sprint(f'EPI реконструкция: train MSE = {float(epi_train_mse):.4f}, val MSE = {float(epi_val_mse):.4f}', style_code = mcs.info)
    params_norm_epi = Branch.params_norm(epi_state)
    sprint(f'Норма параметров EPI: {float(params_norm_epi):.4f}', style_code = mcs.info)

    sem_model, sem_state, sem_losses, hp_sem    = _train_sem_branch(cfg, exp_cfg, data, rng_sem, schedule, target_sem)
    t_sem                                       = _train_sem_branch.elapsed

    (train_epi_lat, train_sem_lat, val_epi_lat_n, val_sem_lat_n, val_epi_lat_m, val_sem_lat_m) = _compute_branch_latents(
        cfg, epi_model, epi_state, sem_model, sem_state, data)
    t_lat = _compute_branch_latents.elapsed

    if exp_cfg.normalize_latent:
        mean_epi, std_epi, (train_epi_lat, val_epi_lat_n, val_epi_lat_m) = _normalize_latent_arrays(
            train_epi_lat, (val_epi_lat_n, val_epi_lat_m), cfg.eps)
        mean_sem, std_sem, (train_sem_lat, val_sem_lat_n, val_sem_lat_m) = _normalize_latent_arrays(
            train_sem_lat, (val_sem_lat_n, val_sem_lat_m), cfg.eps)
        epi_latent_mean = _maybe_tolist(mean_epi.squeeze())
        epi_latent_std  = _maybe_tolist(std_epi.squeeze())
        sem_latent_mean = _maybe_tolist(mean_sem.squeeze())
        sem_latent_std  = _maybe_tolist(std_sem.squeeze())
    else:
        mean_epi        = std_epi = mean_sem = std_sem = None
        epi_latent_mean = epi_latent_std = sem_latent_mean = sem_latent_std = None

    combined_model, combined_state, combined_losses, hp_comb    = _train_combined_branch(
        cfg, exp_cfg, train_epi_lat, train_sem_lat, val_epi_lat_n, val_sem_lat_n, rng_combined, schedule, target_comb)
    t_comb                                                      = _train_combined_branch.elapsed

    mcs.print_subsection('Подбор порога реконструкции')
    val_lbls_mixed_arr                  = jp.asarray(data.val_labels_mixed, dtype = jp.int32)
    val_errors                          = Branch.combined_errors(combined_state, combined_model, val_epi_lat_m, val_sem_lat_m)
    threshold_metric                    = exp_cfg.threshold_metric_or(cfg.threshold_metric)
    best_threshold, best_val_metrics    = select_threshold(
        errors = val_errors, labels = val_lbls_mixed_arr, n_thresholds = cfg.n_thresholds, target_metric = threshold_metric, eps = cfg.eps)
    mcs.print_metric(f'Лучший порог по {threshold_metric}', f'{best_threshold:.6f}')
    _ = tuple(starmap(lambda k, v: mcs.print_metric(f'  {k}', f'{v:.4f}'), asdict(best_val_metrics).items()))

    models      = Models(epi_model, epi_state, sem_model, sem_state, combined_model, combined_state)
    calibration = Calibrate.compute(
        models, data,
        best_threshold  = best_threshold, normalize_latent = exp_cfg.normalize_latent,
        epi_mean        = mean_epi, epi_std = std_epi, sem_mean = mean_sem, sem_std = std_sem, eps = cfg.eps)
    mcs.print_subsection('Калибровка confidence')
    _ = tuple(starmap(lambda k, v: mcs.print_metric(k, f'{v:.6f}'), asdict(calibration).items()))

    epi_loss_type, epi_huber    = exp_cfg.get_epi_loss_params()
    sem_loss_type, sem_huber    = exp_cfg.get_sem_loss_params()
    comb_loss_type, comb_huber  = exp_cfg.get_combined_loss_params()

    sem_train_mse_final = Branch.lstm_recon_mse(sem_model, sem_state, data.train_sem_pad,      data.train_sem_mask,      chunk = cfg.encode_chunk)
    sem_val_mse_final   = Branch.lstm_recon_mse(sem_model, sem_state, data.val_sem_pad_normal, data.val_sem_mask_normal, chunk = cfg.encode_chunk)
    comb_train_mse      = Branch.combined_recon_mse(combined_state, combined_model, train_epi_lat, train_sem_lat)
    comb_val_mse        = Branch.combined_recon_mse(combined_state, combined_model, val_epi_lat_n, val_sem_lat_n)

    mcs.print_subsection('Оценка на тестовых данных')
    test_epi_lat    = Branch.encode(epi_model, epi_state, data.test_epi_pad, data.test_epi_mask, chunk = cfg.encode_chunk)
    test_sem_lat    = Branch.encode(sem_model, sem_state, data.test_sem_pad, data.test_sem_mask, chunk = cfg.encode_chunk)
    if exp_cfg.normalize_latent:
        if mean_epi is not None and std_epi is not None and mean_sem is not None and std_sem is not None:
            test_epi_lat    = (test_epi_lat - mean_epi) / std_epi
            test_sem_lat    = (test_sem_lat - mean_sem) / std_sem
        else: raise ValueError('ошибка нормализации латентов')
    
    test_errors     = Branch.combined_errors(combined_state, combined_model, test_epi_lat, test_sem_lat)
    test_preds      = (test_errors > best_threshold).astype(jp.int32)
    test_metrics    = calculate_metrics(jp.asarray(data.test_labels), test_preds, cfg.eps)

    _               = tuple(starmap(
        lambda k, v: mcs.print_metric(f'Test {k}', f'{float(v):.4f}'),
        (
            ('accuracy',    test_metrics.accuracy),
            ('precision',   test_metrics.precision),
            ('recall',      test_metrics.recall),
            ('f1',          test_metrics.f1))))

    loss_plot_path  = save_loss_plot(experiment_name, epi_losses, sem_losses, combined_losses, viz_dir)
    error_plot_path = save_error_distribution(
        test_errors, list(data.test_labels), title = f'Test Error Distribution - {experiment_name}',
        output_dir = viz_dir, name = f'{experiment_name}_test_errors', threshold = float(best_threshold))

    _thr_test, _sweep_test  = _threshold_sweep(
        test_errors, jp.asarray(data.test_labels, dtype = jp.int32), cfg.n_thresholds, jp.asarray(cfg.eps, dtype = jp.float32))
    roc_plot_path           = save_roc(
        (1.0 - _sweep_test.specificity).tolist(), _sweep_test.recall.tolist(), viz_dir, name = f'{experiment_name}_roc')
    sweep_plot_path         = save_threshold_sweep(
        _thr_test.tolist(),
        dict(map(lambda k: (k, jp.asarray(Operators.named(_sweep_test, k)).tolist()), ('precision', 'recall', 'f1', 'youden'))),
        viz_dir, name = f'{experiment_name}_sweep', chosen = float(best_threshold))
    confusion_plot_path     = save_confusion_preds(
        list(data.test_labels), test_preds.tolist(), viz_dir, name = f'{experiment_name}_confusion')

    viz_paths = {
        'loss':      loss_plot_path,
        'error':     error_plot_path,
        'roc':       roc_plot_path,
        'sweep':     sweep_plot_path,
        'confusion': confusion_plot_path}

    _save_branch_pickle  (experiment_dir / 'epi_model.pkl', epi_state.params, hp_epi, epi_losses)
    _save_branch_pickle  (experiment_dir / 'sem_model.pkl', sem_state.params, hp_sem, sem_losses)
    _save_combined_pickle(
        experiment_dir / 'combined_model.pkl', combined_state, hp_comb, best_threshold, asdict(test_metrics),
        exp_cfg.normalize_latent, epi_latent_mean, epi_latent_std, sem_latent_mean, sem_latent_std, combined_losses, calibration)

    exp_config_dict = {
        'name':          experiment_name,
        'learning_rate': exp_cfg.learning_rate,
        'batch_size':    exp_cfg.batch_size,
        'epi_sz_latent': exp_cfg.epi_sz_latent,
        'sem_sz_latent': exp_cfg.sem_sz_latent,
        'epi_decoder':   exp_cfg.epi_decoder_type,
        'sem_decoder':   exp_cfg.sem_decoder_type,
        'epi_loss':      f'{epi_loss_type}, huber_delta={epi_huber}',
        'sem_loss':      f'{sem_loss_type}, huber_delta={sem_huber}',
        'combined_loss': f'{comb_loss_type}, huber_delta={comb_huber}',
        'target_losses': list(target_losses) if target_losses is not None else None,
        'epochs':        exp_cfg.epochs,
        'patience':      exp_cfg.patience}

    results = {
        'experiment':               experiment_name,
        'best_threshold':           float(best_threshold),
        'test_metrics':             asdict(test_metrics),
        'val_metrics':              asdict(best_val_metrics),
        'epi_losses':               epi_losses,
        'sem_losses':               sem_losses,
        'combined_losses':          combined_losses,
        'time_epi':                 float(t_epi),
        'time_sem':                 float(t_sem),
        'time_combined':            float(t_comb),
        'time_latents':             float(t_lat),
        'epi_train_mse':            float(epi_train_mse),
        'epi_val_mse':              float(epi_val_mse),
        'params_norm_epi':          float(params_norm_epi),
        'epi_final_train_mse':      float(epi_train_mse),
        'epi_final_val_mse':        float(epi_val_mse),
        'sem_final_train_mse':      float(sem_train_mse_final),
        'sem_final_val_mse':        float(sem_val_mse_final),
        'combined_final_train_mse': float(comb_train_mse),
        'combined_final_val_mse':   float(comb_val_mse),
        'viz_loss_plot':            str(loss_plot_path)  if loss_plot_path  else '',
        'viz_error_plot':           str(error_plot_path) if error_plot_path else '',
        'config':                   exp_config_dict}

    FileIO.json_write(experiment_dir / 'results.json', results)
    report_path = generate_experiment_report(experiment_dir, experiment_name, exp_config_dict, results, viz_paths)
    mcs.print_metric('HTML-отчёт сохранён', report_path.as_posix())
    return results


def _pick_best(all_results: Tuple[dict, ...], select_metric: MetricName) -> dict:
    return reduce(
        lambda a, b: a if a['test_metrics'][select_metric] >= b['test_metrics'][select_metric] else b,
        all_results)


@benchmark
def train_experiments(cfg: S2Config) -> Tuple[PreparedData, Tuple[dict, ...]]:
    '''обучает все эксперименты, печатает сводные таблицы, сохраняет best_info.json'''
    mcs                         = cfg.output_color_scheme
    train_df, val_df, test_df   = load_prepared_data(cfg)
    data                        = prepare_arrays(train_df, val_df, test_df, cfg.s1_meta, cfg)
    grid                        = (EXPERIMENTS if cfg.experiments is None
                                   else tuple(filter(lambda e: e().name in cfg.experiments, EXPERIMENTS)))
    if not grid:
        raise ValueError(f'ни один эксперимент не совпал с фильтром: {cfg.experiments}')
    all_results                 = tuple(map(lambda exp_cls: run_experiment(exp_cls, data, cfg), grid))
    print_summary_table(mcs, all_results)
    print_config_table (mcs, all_results)
    best            = _pick_best(all_results, cfg.select_metric)
    best_info       = {
        'best_experiment':  best['experiment'],
        'metric':           cfg.select_metric,
        'value':            float(best['test_metrics'][cfg.select_metric]),
        'test_metrics':     best['test_metrics']}
    best_model_dir  = Path(cfg.output_dir) / 'best'
    best_model_dir.mkdir(parents = True, exist_ok = True)
    FileIO.json_write(best_model_dir / 'best_info.json', best_info)
    print_best_summary(mcs, best)
    return data, all_results


def _build_s2_meta(cfg: S2Config, data: PreparedData, all_results: Tuple[dict, ...]) -> S2Meta:
    best            = _pick_best(all_results, cfg.select_metric)
    experiment_name = best['experiment']
    experiment_dir  = Path(cfg.output_dir) / experiment_name
    combined_data   = FileIO.pickle_read(experiment_dir / 'combined_model.pkl')
    epi_hp          = combined_data['config']
    return S2Meta(
        output_dir          = Path(cfg.output_dir).as_posix(),
        experiment_dir      = experiment_dir.as_posix(),
        best_experiment     = experiment_name,
        select_metric       = cfg.select_metric,
        best_metric_value   = float(best['test_metrics'][cfg.select_metric]),
        max_len             = int(data.max_len),
        epi_dim             = int(data.epi_dim),
        sem_dim             = int(data.sem_dim),
        epi_sz_latent       = int(epi_hp.sz_latent_epi),
        sem_sz_latent       = int(epi_hp.sz_latent_sem),
        seq_pad_chunk       = int(cfg.seq_pad_chunk),
        best_threshold      = float(combined_data['best_threshold']),
        normalize_latent    = bool(combined_data['normalize_latent']),
        epi_latent_mean     = tuple(combined_data['epi_latent_mean']) if combined_data['epi_latent_mean'] else None,
        epi_latent_std      = tuple(combined_data['epi_latent_std'])  if combined_data['epi_latent_std']  else None,
        sem_latent_mean     = tuple(combined_data['sem_latent_mean']) if combined_data['sem_latent_mean'] else None,
        sem_latent_std      = tuple(combined_data['sem_latent_std'])  if combined_data['sem_latent_std']  else None,
        test_metrics        = dict(starmap(lambda k, v: (k, float(v)), best['test_metrics'].items())),
        calibration         = dict(combined_data.get('calibration', {})))


def _print_artifact_lines(mcs: ColorSchemeDataScience, cfg: S2Config, res: dict) -> None:
    experiment_name = res['experiment']
    experiment_dir  = cfg.output_dir / experiment_name
    mcs.print_metric(f'Эксперимент {experiment_name}', experiment_dir.as_posix())
    _ = tuple(starmap(
        lambda label, suffix: mcs.print_metric(f'  {label}', (experiment_dir / suffix).as_posix()),
        (
            ('Модель EPI',      'epi_model.pkl'),
            ('Модель SEM',      'sem_model.pkl'),
            ('Модель Combined', 'combined_model.pkl'),
            ('Результаты JSON', 'results.json'),
            ('HTML-отчёт',      f'report_{experiment_name}.html'))))
    mcs.print_metric('  Графики', (cfg.output_dir / 'visualizations' / experiment_name).as_posix())


def print_artifacts_summary(cfg: S2Config, all_results: None | Tuple[dict, ...]) -> None:
    '''печатает сводку путей к артефактам обучения детектора'''
    mcs = cfg.output_color_scheme
    mcs.print_section('СВОДКА АРТЕФАКТОВ')
    mcs.print_metric('Директория моделей',       cfg.output_dir.as_posix())
    mcs.print_metric('Директория визуализаций', (cfg.output_dir / 'visualizations').as_posix())
    mcs.print_metric('Лучшая модель',           (cfg.output_dir / 'best').as_posix())
    if all_results:
        _ = tuple(map(partial(_print_artifact_lines, mcs, cfg), all_results))


def train_detector(s1_meta: S1Meta, overrides: None | dict = None) -> S2Meta:
    '''публичная точка входа этапа обучения детектора, вызывается из run_all'''
    data_dir    = PurePath(s1_meta.output_dir)
    cfg         = S2Config(
        s1_meta             = s1_meta,
        output_dir          = data_dir / 'models',
        output_prefix       = s1_meta.prefix,
        run_id              = s1_meta.run_id,
        output_color_scheme = ColorSchemeDataScienceCold(),
        **(overrides or {}))
    Path(cfg.output_dir).mkdir(parents = True, exist_ok = True)
    inject_color_scheme(globals(), cfg.output_color_scheme)
    data, all_results   = train_experiments(cfg)
    s2_meta             = _build_s2_meta(cfg, data, all_results)
    print_artifacts_summary(cfg, all_results)
    generate_summary_report(cfg, all_results)
    return s2_meta


def _make_inference_state(model: LSTM_AE | FMLP_AE, params: Params, batch_stats: None | Params) -> TrainState:
    return TrainState.create(
        apply_fn    = model.apply, params = params, tx = ox.identity(),
        batch_stats = batch_stats if batch_stats is not None else {})


def _latents_or(value: None | Tuple[float, ...], size: int, fill: float) -> Array:
    return jp.asarray(value) if value is not None else jp.full((1, size), fill, dtype = jp.float32)


def load_models_for_inference(s2_meta: S2Meta) -> Tuple[InferenceMeta, Models]:
    '''восстанавливает модели и параметры из артефактов лучшего эксперимента'''
    experiment_dir  = PurePath(s2_meta.experiment_dir)
    epi_data        = FileIO.pickle_read(experiment_dir / 'epi_model.pkl')
    sem_data        = FileIO.pickle_read(experiment_dir / 'sem_model.pkl')
    combined_data   = FileIO.pickle_read(experiment_dir / 'combined_model.pkl')
    epi_model       = LSTM_AE(epi_data['config'])
    sem_model       = LSTM_AE(sem_data['config'])
    combined_model  = FMLP_AE(combined_data['config'])
    meta            = InferenceMeta(
        max_len             = s2_meta.max_len,
        best_threshold      = float(combined_data['best_threshold']),
        normalize_latent    = bool(s2_meta.normalize_latent),
        epi_latent_mean     = _latents_or(s2_meta.epi_latent_mean, s2_meta.epi_sz_latent, 0.0),
        epi_latent_std      = _latents_or(s2_meta.epi_latent_std,  s2_meta.epi_sz_latent, 1.0),
        sem_latent_mean     = _latents_or(s2_meta.sem_latent_mean, s2_meta.sem_sz_latent, 0.0),
        sem_latent_std      = _latents_or(s2_meta.sem_latent_std,  s2_meta.sem_sz_latent, 1.0),
        calibration         = Calibration(**dict(s2_meta.calibration)))
    models          = Models(
        epi_model       = epi_model,
        epi_state       = _make_inference_state(epi_model,      epi_data['params'],      None),
        sem_model       = sem_model,
        sem_state       = _make_inference_state(sem_model,      sem_data['params'],      None),
        combined_model  = combined_model,
        combined_state  = _make_inference_state(combined_model, combined_data['params'], combined_data.get('batch_stats')))
    return meta, models


def _predict_batch_errors(meta: InferenceMeta, models: Models, epi_padded: Array, epi_mask: Array, sem_padded: Array, sem_mask: Array) -> Array:
    epi_lat = Branch.encode(models.epi_model, models.epi_state, epi_padded, epi_mask)
    sem_lat = Branch.encode(models.sem_model, models.sem_state, sem_padded, sem_mask)
    if meta.normalize_latent:
        epi_lat = (epi_lat - meta.epi_latent_mean) / meta.epi_latent_std
        sem_lat = (sem_lat - meta.sem_latent_mean) / meta.sem_latent_std
    return Branch.combined_errors(models.combined_state, models.combined_model, epi_lat, sem_lat)


@benchmark
def predict_trace(epi_pad: Array, epi_mask: Array, sem_pad: Array, sem_mask: Array, models: Models, meta: InferenceMeta) -> TracePrediction:
    '''инференс одной трассы с confidence'''
    out = Predict.batch(meta, models, epi_pad, epi_mask, sem_pad, sem_mask)
    return TracePrediction(
        reconstruction_error    = float(out.e_comb[0]),
        p_anomaly               = float(out.p_anomaly[0]),
        confidence              = float(out.confidence[0]),
        is_anomaly              = bool(out.is_anomaly[0]))


def detect_anomalies(data: pl.LazyFrame, s2_meta: S2Meta) -> pl.LazyFrame:
    meta, models            = load_models_for_inference(s2_meta)
    df                      = data.collect()
    epi_padded, epi_mask    = Pad.split(df, 'epi_sequence',            s2_meta.epi_dim, s2_meta.max_len, s2_meta.seq_pad_chunk)
    sem_padded, sem_mask    = Pad.split(df, 'sem_sequence_sem_vector', s2_meta.sem_dim, s2_meta.max_len, s2_meta.seq_pad_chunk)
    out                     = Predict.batch(meta, models, epi_padded, epi_mask, sem_padded, sem_mask)
    epi_lat                 = Branch.encode(models.epi_model, models.epi_state, epi_padded, epi_mask)
    sem_lat                 = Branch.encode(models.sem_model, models.sem_state, sem_padded, sem_mask)
    z_epi                   = (epi_lat - meta.epi_latent_mean) / meta.epi_latent_std if meta.normalize_latent else epi_lat
    z_sem                   = (sem_lat - meta.sem_latent_mean) / meta.sem_latent_std if meta.normalize_latent else sem_lat
    return pl.concat((
                df.drop(('epi_sequence', 'sem_sequence_sem_vector')).lazy(),
                pl.DataFrame({
                    'detector_reconstruction_error':    map(jp.asarray, out.e_comb),
                    'detector_e_epi':                   map(jp.asarray, out.e_epi),
                    'detector_e_sem':                   map(jp.asarray, out.e_sem),
                    'detector_p_anomaly':               map(jp.asarray, out.p_anomaly),
                    'detector_confidence':              map(jp.asarray, out.confidence),
                    'detector_z_epi':                   z_epi.tolist(),
                    'detector_z_sem':                   z_sem.tolist(),
                    'detector_is_anomaly':              map(jp.asarray, out.is_anomaly)}).lazy()),
        how = 'horizontal').filter(pl.col('detector_is_anomaly').eq(True)).drop('detector_is_anomaly')


def load_best_model(cfg: S2Config) -> Tuple[InferenceMeta, Models]:
    '''восстанавливает лучшую модель из артефактов train_experiments по best_info.json'''
    best_info       = FileIO.json_read(Path(cfg.output_dir) / 'best' / 'best_info.json')
    experiment_dir  = Path(cfg.output_dir) / best_info['best_experiment']
    epi_data        = FileIO.pickle_read(experiment_dir / 'epi_model.pkl')
    sem_data        = FileIO.pickle_read(experiment_dir / 'sem_model.pkl')
    combined_data   = FileIO.pickle_read(experiment_dir / 'combined_model.pkl')
    epi_model       = LSTM_AE(epi_data['config'])
    sem_model       = LSTM_AE(sem_data['config'])
    combined_model  = FMLP_AE(combined_data['config'])
    meta            = InferenceMeta(
        max_len             = 0,
        best_threshold      = float(combined_data['best_threshold']),
        normalize_latent    = bool(combined_data.get('normalize_latent', False)),
        epi_latent_mean     = _latents_or(combined_data['epi_latent_mean'], int(epi_data['config'].sz_latent), 0.0),
        epi_latent_std      = _latents_or(combined_data['epi_latent_std'],  int(epi_data['config'].sz_latent), 1.0),
        sem_latent_mean     = _latents_or(combined_data['sem_latent_mean'], int(sem_data['config'].sz_latent), 0.0),
        sem_latent_std      = _latents_or(combined_data['sem_latent_std'],  int(sem_data['config'].sz_latent), 1.0),
        calibration         = Calibration(**dict(combined_data['calibration'])))
    models          = Models(
        epi_model       = epi_model,
        epi_state       = _make_inference_state(epi_model,      epi_data['params'],      None),
        sem_model       = sem_model,
        sem_state       = _make_inference_state(sem_model,      sem_data['params'],      None),
        combined_model  = combined_model,
        combined_state  = _make_inference_state(combined_model, combined_data['params'], combined_data.get('batch_stats')))
    return meta, models


def _select_inference_indices(labels: Tuple[int, ...], n_normal: int, n_anomalous: int, seed: int) -> Tuple[int, ...]:
    rnd         = Random(seed)
    indexed     = tuple(enumerate(labels))
    normal_idx  = tuple(map(itemgetter(0), filter(lambda iv: iv[1] == 0, indexed)))
    anom_idx    = tuple(map(itemgetter(0), filter(lambda iv: iv[1] == 1, indexed)))
    take_norm   = min(n_normal,    len(normal_idx))
    take_anom   = min(n_anomalous, len(anom_idx))
    chosen      = list(chain(rnd.sample(normal_idx, take_norm), rnd.sample(anom_idx, take_anom)))
    rnd.shuffle(chosen)
    return tuple(chosen)


def _inference_row(models: Models, meta: InferenceMeta, labels: Tuple[int, ...], data: PreparedData, idx: int) -> dict:
    prediction  = predict_trace(
        data.test_epi_pad [idx:idx + 1], data.test_epi_mask[idx:idx + 1],
        data.test_sem_pad [idx:idx + 1], data.test_sem_mask[idx:idx + 1], models, meta)
    elapsed_ms  = predict_trace.elapsed
    return {
        'Индекс':        int(idx),
        'Истинная':      'аномалия' if labels[idx] == 1 else 'норма',
        'Предсказание':  'аномалия' if prediction.is_anomaly else 'норма',
        'Ошибка':        float(prediction.reconstruction_error),
        'Время (мс)':    float(elapsed_ms)}


def run_inference(cfg: S2Config, data: PreparedData) -> None:
    '''CLI-инференс по случайной выборке тестовых трасс'''
    mcs = cfg.output_color_scheme
    mcs.print_section('ИНФЕРЕНС НА СЛУЧАЙНЫХ ТЕСТОВЫХ ТРАССАХ')
    inference_meta, models  = load_best_model(cfg)
    inference_meta          = replace(inference_meta, max_len = int(data.max_len))
    test_labels             = tuple(map(int, data.test_labels))
    chosen                  = _select_inference_indices(test_labels, cfg.inference_normal_count, cfg.inference_anomalous_count, cfg.seed)
    rows                    = tuple(map(partial(_inference_row, models, inference_meta, test_labels, data), chosen))
    print_table(
        pl.DataFrame(list(rows)),
        cols = ('Индекс', 'Истинная', 'Предсказание', 'Ошибка', 'Время (мс)'), float_fmt = '.6f', scheme = mcs)
    y_true  = tuple(map(lambda i: 1 if test_labels[i] == 1 else 0, chosen))
    y_pred  = tuple(map(lambda r: 1 if r['Предсказание'] == 'аномалия' else 0, rows))
    errors  = tuple(map(lambda r: r['Ошибка'], rows))
    metrics = calculate_metrics(jp.asarray(y_true), jp.asarray(y_pred), cfg.eps)

    def _safe_auc(fn: Callable, yt: Tuple[int, ...], ys: Tuple[float, ...]) -> float:
        try:                return float(fn(yt, ys))
        except ValueError:  return float('nan')

    roc_auc = _safe_auc(roc_auc_score,           y_true, errors)
    pr_auc  = _safe_auc(average_precision_score, y_true, errors)
    mcs.print_subsection('Метрики')
    sprint(f'Accuracy:    {metrics.accuracy:.4f}',  style_code = mcs.success)
    sprint(f'Precision:   {metrics.precision:.4f}', style_code = mcs.success)
    sprint(f'Recall:      {metrics.recall:.4f}',    style_code = mcs.success)
    sprint(f'F1:          {metrics.f1:.4f}',        style_code = mcs.success)
    sprint(f'ROC-AUC:     {roc_auc:.4f}',           style_code = mcs.info)
    sprint(f'PR-AUC:      {pr_auc:.4f}',            style_code = mcs.info)


def _load_s1_meta_from_dir(data_dir: Path, prefix: str) -> S1Meta:
    candidates = tuple(filter(lambda p: p.name.startswith(prefix), Path(data_dir).glob('*_meta.json')))
    if not candidates:
        raise FileNotFoundError(f'meta.json (prefix={prefix}) не найден в {data_dir}')
    return S1Meta(**FileIO.json_read(candidates[0]))


@benchmark('этап 2: обучение детектора аномалий')
def main(data_dir: Path, do_train: bool, do_inference: bool, output_dir: None | Path, output_prefix: str) -> None:
    '''CLI-точка входа для standalone запуска этапа детектора'''
    s1_meta = _load_s1_meta_from_dir(data_dir, output_prefix)
    cfg     = S2Config(
        s1_meta         = s1_meta, output_dir = (output_dir or (data_dir / 'models')),
        output_prefix   = output_prefix, run_id = s1_meta.run_id)
    Path(cfg.output_dir).mkdir(parents = True, exist_ok = True)
    inject_color_scheme(globals(), cfg.output_color_scheme)
    data        = None
    all_results = None
    if do_train:
        data, all_results = train_experiments(cfg)
    if do_inference:
        if data is None:
            train_df, val_df, test_df   = load_prepared_data(cfg)
            data                        = prepare_arrays(train_df, val_df, test_df, cfg.s1_meta, cfg)
        run_inference(cfg, data)
    if do_train:                    print_artifacts_summary(cfg, all_results)
    if do_train and all_results:    generate_summary_report(cfg, all_results)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description = 'обучение и инференс детектора аномалий')
    parser.add_argument('--data_dir',       type = Path,    required = True,            help = 'каталог с артефактами s1_data')
    parser.add_argument('--train',          action = 'store_true',                      help = 'запустить обучение')
    parser.add_argument('--inference',      action = 'store_true',                      help = 'запустить инференс')
    parser.add_argument('--output_dir',     type = Path,    default = None,             help = 'каталог для сохранения моделей')
    parser.add_argument('--output_prefix',  type = str,     default = 'traces_fa_',     help = 'префикс выходных файлов')
    args = parser.parse_args()
    if not (args.train or args.inference):
        parser.error('укажите хотя бы один из флагов: --train или --inference')
    main(
        data_dir    = args.data_dir, do_train = args.train, do_inference = args.inference,
        output_dir  = args.output_dir, output_prefix = args.output_prefix)