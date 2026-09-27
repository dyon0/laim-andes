from    typing                                  import Tuple, Callable, Literal, Mapping, Sequence
from    dataclasses                             import dataclass, make_dataclass
from    functools                               import reduce

from    ars.models.metrics                      import MetricName, LossKind
from    ars.models.m2__detector.architecture    import Direction, Activation, LayersArch


type LayersDims = Tuple[int, ...]
type LossOpt    = None | LossKind


@dataclass(frozen = True)
class Experiment:
    name    : str   = ''

    epi_layers_arch         : LayersArch[Direction]     = ()
    sem_layers_arch         : LayersArch[Direction]     = ()
    combined_layers_arch    : LayersArch[Activation]    = ()

    epi_decoder_type    : Literal['autoregressive', 'linear']   = 'autoregressive'
    epi_sz_latent       : int                                   = 768
    sem_decoder_type    : Literal['autoregressive', 'linear']   = 'autoregressive'
    sem_sz_latent       : int                                   = 768

    use_batch_norm      : bool  = True
    normalize_latent    : bool  = True

    epi_loss_type           : LossOpt                           = None
    epi_huber_delta         : float                             = 1.0
    sem_loss_type           : LossOpt                           = None
    sem_huber_delta         : float                             = 1.0
    combined_loss_type      : LossOpt                           = None
    combined_huber_delta    : float                             = 1.0
    loss_type               : LossKind                          = 'mse'
    huber_delta             : float                             = 1.0
    target_losses           : None | Tuple[float, float, float] = None

    learning_rate   : float = 1e-4
    epochs          : int   = 10 #500
    patience        : int   = 50
    batch_size      : int   = 8
    dropout_rate    : float = 0.2
    weight_decay    : float = 1e-6
    clip_grad       : float = 1.0

    select_metric       : None | MetricName = None
    threshold_metric    : None | MetricName = None

    def get_epi_loss_params(self) -> Tuple[LossKind, float]:
        return (self.epi_loss_type if self.epi_loss_type is not None else self.loss_type,
                self.epi_huber_delta if self.epi_loss_type is not None else self.huber_delta)

    def get_sem_loss_params(self) -> Tuple[LossKind, float]:
        return (self.sem_loss_type if self.sem_loss_type is not None else self.loss_type,
                self.sem_huber_delta if self.sem_loss_type is not None else self.huber_delta)

    def get_combined_loss_params(self) -> Tuple[LossKind, float]:
        return (self.combined_loss_type if self.combined_loss_type is not None else self.loss_type,
                self.combined_huber_delta if self.combined_loss_type is not None else self.huber_delta)

    def threshold_metric_or(self, default: MetricName) -> MetricName:
        return self.threshold_metric if self.threshold_metric is not None else default

    def select_metric_or(self, default: MetricName) -> MetricName:
        return self.select_metric if self.select_metric is not None else default


_shrink = lambda acc, _: acc + (acc[-1] // 2,)
_repair = lambda acc, _: acc + (acc[-1]  * 2,)


def _round_to_power2(x: int) -> int:
    if x <= 0: return 1
    lower   = 1 << (x.bit_length() - 1)
    upper   = lower << 1
    return lower if (x - lower) < (upper - x) else upper


def _make_deep_fmlp(
        acts        : Tuple[Literal['relu', 'tanh'], ...],
        n_steps     : int,
        init_dim    : int,
        mid_dim     : int,
        out_dim     : None | int = None,
        direction   : Callable[[LayersDims, int], LayersDims] = _shrink,
        normalize   : bool = False,
) -> LayersArch:
    assert len(acts) == n_steps * 2 + 1 + (1 if out_dim is not None else 0)
    down    = reduce(direction, range(n_steps), (init_dim,))
    up      = tuple(reversed(down))
    dims    = down + (mid_dim,) + up + ((out_dim,) if out_dim is not None else ())
    return tuple(zip(acts, map(_round_to_power2 if normalize else (lambda x: x), dims)))


def _make_deep_lstm(
        n_steps     : int,
        init_dim    : int,
        mid_dim     : int,
        out_dim     : None | int = None,
        alternate   : bool = True,
        start_dir   : Literal['unidirectional', 'bidirectional'] = 'bidirectional',
        direction   : Callable[[LayersDims, int], LayersDims] = _shrink,
        normalize   : bool = False,
) -> LayersArch:
    down    = reduce(direction, range(n_steps), (init_dim,))
    up      = tuple(reversed(down))
    dims    = down + (mid_dim,) + up + ((out_dim,) if out_dim is not None else ())
    u       = 'unidirectional'
    b       = 'bidirectional'
    n       = len(dims)
    dirs    = (tuple(map(lambda i: u if (i % 2) ^ (start_dir == u) else b, range(n)))
               if alternate else (start_dir,) * n)
    return tuple(zip(dirs, map(_round_to_power2 if normalize else (lambda x: x), dims)))


def _arch_default() -> Tuple[LayersArch, LayersArch, LayersArch, int, int]:
    return (
        (('unidirectional',  64), ('bidirectional',   32)),
        (('bidirectional',  512), ('unidirectional', 256)),
        (('relu', 128), ('tanh', 32), ('relu', 64)),
        128, 768)


def _arch_deep() -> Tuple[LayersArch, LayersArch, LayersArch, int, int]:
    return (
        _make_deep_lstm(n_steps = 3, init_dim = 64,  mid_dim = 32,  out_dim = 256,
                        alternate = True, start_dir = 'unidirectional'),
        _make_deep_lstm(n_steps = 5, init_dim = 768, mid_dim = 256, out_dim = 512,
                        alternate = True, start_dir = 'bidirectional'),
        _make_deep_fmlp(
            acts = ('relu', 'relu', 'tanh', 'relu', 'relu', 'tanh',
                    'relu', 'relu', 'relu', 'tanh', 'relu', 'relu', 'tanh', 'relu'),
            n_steps = 6, init_dim = 768, mid_dim = 32, out_dim = 64),
        128, 768)


def _arch_wide() -> Tuple[LayersArch, LayersArch, LayersArch, int, int]:
    return (
        _make_deep_lstm(n_steps = 2, init_dim = 512,  mid_dim = 256, out_dim = None,
                        alternate = False, start_dir = 'unidirectional', direction = _repair, normalize = True),
        _make_deep_lstm(n_steps = 1, init_dim = 2048, mid_dim = 512, out_dim = None,
                        alternate = False, start_dir = 'bidirectional',  direction = _repair, normalize = True),
        _make_deep_fmlp(
            acts    = ('relu', 'tanh', 'relu', 'relu', 'tanh', 'relu'),
            n_steps = 2, init_dim = 768, mid_dim = 768, out_dim = 512, direction = _repair, normalize = True),
        128, 768)


def _arch_builder(name: str) -> Tuple[LayersArch, LayersArch, LayersArch, int, int]:
    match name:
        case 'default': return _arch_default()
        case 'deep':    return _arch_deep()
        case 'wide':    return _arch_wide()
        case _:         raise ValueError(f'неизвестная архитектура: {name}')


def _loss_code(code: str) -> LossKind:
    match code:
        case 'mse': return 'mse'
        case 'hub': return 'huber'
        case _:     raise ValueError(f'неизвестный код потерь: {code}')


CODE_FORMAT = ('{epi}_{sem}_{comb}_{batch}_{lr}[_{arch}], где epi/sem/comb ∈ {mse, hub}, '
               'batch — целое > 0, lr — показатель степени (4 -> 1e-4), '
               'arch ∈ {default, deep, wide}; пример: hub_mse_mse_08_4, hub_mse_hub_32_4_deep')


def parse_code(code: str) -> Tuple[str, str, str, int, int, str]:
    '''F-78: разбор кода эксперимента с понятной ошибкой вместо IndexError /
    голого int(); годится любой корректный код, не только перечисленные в CODES'''
    def _bad(reason: str) -> ValueError:
        return ValueError(f'некорректный код эксперимента {code!r}: {reason}. Формат: {CODE_FORMAT}')

    parts = str(code).strip().split('_')
    if len(parts) < 5:
        raise _bad(f'ожидается не меньше 5 частей через "_", получено {len(parts)}')
    epi, sem, comb, batch, lr = parts[:5]
    arch = '_'.join(parts[5:]) or 'default'
    if (bad := next(filter(lambda c: c not in ('mse', 'hub'), (epi, sem, comb)), None)) is not None:
        raise _bad(f'неизвестный код потерь {bad!r}')
    if not batch.isdigit() or int(batch) <= 0:
        raise _bad(f'batch {batch!r} не целое положительное число')
    if not lr.isdigit():
        raise _bad(f'показатель learning rate {lr!r} не целое неотрицательное число')
    if arch not in ('default', 'deep', 'wide'):
        raise _bad(f'неизвестная архитектура {arch!r}')
    return epi, sem, comb, int(batch), int(lr), arch


def _experiment_from_code(code: str, metric: None | MetricName = None) -> type[Experiment]:
    # F-77: metric=None keeps detector.threshold_metric / select_metric from the
    # config in force; a per-experiment metric is an explicit override only
    epi, sem, comb, batch, lr, arch                         = parse_code(code)
    epi_arch, sem_arch, comb_arch, epi_latent, sem_latent   = _arch_builder(arch)
    overrides                                               = (
        ('name',                 str,                code),
        ('epi_layers_arch',      LayersArch,         epi_arch),
        ('sem_layers_arch',      LayersArch,         sem_arch),
        ('combined_layers_arch', LayersArch,         comb_arch),
        ('epi_sz_latent',        int,                epi_latent),
        ('sem_sz_latent',        int,                sem_latent),
        ('epi_loss_type',        LossOpt,            _loss_code(epi)),
        ('sem_loss_type',        LossOpt,            _loss_code(sem)),
        ('combined_loss_type',   LossOpt,            _loss_code(comb)),
        ('batch_size',           int,                batch),
        ('learning_rate',        float,              10.0 ** (-lr)),
        ('threshold_metric',     None | MetricName,  metric),
        ('select_metric',        None | MetricName,  metric))
    return make_dataclass(code, overrides, bases = (Experiment,), frozen = True)


def build_grid(codes: Tuple[str, ...], metrics: None | Mapping[str, MetricName] = None) -> Tuple[type[Experiment], ...]:
    '''F-77: experiments carry NO metric unless `metrics` names one for their
    code explicitly (the old per-position cycling youden/precision/recall/f1
    silently overrode detector.threshold_metric). Duplicate codes collapse to
    one experiment (same name = same output directory).'''
    explicit = dict(metrics or {})
    unique   = tuple(dict.fromkeys(map(lambda c: str(c).strip(), codes)))
    return tuple(map(lambda code: _experiment_from_code(code, explicit.get(code)), unique))


def resolve_grid(experiments: None | Sequence[str]) -> Tuple[type[Experiment], ...]:
    '''F-78: the grid is built FROM the requested codes (any well-formed code,
    not only the ones listed in CODES); None keeps the compiled default grid.
    Malformed codes raise ValueError naming the expected format.'''
    if experiments is None:
        return EXPERIMENTS
    grid = build_grid(tuple(experiments))
    if not grid:
        raise ValueError(f'список экспериментов пуст. Формат кода: {CODE_FORMAT}')
    return grid


@dataclass(frozen = True)
class StackSpec:
    enabled                     : bool  = False
    meta_solver                 : str   = 'logreg'
    n_folds                     : int   = 5
    seed                        : int   = 12345
    use_confidence_features     : bool  = True
    use_calibration_features    : bool  = True


CODES   : Tuple[str, ...]   = (
    # 'mse_mse_mse_08_5',
    # 'mse_mse_mse_16_4',
    # 'mse_mse_mse_32_3',
    'hub_mse_mse_08_4',
    # 'hub_mse_mse_16_3',
    # 'hub_mse_mse_32_5',
    # 'mse_mse_hub_08_3',
    # 'mse_mse_hub_16_5',
    # 'mse_mse_hub_32_4',
    # 'hub_mse_hub_08_5',
    # 'hub_mse_hub_16_4',
    # 'hub_mse_hub_32_3',
    # 'mse_mse_mse_08_4_deep',
    # 'mse_mse_mse_16_3_deep',
    'hub_mse_hub_32_4_deep',
    # 'mse_mse_mse_08_4_wide',
    # 'mse_mse_mse_16_3_wide',
    # 'hub_mse_hub_32_4_wide',
)

EXPERIMENTS : Tuple[type[Experiment], ...]  = build_grid(CODES)

STACK   : StackSpec = StackSpec(enabled = True, meta_solver = 'logreg', n_folds = 5)


def show_experiment(e: type[Experiment]) -> None:
    x = e()
    print(x.name)
    print(f'epi -> {x.epi_layers_arch} -> {x.epi_sz_latent}')
    print(f'sem -> {x.sem_layers_arch} -> {x.sem_sz_latent}')
    print(f'cmb {x.epi_sz_latent + x.sem_sz_latent} -> {x.combined_layers_arch}')
    print(f'loss epi/sem/cmb = {x.get_epi_loss_params()[0]}/{x.get_sem_loss_params()[0]}/{x.get_combined_loss_params()[0]}'
          f' | lr={x.learning_rate} batch={x.batch_size} metric={x.threshold_metric}')
    print()