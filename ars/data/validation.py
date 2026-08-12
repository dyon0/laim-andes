from    typing                          import ClassVar, Iterator, Callable
from    dataclasses                     import dataclass, field
from    itertools                       import chain, starmap
from    functools                       import reduce, partial
from    operator                        import add, or_, attrgetter

from    math                            import log10
from    pathlib                         import PurePath
from    tempfile                        import mkdtemp

import  polars                          as pl
import  altair                          as al

from    ars.configuration.c0__device    import Device
from    ars.specification.core          import SpansDataSpec, SpanAttr
from    ars.specification.spec          import SpansData, Recast, recast, DType
from    ars.tools.performance           import perf
from    ars.tools.visualisations        import viz
from    ars.tools.tui.tui               import ColorSchemeDataScience
from    ars.tools.tui.tui_data          import ColorSchemeDataScienceCold


type Aggregate  = Callable[[pl.Expr], pl.Expr]
type MetricFn   = Callable[[SpansDataSpec, Resolution], Iterator[pl.Expr]]


@dataclass(frozen = True)
class Ns:
    per_second  : ClassVar[int] = 1_000_000_000
    per_day     : ClassVar[int] = 86_400 * per_second


@dataclass(frozen = True)
class Num:
    types   : ClassVar[tuple[type[pl.DataType], ...]]   = (pl.Int64, pl.Float32, pl.Float64)


@dataclass(frozen = True, eq = True)
class Resolution:
    percentiles : tuple[float, ...] = (0.25, 0.50, 0.75, 0.90, 0.95)
    grains      : tuple[str, ...]   = ('1d', '4h')
    windows     : tuple[int, ...]   = (10, 50, 100)


@dataclass(frozen = True, eq = False)
class Dim:
    name    : str
    expr    : pl.Expr

    def keyed(self) -> pl.Expr:
        return self.expr.alias(self.name)


@dataclass(frozen = True, eq = False)
class By:
    trace       : ClassVar[Dim] = Dim('trace_id',   pl.col('trace_id'))
    agent       : ClassVar[Dim] = Dim('agent_id',   pl.col('agent_id'))
    session     : ClassVar[Dim] = Dim('session_id', pl.col('session_id'))
    ver_major   : ClassVar[Dim] = Dim('ver_major',  pl.col('nexus_distrib_ver').str.extract(r'^(\d+)', 1))
    ver_minor   : ClassVar[Dim] = Dim('ver_minor',  pl.col('nexus_distrib_ver').str.extract(r'^(\d+\.\d+)', 1))
    ver_patch   : ClassVar[Dim] = Dim('ver_patch',  pl.col('nexus_distrib_ver'))

    @staticmethod
    def time(grain: str) -> Dim:
        return Dim('ts', pl.from_epoch(pl.col('start_time_ns'), time_unit = 'ns').dt.truncate(grain))


@dataclass(frozen = True, eq = False)
class Group:
    @staticmethod
    def keyed(dims: tuple[Dim, ...]) -> tuple[pl.Expr, ...]:
        return tuple(map(Dim.keyed, dims))

    @staticmethod
    def named(dims: tuple[Dim, ...]) -> tuple[str, ...]:
        return tuple(map(attrgetter('name'), dims))

    @staticmethod
    def aggregate(lf: pl.LazyFrame, dims: tuple[Dim, ...], body: Iterator[pl.Expr]) -> pl.LazyFrame:
        keys = Group.keyed(dims)

        return lf.select(body) if not keys else lf.group_by(keys).agg(body)


@dataclass(frozen = True, eq = False)
class FromAttr:
    @staticmethod
    def real_values(a: SpanAttr) -> pl.Expr:
        col = pl.col(a.name)

        return col if a.empty is None else col.filter(col != a.empty)

    @staticmethod
    def missing(a: SpanAttr) -> pl.Expr:
        col = pl.col(a.name)

        return col.is_null() | ((col == a.empty) if a.empty is not None else pl.lit(False))

    @staticmethod
    def trash(a: SpanAttr) -> pl.Expr:
        return (pl.col(a.name) == a.trash) if a.trash is not None else pl.lit(False)

    @staticmethod
    def violation(a: SpanAttr) -> pl.Expr:
        return (~a.validity).fill_null(True)


@dataclass(frozen = True, eq = False)
class Across:
    @staticmethod
    def missing(attrs: tuple[SpanAttr, ...], dtype: type[pl.DataType] = pl.Int64) -> pl.Expr:
        return reduce(add, map(lambda a: FromAttr.missing(a).cast(dtype), attrs))

    @staticmethod
    def violations(attrs: tuple[SpanAttr, ...], dtype: type[pl.DataType] = pl.Int64) -> pl.Expr:
        return reduce(add, map(lambda a: FromAttr.violation(a).cast(dtype), attrs))

    @staticmethod
    def nulls(attrs: tuple[SpanAttr, ...], dtype: type[pl.DataType] = pl.Int64) -> pl.Expr:
        return reduce(add, map(lambda a: pl.col(a.name).is_null().cast(dtype), attrs))

    @staticmethod
    def trash(attrs: tuple[SpanAttr, ...], dtype: type[pl.DataType] = pl.Int64) -> pl.Expr:
        return reduce(add, map(lambda a: FromAttr.trash(a).cast(dtype), attrs))

    @staticmethod
    def violation_rate(attrs: tuple[SpanAttr, ...]) -> pl.Expr:
        return Across.violations(attrs, pl.Float64) / len(attrs)

    @staticmethod
    def null_rate(attrs: tuple[SpanAttr, ...]) -> pl.Expr:
        return Across.nulls(attrs, pl.Float64) / len(attrs)


@dataclass(frozen = True, eq = False)
class FromSpec:
    @staticmethod
    def numeric_attrs(spec: SpansDataSpec) -> Iterator[SpanAttr]:
        return filter(lambda a: a.type_polars in Num.types, spec.attrs)

    @staticmethod
    def mandatory(spec: SpansDataSpec) -> tuple[SpanAttr, ...]:
        return tuple(filter(lambda a: a.mandatory, spec.attrs))

    @staticmethod
    def distribution_signals(spec: SpansDataSpec) -> Iterator[tuple[str, pl.Expr]]:
        return chain(
            map(lambda a: (a.name, FromAttr.real_values(a)), FromSpec.numeric_attrs(spec)),
            (('duration_ns', pl.col('end_time_ns') - pl.col('start_time_ns')),))

    @staticmethod
    def dynamic_signals(spec: SpansDataSpec) -> tuple[tuple[str, pl.Expr], ...]:
        return (
            ('missing',  Across.missing(spec.attrs)),
            ('invalid',  Across.violations(spec.attrs)),
            ('duration', pl.col('end_time_ns') - pl.col('start_time_ns')),
            ('error',    (pl.col('status_code') == 'STATUS_CODE_ERROR').cast(pl.Int64)))


@dataclass(frozen = True, eq = False)
class Metric:
    @staticmethod
    def completeness(spec: SpansDataSpec, resolution: Resolution, agg: Aggregate = pl.Expr.mean) -> Iterator[pl.Expr]:
        return map(lambda a: agg(FromAttr.missing(a)).alias(a.name), spec.attrs)

    @staticmethod
    def correctness(spec: SpansDataSpec, resolution: Resolution, agg: Aggregate = pl.Expr.mean) -> Iterator[pl.Expr]:
        return map(lambda a: agg(FromAttr.violation(a)).alias(a.name), spec.attrs)

    @staticmethod
    def contamination(spec: SpansDataSpec, resolution: Resolution, agg: Aggregate = pl.Expr.mean) -> Iterator[pl.Expr]:
        return map(lambda a: agg(FromAttr.trash(a)).alias(a.name), spec.attrs)

    @staticmethod
    def nulls(spec: SpansDataSpec, resolution: Resolution, agg: Aggregate = pl.Expr.sum) -> Iterator[pl.Expr]:
        return map(lambda a: agg(pl.col(a.name).is_null()).alias(a.name), spec.attrs)

    @staticmethod
    def distribution(spec: SpansDataSpec, resolution: Resolution) -> Iterator[pl.Expr]:
        quantiles = lambda name, value: map(
            lambda q: value.quantile(q, interpolation = 'linear').alias(f'{name}__q{int(q * 100):02d}'),
            resolution.percentiles)

        return chain.from_iterable(starmap(quantiles, FromSpec.distribution_signals(spec)))


@dataclass(frozen = True, eq = False)
class Quality:
    scanned     : pl.LazyFrame
    spec        : SpansDataSpec = SpansData
    resolution  : Resolution    = field(default_factory = Resolution)

    invariants  : ClassVar[tuple[str, ...]] = (
        'agent_id', 'session_id', 'service_name', 'service_version', 'nexus_distrib_ver')

    def profile(self, metric: MetricFn, *dims: Dim, grain: None | str = None) -> pl.LazyFrame:
        graining    = (By.time(grain),) if grain else ()
        keys        = frozenset(Group.named(dims + graining))
        body        = chain(
            (pl.len().alias('n'),),
            filter(lambda e: e.meta.output_name() not in keys, metric(self.spec, self.resolution)))

        return Group.aggregate(self.scanned, dims + graining, body)

    def completeness(self, *dims: Dim, grain: None | str = None, agg: Aggregate = pl.Expr.mean) -> pl.LazyFrame:
        return self.profile(partial(Metric.completeness, agg = agg), *dims, grain = grain)

    def correctness(self, *dims: Dim, grain: None | str = None, agg: Aggregate = pl.Expr.mean) -> pl.LazyFrame:
        return self.profile(partial(Metric.correctness, agg = agg), *dims, grain = grain)

    def contamination(self, *dims: Dim, grain: None | str = None, agg: Aggregate = pl.Expr.mean) -> pl.LazyFrame:
        return self.profile(partial(Metric.contamination, agg = agg), *dims, grain = grain)

    def nulls(self, *dims: Dim, grain: None | str = None, agg: Aggregate = pl.Expr.sum) -> pl.LazyFrame:
        return self.profile(partial(Metric.nulls, agg = agg), *dims, grain = grain)

    def distribution(self, *dims: Dim, grain: None | str = None) -> pl.LazyFrame:
        return self.profile(Metric.distribution, *dims, grain = grain)

    def tagged(self, over: Dim = By.trace) -> pl.LazyFrame:
        rejected    = reduce(or_, map(FromAttr.violation, FromSpec.mandatory(self.spec)))
        flags       = self.scanned.group_by(over.name).agg(rejected.any().alias('rejected'))

        return self.scanned.join(flags, on = over.name, how = 'left')

    def rejection(self, *dims: Dim, grain: None | str = None, over: Dim = By.trace, agg: Aggregate = pl.Expr.mean) -> pl.LazyFrame:
        graining    = (By.time(grain),) if grain else ()
        rejected    = pl.col('rejected')
        body        = (
            pl.len().alias('n'),
            agg(rejected).alias('dropped_rate'),
            rejected.sum().cast(pl.UInt64).alias('dropped'),
            (~rejected).sum().cast(pl.UInt64).alias('kept'))

        return Group.aggregate(self.tagged(over), dims + graining, iter(body))

    def trace_status(self) -> pl.LazyFrame:
        canon   = map(lambda c: pl.first(c).alias(c), ('session_id', 'nexus_distrib_ver'))
        agreed  = reduce(add, map(lambda c: (pl.col(c).n_unique() > 1).cast(pl.Int64), self.invariants)) == 0
        body    = chain(canon, (
            pl.len().alias('spans'),
            pl.col('rejected').max().alias('rejected'),
            agreed.alias('consistent')))

        return self.tagged().group_by('trace_id').agg(body)

    def consistency(self, *cols: str) -> pl.LazyFrame:
        targets = cols or self.invariants
        counts  = map(lambda c: pl.col(c).n_unique().alias(c), targets)

        return self.scanned.group_by('trace_id').agg(chain((pl.len().alias('spans'),), counts))

    def trace_consistency(self, *cols: str) -> pl.LazyFrame:
        targets = cols or self.invariants
        broken  = map(lambda c: (pl.col(c).n_unique() > 1).alias(c), targets)
        rolled  = self.scanned.group_by('trace_id').agg(broken)

        return rolled.select(map(lambda c: pl.col(c).sum().cast(pl.UInt64).alias(c), targets))

    def dynamics(self, window: None | int = None) -> pl.LazyFrame:
        w       = window or self.resolution.windows[0]
        order   = 'start_time_ns'
        seq     = pl.col('span_id').cum_count().over('trace_id', order_by = order)

        def running(name: str, signal: pl.Expr) -> tuple[pl.Expr, ...]:
            cumulative = signal.cum_sum().over('trace_id', order_by = order)

            return (
                signal.alias(name),
                cumulative.alias(f'{name}__cum'),
                (cumulative / seq).alias(f'{name}__mean'),
                signal.rolling_mean(window_size = w, min_samples = 1).over('trace_id', order_by = order).alias(f'{name}__roll{w}'))

        columns = chain.from_iterable(starmap(running, FromSpec.dynamic_signals(self.spec)))

        return self.scanned.with_columns(chain((seq.alias('span_seq'),), columns))

    def verdict(self, *, max_nulls: int = 0, max_rules: int = 0, max_dups: int = 0, max_breaks: int = 0) -> pl.LazyFrame:
        attrs   = self.spec.attrs
        nulls   = Across.nulls(attrs).sum().cast(pl.UInt64)
        miss    = Across.missing(attrs).sum().cast(pl.UInt64)
        trash   = Across.trash(attrs).sum().cast(pl.UInt64)
        rules   = Across.violations(attrs).sum().cast(pl.UInt64)
        dups    = (pl.len() - pl.struct('trace_id', 'span_id').n_unique()).cast(pl.UInt64)
        broken  = lambda c: pl.col('trace_id').filter(pl.col(c).n_unique().over('trace_id') > 1).n_unique()
        breaks  = reduce(add, map(broken, self.invariants)).cast(pl.UInt64)

        return self.scanned.select(
            nulls.alias('null_violations'),
            miss.alias('missing'),
            trash.alias('trash_cells'),
            rules.alias('rule_violations'),
            dups.alias('duplicate_keys'),
            breaks.alias('trace_breaks'),
            (  (nulls  <= max_nulls)
             & (rules  <= max_rules)
             & (dups   <= max_dups)
             & (breaks <= max_breaks)).alias('valid'))

    def is_valid(self, **thresholds: int) -> bool:
        return bool(self.verdict(**thresholds).collect().get_column('valid').item())

    def flow(self, grain: str = '1d') -> pl.LazyFrame:
        viol = Across.violation_rate(self.spec.attrs)

        return (self.scanned
            .group_by(By.time(grain).keyed())
            .agg(
                pl.col('trace_id').n_unique().alias('traces'),
                (1 - viol.mean()).alias('validity'))
            .sort('ts'))

def scan(path: PurePath, spec: SpansDataSpec = SpansData) -> pl.LazyFrame:
    lazy = pl.scan_parquet(path.as_posix(), rechunk = True)

    return Recast.frame(lazy, lazy.collect_schema(), spec)


@dataclass(frozen = True, eq = False)
class SchemaState:
    present         : ClassVar[str]             = 'ок'
    converted       : ClassVar[str]             = 'приведён'
    absent          : ClassVar[str]             = 'отсутствует'
    incompatible    : ClassVar[str]             = 'несовместим'
    broken          : ClassVar[tuple[str, ...]] = (absent, incompatible)


@dataclass(frozen = True, eq = False)
class SchemaCheck:
    @staticmethod
    def name(dtype: None | DType) -> str:
        return '—' if dtype is None else str(dtype)

    @staticmethod
    def state(source: None | DType, target: DType) -> str:
        return (SchemaState.absent          if source is None
                else SchemaState.incompatible   if not Recast.castable(source, target)
                else SchemaState.present        if source == target
                else SchemaState.converted)

    @staticmethod
    def row(a: SpanAttr, mandatory: frozenset[str], schema: pl.Schema) -> tuple:
        source = schema.get(a.name)

        return (a.name, a.name in mandatory, SchemaCheck.name(source),
                SchemaCheck.name(a.type_polars), SchemaCheck.state(source, a.type_polars))

    @staticmethod
    def diff(schema: pl.Schema, spec: SpansDataSpec, preset: 'Preset') -> dict:
        names       = frozenset(spec.column_names)
        mandatory   = spec.mandatory_names
        rows        = tuple(map(lambda a: SchemaCheck.row(a, mandatory, schema), spec.attrs))
        broken      = lambda r: r[4] in SchemaState.broken

        crit        = tuple(filter(lambda r: r[1], rows))
        soft        = tuple(filter(lambda r: not r[1], rows))
        crit_bad    = tuple(filter(broken, crit))
        soft_bad    = tuple(filter(broken, soft))
        extra       = tuple(filter(lambda c: c not in names, schema.names()))
        critical_ok = len(crit_bad) == 0
        ceiling     = max(0.0, min(1.0, 1.0 - preset.ceiling_penalty * len(soft_bad)))
        degraded    = ((not critical_ok) or bool(soft_bad) or bool(extra)
                       or any(map(lambda r: r[4] == SchemaState.converted, rows)))

        return {
            'rows'          : rows,
            'crit'          : crit,
            'soft_bad'      : soft_bad,
            'extra'         : extra,
            'critical_ok'   : critical_ok,
            'ceiling'       : ceiling,
            'degraded'      : degraded,
            'counts'        : {
                'колонок по спецификации'   : len(spec.column_names),
                'из них присутствует'       : len(frozenset(schema.names()) & names),
                'критичных (красных)'       : len(mandatory),
                'критичных нарушено'        : len(crit_bad),
                'некритичных с дефектом'    : len(soft_bad),
                'лишних колонок'            : len(extra),
                'потолок качества, %'       : round(ceiling * 100)}}


@dataclass(frozen = True, eq = False)
class Preset:
    volume_target   : float
    min_eff_traces  : float
    max_loss        : float
    min_quality     : float
    ready_at        : float
    pilot_at        : float
    trend_gain      : float
    rate_target     : float
    stability_tol   : float
    ceiling_penalty : float


@dataclass(frozen = True, eq = False)
class Significance:
    A   : ClassVar[Preset]  = Preset(1e9, 1e6, 0.02, 0.98, 0.92, 0.80, 4.0, 1e5, 0.05, 0.30)
    B   : ClassVar[Preset]  = Preset(1e7, 1e5, 0.07, 0.93, 0.85, 0.70, 3.5, 1e4, 0.07, 0.20)
    C   : ClassVar[Preset]  = Preset(1e5, 1e4, 0.30, 0.80, 0.70, 0.50, 3.0, 1e3, 0.10, 0.12)
    D   : ClassVar[Preset]  = Preset(1e4, 1e3, 0.45, 0.65, 0.55, 0.35, 2.5, 1e2, 0.15, 0.07)
    E   : ClassVar[Preset]  = Preset(1e3, 1e2, 0.60, 0.50, 0.40, 0.25, 2.0, 1e1, 0.25, 0.04)

    levels  : ClassVar[tuple[str, ...]] = ('A', 'B', 'C', 'D', 'E')

    @staticmethod
    def of(level: str) -> Preset:
        return getattr(Significance, level if level in Significance.levels else 'C')


@dataclass(frozen = True, eq = True)
class Policy:
    window_days : int   = 30

    volume_target       : float = 1e5
    volume_steepness    : float = 1.5

    w_validity  : float = 0.40
    w_presence  : float = 0.20
    w_retention : float = 0.40

    trend_gain      : float = 3.0
    rate_target     : float = 1e3
    stability_tol   : float = 0.10

    w_trend     : float = 0.40
    w_stability : float = 0.40
    w_rate      : float = 0.20

    w_volume    : float = 0.40
    w_quality   : float = 0.40
    w_accel     : float = 0.20

    min_eff_traces  : float = 1e4
    max_loss        : float = 0.30
    min_quality     : float = 0.80

    ready_at    : float = 0.70
    pilot_at    : float = 0.50

    @staticmethod
    def of(significance: str, **overrides: float) -> 'Policy':
        p       = Significance.of(significance)
        days    = int(overrides.get('window_days', 30))
        rest    = dict(filter(lambda kv: kv[0] != 'window_days', overrides.items()))
        base    = dict(
            volume_target   = p.volume_target,
            min_eff_traces  = p.min_eff_traces,
            max_loss        = p.max_loss,
            min_quality     = p.min_quality,
            ready_at        = p.ready_at,
            pilot_at        = p.pilot_at,
            trend_gain      = p.trend_gain,
            rate_target     = p.rate_target,
            stability_tol   = p.stability_tol)

        return Policy(window_days = days, **{**base, **rest})

    @staticmethod
    def logistic(x: pl.Expr) -> pl.Expr:
        return 1 / (1 + (-x).exp())

    def volume(self, eff_traces: pl.Expr) -> pl.Expr:
        return self.logistic(self.volume_steepness * ((eff_traces + 1).log10() - log10(self.volume_target)))

    def quality_sum(self, validity: pl.Expr, presence: pl.Expr, retention: pl.Expr) -> pl.Expr:
        return self.w_validity * validity + self.w_presence * presence + self.w_retention * retention
    
    def quality_mul(self, validity: pl.Expr, presence: pl.Expr, retention: pl.Expr) -> pl.Expr:
        return validity ** self.w_validity * presence ** self.w_presence * retention ** self.w_retention

    def acceleration(self, trend: pl.Expr, stability: pl.Expr, rate: pl.Expr) -> pl.Expr:
        return (self.w_trend     * self.logistic(self.trend_gain * trend)
              + self.w_stability * stability
              + self.w_rate      * (1 - (-rate / self.rate_target).exp()))

    def readiness(self, volume: pl.Expr, quality: pl.Expr, accel: pl.Expr) -> pl.Expr:
        return self.w_volume * volume + self.w_quality * quality + self.w_accel * accel

    def verdict(self, eff_traces: pl.Expr, loss: pl.Expr, quality: pl.Expr, readiness: pl.Expr) -> pl.Expr:
        gated = (eff_traces >= self.min_eff_traces) & (loss <= self.max_loss) & (quality >= self.min_quality)

        return (pl.when(~gated).then(pl.lit('not_ready'))
                  .when(readiness >= self.ready_at).then(pl.lit('ready'))
                  .when(readiness >= self.pilot_at).then(pl.lit('pilot'))
                  .otherwise(pl.lit('not_ready')))

    def reason(self, eff_traces: pl.Expr, loss: pl.Expr, quality: pl.Expr, readiness: pl.Expr, worst: pl.Expr, trend: pl.Expr) -> pl.Expr:
        return (pl.when(eff_traces < self.min_eff_traces).then(pl.lit('недостаточно трейсов'))
                  .when(loss > self.max_loss).then(pl.concat_str((pl.lit('высокая отбраковка, виновник: '), worst)))
                  .when(quality < self.min_quality).then(pl.concat_str((pl.lit('качество ниже порога, слабое звено: '), worst)))
                  .when(trend < 0).then(pl.lit('приток данных снижается'))
                  .when(readiness >= self.ready_at).then(pl.lit('готов к мониторингу'))
                  .when(readiness >= self.pilot_at).then(pl.lit('кандидат в пилот'))
                  .otherwise(pl.lit('накапливаем данные')))


@dataclass(frozen = True, eq = False)
class Readiness:
    quality : Quality
    policy  : Policy    = field(default_factory = Policy)

    def windowed(self) -> pl.LazyFrame:
        span_ns = self.policy.window_days * Ns.per_day

        return self.quality.tagged().filter(
            pl.col('start_time_ns') >= pl.col('start_time_ns').max() - span_ns)

    def components(self, win: pl.LazyFrame, *dims: Dim) -> pl.LazyFrame:
        attrs   = self.quality.spec.attrs
        mand    = FromSpec.mandatory(self.quality.spec)
        viol    = Across.violation_rate(attrs)
        gaps    = Across.null_rate(mand)
        base    = (
            pl.col('trace_id').n_unique().alias('traces'),
            pl.col('trace_id').filter(~pl.col('rejected')).n_unique().alias('eff_traces'),
            pl.len().alias('spans'),
            (1 - viol.mean()).alias('validity'),
            (1 - gaps.mean()).alias('presence'),
            (1 - pl.col('rejected').mean()).alias('retention'))
        blame   = map(lambda m: FromAttr.violation(m).mean().alias(f'_viol__{m.name}'), mand)
        keys    = Group.keyed(dims)
        body    = chain(base, blame)
        frame   = win.select(body) if not keys else win.group_by(keys).agg(body)

        return self.scored(frame)

    def scored(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        policy      = self.policy
        mand        = FromSpec.mandatory(self.quality.spec)
        worst_rate  = pl.max_horizontal(map(lambda m: pl.col(f'_viol__{m.name}'), mand))
        worst_name  = reduce(
            lambda acc, m: pl.when(pl.col(f'_viol__{m.name}') >= worst_rate).then(pl.lit(m.name)).otherwise(acc),
            mand, pl.lit('—'))

        return (frame
            .with_columns(
                (1 - pl.col('eff_traces') / pl.col('traces')).alias('loss_rate'),
                worst_rate.alias('worst_rule_rate'),
                pl.when(worst_rate <= 0).then(pl.lit('—')).otherwise(worst_name).alias('worst_rule_attr'))
            .with_columns(
                policy.volume(pl.col('eff_traces')).alias('volume_score'),
                policy.quality_sum(pl.col('validity'), pl.col('presence'), pl.col('retention')).alias('quality_score'))
            .select(pl.exclude(r'^_viol__.*$')))

    def acceleration(self, win: pl.LazyFrame, *dims: Dim, grain: str = '1d') -> pl.LazyFrame:
        policy  = self.policy
        attrs   = self.quality.spec.attrs
        viol    = Across.violation_rate(attrs)
        bkeys   = Group.keyed(dims) + (By.time(grain).keyed(),)
        bucket  = win.group_by(bkeys).agg(
            pl.col('trace_id').n_unique().alias('b_traces'),
            (1 - viol.mean()).alias('b_validity'))
        names   = Group.named(dims)
        order   = pl.col('ts').dt.epoch('s')
        body    = (
            pl.col('b_traces').mean().alias('rate'),
            pl.corr(order, pl.col('b_traces')).alias('trend'),
            pl.col('b_validity').std().alias('_qual_vol'))
        slices  = bucket.group_by(names).agg(body) if names else bucket.select(body)

        return (slices
            .with_columns(
                (1 - pl.col('_qual_vol') / policy.stability_tol).clip(0, 1).fill_null(1.0).alias('stability'),
                pl.col('trend').fill_null(0.0).alias('trend'))
            .with_columns(
                policy.acceleration(pl.col('trend'), pl.col('stability'), pl.col('rate')).alias('accel_score'))
            .select(pl.exclude(r'^_qual_vol$')))

    def assess(self, *dims: Dim, grain: str = '1d') -> pl.LazyFrame:
        policy  = self.policy
        names   = Group.named(dims)
        win     = self.windowed()
        comp    = self.components(win, *dims)
        accel   = self.acceleration(win, *dims, grain = grain)
        joined  = comp.join(accel, on = names, how = 'left') if names else comp.join(accel, how = 'cross')
        report  = (
            'traces', 'eff_traces', 'loss_rate', 'worst_rule_attr',
            'volume_score', 'quality_score', 'accel_score', 'readiness', 'verdict', 'reason')

        return (joined
            .with_columns(
                policy.readiness(pl.col('volume_score'), pl.col('quality_score'), pl.col('accel_score')).alias('readiness'))
            .with_columns(
                policy.verdict(pl.col('eff_traces'), pl.col('loss_rate'), pl.col('quality_score'), pl.col('readiness')).alias('verdict'),
                policy.reason(pl.col('eff_traces'), pl.col('loss_rate'), pl.col('quality_score'), pl.col('readiness'), pl.col('worst_rule_attr'), pl.col('trend')).alias('reason'))
            .select(names + report)
            .sort('readiness', descending = True))

    def blame(self, *dims: Dim) -> pl.LazyFrame:
        names   = Group.named(dims)
        report  = self.quality.correctness(*dims)

        return (report
            .unpivot(index = names + ('n',), variable_name = 'attr', value_name = 'violation_rate')
            .filter(pl.col('violation_rate') > 0)
            .sort('violation_rate', descending = True))


@dataclass(frozen = True, eq = False)
class Args:
    ints    : ClassVar[tuple[str, ...]] = ('window_days',)
    gates   : ClassVar[tuple[str, ...]] = ('max_nulls', 'max_rules', 'max_dups', 'max_breaks')
    floats  : ClassVar[tuple[str, ...]] = (
        'volume_target', 'volume_steepness', 'rate_target', 'min_eff_traces', 'max_loss', 'min_quality',
        'ready_at', 'pilot_at', 'trend_gain', 'stability_tol',
        'w_validity', 'w_presence', 'w_retention', 'w_trend', 'w_stability', 'w_rate',
        'w_volume', 'w_quality', 'w_accel')

    @staticmethod
    def given(params: dict, key: str) -> bool:
        return key in params and params[key] not in (None, '')

    @staticmethod
    def overrides(params: dict) -> dict:
        pick = lambda keys, fn: map(
            lambda k: (k, fn(params[k])),
            filter(partial(Args.given, params), keys))

        return dict(chain(pick(Args.ints, int), pick(Args.floats, float)))

    @staticmethod
    def thresholds(params: dict) -> dict:
        return dict(map(
            lambda k: (k, int(params[k])),
            filter(partial(Args.given, params), Args.gates)))


@dataclass(frozen = True, eq = False)
class Report:
    title   : ClassVar[str] = 'Отчёт валидации данных'

    @staticmethod
    def note(text: str) -> str:
        return f'<p><small>{text}</small></p>'

    @staticmethod
    def kv(mapping: dict) -> pl.DataFrame:
        return pl.DataFrame({
            'показатель': map(str, mapping.keys()),
            'значение'  : map(str, mapping.values())})

    @staticmethod
    def attrs_frame(rows: tuple) -> pl.DataFrame:
        column = lambda i: map(lambda r: r[i], rows)

        return pl.DataFrame({
            'атрибут'       : column(0),
            'тип на входе'  : column(2),
            'ожидается'     : column(3),
            'статус'        : column(4)})

    @staticmethod
    def verdict_word(critical_ok: bool, ceiling: float) -> str:
        return ('критичные атрибуты нарушены — данные невалидны' if not critical_ok
                else f'критичные атрибуты в норме; потолок качества по структуре ≈ {round(ceiling * 100)}%')

    @staticmethod
    def schema_blocks(level: str, diff: dict) -> tuple[str, ...]:
        return (
            viz.block_table('Фаза 1 · техническая корректность (схема ТЗ 1.9.2)', Report.kv(diff['counts'])),
            Report.note(f'* значимость агентов: {level}. Сверяются присутствие, тип и лишние колонки. '
                        f'{Report.verdict_word(diff['critical_ok'], diff['ceiling'])}.'),
            viz.block_table('Критичные («красные») атрибуты', Report.attrs_frame(diff['crit'])),
            Report.note('* статус: «ок» — тип совпал; «приведён» — тип отличается, но безопасно сконвертирован; '
                        '«отсутствует»/«несовместим» по критичному атрибуту → данные невалидны (обучение невозможно).'),
            viz.block_table('Некритичные атрибуты с дефектом структуры', Report.attrs_frame(diff['soft_bad'])),
            Report.note('* поля важны, но не критичны: можно жить после фиксов данных и пробовать обучать модели, '
                        f'но качество будет ниже — потолок при текущей структуре ≈ {round(diff['ceiling'] * 100)}% '
                        '(каждый дефект некритичного поля снижает потолок тем сильнее, чем выше значимость агентов).'),
            viz.block_table('Лишние колонки (игнорируются)', pl.DataFrame({'лишняя колонка': diff['extra']})),
            Report.note('* лишние колонки не входят в спецификацию и в анализе значений не участвуют.'))

    @staticmethod
    def quality_blocks(frames: tuple) -> tuple[str, ...]:
        verdict, consistency, completeness, correctness, rejection, assess, blame, distribution = frames[:8]

        return (
            viz.block_table('Фаза 2 · вердикт целостности (по значениям)', verdict),
            Report.note('* null_violations — null в обязательных полях; missing — пропуск (null или страж empty); '
                        'trash_cells — ячейки со стражем trash (мусор в сыром источнике, не приведённый к типу); '
                        'rule_violations — нарушения правил валидности (включая trash, не проходящий правила); '
                        'duplicate_keys — дубли пары (trace_id, span_id); trace_breaks — расхождение инвариантов внутри трассы; '
                        'valid=true только при нулях по всем счётчикам (с учётом порогов max_*).'),
            viz.block_table('Несогласованность инвариантов трасс', consistency),
            Report.note('* сколько трасс имеют более одного значения инварианта (agent_id/session_id/service_*/версия): '
                        'внутри одной трассы они обязаны совпадать.'),
            viz.block_table('Доля пропусков по агентам', completeness),
            Report.note('* доля пропусков = (null или sentinel) / N по каждому атрибуту в разрезе агента; меньше — лучше.'),
            viz.block_table('Доля нарушений правил по агентам (error rate)', correctness),
            Report.note('* доля нарушений = записи, не прошедшие правило валидности атрибута, / N; меньше — лучше.'),
            viz.block_table('Отбраковка по агентам', rejection),
            Report.note('* трасса отбраковывается, если хотя бы один её спан нарушает критичный атрибут; '
                        'dropped_rate — доля таких трасс.'),
            viz.block_table('Распределения числовых сигналов и длительностей (квантили)', distribution),
            Report.note('* q25..q95 — квантили по каждому числовому атрибуту и длительности спана (duration_ns); '
                        'форма распределения и хвосты выявляют выбросы и деградацию.'),
            viz.block_table('Готовность агентов к мониторингу и пилоту', assess),
            Report.note('* readiness — взвешенная свёртка объёма, качества и динамики притока (пороги зависят от значимости); '
                        'verdict — ready / pilot / not_ready; reason — лимитирующий фактор.'),
            viz.block_table('Виновные атрибуты (по доле нарушений)', blame),
            Report.note('* атрибуты с ненулевой долей нарушений в разрезе агента, по убыванию — где чинить в первую очередь.'))

    @staticmethod
    def policy_block(level: str, policy: Policy) -> tuple[str, ...]:
        coeffs = {
            'уровень значимости агентов'            : level,
            'эталонный объём N* (volume_target)'    : f'{policy.volume_target:.0f}',
            'крутизна объёма k (volume_steepness)'  : policy.volume_steepness,
            'мин. эфф. трасс N_min (min_eff_traces)': f'{policy.min_eff_traces:.0f}',
            'макс. потери L_max (max_loss)'         : policy.max_loss,
            'мин. качество Q_min (min_quality)'     : policy.min_quality,
            'порог ready (ready_at)'                : policy.ready_at,
            'порог pilot (pilot_at)'                : policy.pilot_at,
            'усиление тренда g (trend_gain)'        : policy.trend_gain,
            'эталонный темп r* (rate_target)'       : f'{policy.rate_target:.0f}',
            'разброс θ (stability_tol)'             : policy.stability_tol,
            'окно анализа, дней (window_days)'      : policy.window_days,
            'веса качества w_v / w_p / w_r'         : f'{policy.w_validity} / {policy.w_presence} / {policy.w_retention}',
            'веса динамики w_τ / w_s / w_ρ'         : f'{policy.w_trend} / {policy.w_stability} / {policy.w_rate}',
            'веса готовности w_V / w_Q / w_A'       : f'{policy.w_volume} / {policy.w_quality} / {policy.w_accel}'}

        return (
            viz.block_table('Политика и коэффициенты (из уровня значимости, §11)', Report.kv(coeffs)),
            Report.note('* все пороги и веса берутся из пресета значимости и переопределяемы через Policy.of(level, **overrides).'))

    @staticmethod
    def formulas_block(policy: Policy) -> tuple[str, ...]:
        formulas = {
            'Q — балл качества (§06)'       : f'{policy.w_validity}·validity + {policy.w_presence}·presence + {policy.w_retention}·retention',
            'V — балл объёма (§07)'         : f'σ({policy.volume_steepness}·(log10(N_eff+1) − log10({policy.volume_target:.0f})))',
            'A — динамика (§08)'            : f'{policy.w_trend}·σ({policy.trend_gain}·τ) + {policy.w_stability}·stab + {policy.w_rate}·(1−e^(−r/{policy.rate_target:.0f}))',
            'R — готовность (§09)'          : f'{policy.w_volume}·V + {policy.w_quality}·Q + {policy.w_accel}·A',
            'gate — обязательные критерии'  : f'[N_eff ≥ {policy.min_eff_traces:.0f}] ∧ [loss ≤ {policy.max_loss}] ∧ [Q ≥ {policy.min_quality}]',
            'вердикт'                       : f'gate ∧ R ≥ {policy.ready_at} → ready · gate ∧ R ≥ {policy.pilot_at} → pilot · иначе not_ready'}

        return (
            viz.block_table('Применённые формулы и пороги (§06–§09)', Report.kv(formulas)),
            Report.note('* σ(x)=1/(1+e^−x); stab=clip₍₀,₁₎(1−σ_Q/θ); τ=corr(индекс корзины, приток); числа подставлены из активной политики.'))

    @staticmethod
    def dynamics_chart(flow: pl.DataFrame, trend: float) -> al.VConcatChart:
        arrow   = '↑ рост' if trend > 0.05 else '↓ спад' if trend < -0.05 else '→ ровно'
        base    = al.Chart(flow).encode(x = al.X('ts:T', title = 'время (корзины grain)'))
        line    = base.mark_line(strokeWidth = 2.5, point = True, interpolate = 'monotone', color = '#2563eb').encode(
            y = al.Y('traces:Q', title = 'трасс в корзине'))
        mean    = base.mark_rule(strokeDash = [6, 4], color = '#64748b').encode(y = 'mean(traces):Q')
        qual    = base.mark_line(strokeWidth = 2.5, point = True, interpolate = 'monotone', color = '#16a34a').encode(
            y = al.Y('validity:Q', title = 'валидность в корзине', scale = al.Scale(domain = [0, 1])))

        return ((line + mean) & qual).properties(
            title = f'Динамика притока и валидности по корзинам · тренд притока {arrow} (corr = {trend:+.2f})')

    @staticmethod
    def dynamics_safe(flow: pl.DataFrame) -> str:
        try:
            trend   = flow.with_row_index('i').select(pl.corr('i', 'traces')).item() or 0.0
            image   = viz.save_chart(Report.dynamics_chart(flow, trend), PurePath(mkdtemp()) / 'flow_dynamics')

            return (viz.block_image('Фаза 2 · динамика потока данных', image)
                  + Report.note('* верх — приток трасс по временным корзинам (пунктир — среднее, тренд = corr с индексом корзины); '
                                'низ — валидность по корзинам (разброс = нестабильность качества во времени, §08).'))
        except Exception:
            return Report.note('* график динамики не построен (недостаточно временных корзин или ошибка рендера).')

    @staticmethod
    def stop_note(diff: dict) -> str:
        crit    = diff['crit']
        present = tuple(filter(lambda r: r[4] not in SchemaState.broken, crit))
        message = ('не найдено ни одного обязательного (критичного) атрибута — анализ значений остановлен'
                   if not present
                   else f'нарушена структурная корректность по критичным атрибутам ({len(crit) - len(present)} из {len(crit)}) — '
                        'анализ значений остановлен (жёсткий контур, §05); выше — только техническая корректность')

        return Report.note(f'⛔ {message}.')

    @staticmethod
    def schema_only(level: str, diff: dict) -> str:
        blocks = Report.schema_blocks(level, diff) + (Report.stop_note(diff),)

        return viz.Html.doc(Report.title, f'значимость агентов: {level} · структурная проверка не пройдена', str().join(blocks))

    @staticmethod
    def doc(level: str, policy: Policy, diff: dict, frames: tuple) -> str:
        subtitle    = f'значимость агентов: {level} · схема → политика → значения → динамика → готовность'
        blocks      = (Report.schema_blocks(level, diff)
                  + Report.policy_block(level, policy)
                  + Report.formulas_block(policy)
                  + Report.quality_blocks(frames)
                  + (Report.dynamics_safe(frames[8]),))

        return viz.Html.doc(Report.title, subtitle, str().join(blocks))

    @staticmethod
    def failure(message: str) -> dict:
        return {
            'report_html' : viz.Html.doc(Report.title, 'ошибка валидации', Report.note(f'Валидация не выполнена: {message}')),
            'status'      : {'value': 'red', 'title': 'валидация не выполнена'},
            'verdict'     : {'error': message}}


def main(**params: str) -> dict:
    try:
        dont_colorize_log   = str(params.get('dont_colorize_log', 'true')).strip().lower() in ('true', '1', 'yes', 'on')
        cs                  = ColorSchemeDataScience() if dont_colorize_log else ColorSchemeDataScienceCold()
        cs.print_section('Валидация данных')
        
        source  = str(params.get('path_spans') or '')
        if not source: return Report.failure('источник не задан: подключите порт path_spans')

        level   = str(params.get('significance', 'C')).strip().upper()
        policy  = Policy.of(level, **Args.overrides(params))
        gates   = Args.thresholds(params)
        device  = Device.of(params.get('device', ''))
        recast  = str(params.get('recast', 'true')).strip().lower() in ('true', '1', 'yes', 'on')

        lazy    = pl.scan_parquet(PurePath(source).as_posix(), rechunk = True)
        schema  = lazy.collect_schema()
        diff    = SchemaCheck.diff(schema, SpansData, Significance.of(level))

        if not diff['critical_ok']:
            return {
                'report_html' : Report.schema_only(level, diff),
                'status'      : {'value': 'red', 'title': 'структурная проверка не пройдена'},
                'verdict'     : {'schema': diff['counts'], 'quality': None}}

        conformed   = (Recast.frame(lazy, schema, SpansData) if recast
                     else lazy.select(map(lambda a: pl.col(a.name).cast(a.type_polars, strict = False)
                                                    if a.name in schema.names()
                                                    else pl.lit(a.empty, dtype = a.type_polars).alias(a.name), SpansData.attrs)))
        quality     = Quality(conformed)
        ready       = Readiness(quality, policy)

        lazies = (
            quality.verdict(**gates),
            quality.trace_consistency(),
            quality.completeness(By.agent),
            quality.correctness(By.agent),
            quality.rejection(By.agent),
            ready.assess(By.agent),
            ready.blame(By.agent),
            quality.distribution(),
            quality.flow())
        cs.print_subsection('расчёт метрик качества')
        frames, m_elapsed, m_cpu, m_gpu = perf.measure_during_call(pl.collect_all, (lazies,), {'engine': device.engine})
        cs.print_performance_metrics('валидация (9 метрик)', m_elapsed, m_cpu, m_gpu)
        frames = tuple(frames)

        valid   = bool(frames[0].get_column('valid').item())
        light   = 'green' if (valid and not diff['degraded']) else 'amber' if valid else 'red'
        titled  = ('данные валидны'                   if (valid and not diff['degraded'])
                   else 'данные валидны (с оговорками)' if valid
                   else 'данные невалидны')

        return {
            'report_html' : Report.doc(level, policy, diff, frames),
            'status'      : {'value': light, 'title': titled},
            'verdict'     : {'schema': diff['counts'], 'quality': frames[0].to_dicts()}}

    except Exception as error:
        return Report.failure(f'{type(error).__name__}: {error}')


if __name__ == '__main__':
    device              = Device.of()
    dont_colorize_log   = True
    cs                  = ColorSchemeDataScience() if dont_colorize_log else ColorSchemeDataScienceCold()

    cs.print_section('Валидация данных')

    lazy = pl.scan_parquet('/mnt/data/traces/raw/synthetic__trip_planner.parquet', rechunk = True)

    lazy = recast(lazy, SpansData)

    schema  = lazy.collect_schema()
    diff    = SchemaCheck.diff(schema, SpansData, Significance.of('C'))

    quality = Quality(Recast.frame(lazy, schema, SpansData))
    ready   = Readiness(quality, Policy.of('C'))

    lazies = (
        quality.verdict(),
        quality.trace_consistency(),
        quality.completeness(By.agent),
        quality.correctness(By.agent),
        quality.rejection(By.agent),
        ready.assess(By.agent),
        ready.blame(By.agent),
        quality.distribution(),
        quality.flow())
    cs.print_subsection('расчёт метрик качества')
    frames, m_elapsed, m_cpu, m_gpu = perf.measure_during_call(pl.collect_all, (lazies,), {'engine': device.engine})
    cs.print_performance_metrics('валидация (9 метрик)', m_elapsed, m_cpu, m_gpu)
    frames = tuple(frames)

    cs.print_section('Отчёт')

    viz.write_report(
        PurePath('/mnt/data/reports/validation_report__trip_planner.html'),
        Report.title,
        'значимость агентов: C · схема → политика → значения → динамика → готовность',
        Report.schema_blocks('C', diff)
            + Report.policy_block('C', Policy.of('C'))
            + Report.formulas_block(Policy.of('C'))
            + Report.quality_blocks(frames)
            + (Report.dynamics_safe(frames[8]),))