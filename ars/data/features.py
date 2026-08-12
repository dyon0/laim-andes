from    typing                      import ClassVar, Callable
from    dataclasses                 import dataclass, field
from    functools                   import partial, reduce
from    itertools                   import product, chain, starmap, groupby
from    inspect                     import signature

import  polars                      as pl

from    ars.configuration.c1__data  import FeatureParams
from    ars.specification.spec      import SpansData, DataObject


@dataclass(frozen = True)
class RawSchema:
    schema                  : dict[str, type[pl.DataType] | pl.Enum]    = field(
        default_factory = lambda: dict(SpansData.schema)) #type: ignore
    # Type "dict[str, DataType]" is not assignable to declared type "dict[str, type[DataType] | Enum]"
    #  "dict[str, DataType]" is not assignable to "dict[str, type[DataType] | Enum]"
    #    Type parameter "_VT@dict" is invariant, but "DataType" is not the same as "type[DataType] | Enum"
    #    Consider switching from "dict" to "Mapping" which is covariant in the value type
    drop_mask               : str                                       = r'^.*-UNAVAIL$'


@dataclass(frozen = True)
class FeaturePatterns:
    label_col           : ClassVar[str]                             = DataObject.label
    sublabel_col        : ClassVar[str]                             = DataObject.sublabel
    objects_order       : ClassVar[tuple[tuple[str, bool], ...]]    = DataObject.objects_order

    error               : ClassVar[str]                             = FeatureParams.error
    sentence            : ClassVar[str]                             = FeatureParams.sentence
    punctuation         : ClassVar[str]                             = FeatureParams.punctuation
    uppercase           : ClassVar[str]                             = FeatureParams.uppercase
    special_char        : ClassVar[str]                             = FeatureParams.special_char
    digit               : ClassVar[str]                             = FeatureParams.digit

    quantiles           : ClassVar[tuple[float, ...]]               = FeatureParams.quantiles
    rolling_windows     : ClassVar[tuple[int, ...]]                 = FeatureParams.rolling_windows
    eps                 : ClassVar[float]                           = FeatureParams.eps


@dataclass(frozen = True, slots = True)
class FeatureDefinition:
    name                : str
    expr_builder        : Callable[..., pl.Expr]
    object_aggregation  : tuple[str, ...]                           = DataObject.object_aggregation
    log_transform       : bool                                      = False
    include_in_sequence : bool                                      = True
    aggs_static         : frozenset[str]                            = frozenset()
    aggs_dynamic        : frozenset[str]                            = frozenset()
    calculation_stage   : int                                       = 1

    def build_expr(self, **kwargs) -> pl.Expr:
        params  = signature(self.expr_builder).parameters
        passed  = dict(filter(lambda kv: kv[0] in params, kwargs.items()))
        base    = self.expr_builder(**passed)
        logged  = pl.when(base < 0).then(base.abs().log1p().mul(-1)).otherwise(base.abs().log1p())

        return (logged if self.log_transform else base).alias(self.name)

    def _static(self, agg: str, col: pl.Expr, cfg: FeaturePatterns) -> tuple[pl.Expr, ...]:
        if agg != 'quantiles' and not hasattr(col, agg):
            raise NotImplementedError(f'неизвестная статическая агрегация {agg!r}')

        quant = lambda q: col.quantile(q, interpolation = 'linear').over(self.object_aggregation).alias(f'{self.name}_q{int(q * 100)}')

        return (tuple(map(quant, cfg.quantiles)) if agg == 'quantiles'
            else (getattr(col, agg)().over(self.object_aggregation).alias(f'{self.name}_{agg}'),))

    def _dynamic(self, agg: str, window: int, col: pl.Expr, cfg: FeaturePatterns) -> tuple[pl.Expr, ...]:
        rolling = f'rolling_{agg}'
        if agg != 'quantiles' and not hasattr(col, rolling):
            raise NotImplementedError(f'неизвестная динамическая агрегация {agg!r}')

        quant = lambda q: (col
            .rolling_quantile(quantile = q, window_size = window, min_samples = 1, interpolation = 'linear')
            .over(self.object_aggregation)
            .alias(f'{self.name}_rolling_q{int(q * 100)}_w{window}'))

        return (tuple(map(quant, cfg.quantiles)) if agg == 'quantiles'
            else (getattr(col, rolling)(window_size = window, min_samples = 1).over(self.object_aggregation).alias(f'{self.name}_rolling_{agg}_w{window}'),))

    def build_aggregated_exprs(self, feature_cfg: FeaturePatterns = FeaturePatterns()) -> tuple[pl.Expr, ...]:
        col         = pl.col(self.name)
        statics     = chain.from_iterable(map(lambda a: self._static(a, col, feature_cfg), self.aggs_static))
        windows     = product(self.aggs_dynamic, feature_cfg.rolling_windows)
        dynamics    = chain.from_iterable(starmap(lambda a, w: self._dynamic(a, w, col, feature_cfg), windows))

        return tuple(chain(statics, dynamics))


@dataclass(frozen = True)
class FeaturesSpan:
    full_agg    : ClassVar[frozenset[str]]  = frozenset({'max', 'min', 'mean', 'std', 'sum', 'quantiles'})
    spread_agg  : ClassVar[frozenset[str]]  = frozenset({'max', 'min', 'mean', 'std', 'quantiles'})

    total_duration  : FeatureDefinition = FeatureDefinition('total_duration',
        expr_builder    = lambda: (pl.col('end_time_ns') - pl.col('start_time_ns')).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    duration        : FeatureDefinition = FeatureDefinition('duration',
        expr_builder    = lambda: (pl.col('end_time_ns') - pl.col('start_time_ns')).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    duration_diff   : FeatureDefinition = FeatureDefinition('duration_diff',
        # F-50: was (end-start)-(end-start) == 0; now the step-to-step duration
        # change within the (trace, agent) sequence
        expr_builder    = lambda object_aggregation: (pl.col('end_time_ns') - pl.col('start_time_ns'))
                            .diff(n = 1).over(object_aggregation, order_by = 'start_time_ns')
                            .cast(pl.Float64).fill_null(0.0),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    delta_time      : FeatureDefinition = FeatureDefinition('delta_time',
        expr_builder    = lambda object_aggregation: pl.col('start_time_ns').diff(n = 1).over(object_aggregation, order_by = 'start_time_ns').cast(pl.Float64).fill_null(0.0),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    exec_gap        : FeatureDefinition = FeatureDefinition('exec_gap',
        expr_builder    = lambda object_aggregation: (pl.col('start_time_ns') - pl.col('end_time_ns').shift(1)).over(object_aggregation, order_by = 'start_time_ns').cast(pl.Float64).fill_null(0.0),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)

    llm_tokens      : FeatureDefinition = FeatureDefinition('llm_tokens',
        # F-07: -1 is the "unknown" sentinel, not a token count — the data spec
        # (section 7) requires aggregations to ignore sentinel values, so an LLM
        # span with unknown usage becomes null (skipped by polars aggregates)
        expr_builder    = lambda: pl.when((pl.col('aef_kind') == 'llm') & (pl.col('llm_total_tokens') >= 0))
                                    .then(pl.col('llm_total_tokens'))
                                    .when(pl.col('aef_kind') != 'llm')
                                    .then(0)
                                    .otherwise(None).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    llm_duration    : FeatureDefinition = FeatureDefinition('llm_duration',
        expr_builder    = lambda: pl.when(pl.col('aef_kind') == 'llm').then(pl.col('end_time_ns') - pl.col('start_time_ns')).otherwise(0).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    token_ratio     : FeatureDefinition = FeatureDefinition('token_ratio',
        expr_builder = lambda feature_cfg: (
            pl.when((pl.col('aef_kind') == 'llm') & (pl.col('output_text').str.len_chars() > 0))
                .then(pl.col('input_text').str.len_chars().truediv(pl.col('output_text').str.len_chars()))
                .otherwise(None)),
        aggs_static = frozenset({'mean', 'std'}))

    tool_duration       : FeatureDefinition = FeatureDefinition('tool_duration',
        expr_builder    = lambda: pl.when(pl.col('aef_kind') == 'tool').then(pl.col('end_time_ns') - pl.col('start_time_ns')).otherwise(0).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    tool_success        : FeatureDefinition = FeatureDefinition('tool_success',
        expr_builder    = lambda: pl.when(pl.col('aef_kind') == 'tool').then((pl.col('status_code') == 'STATUS_CODE_OK').cast(pl.Int32)).otherwise(0),
        aggs_static     = frozenset({'sum'}),
        aggs_dynamic    = frozenset({'sum'}))
    unique_tools_local  : FeatureDefinition = FeatureDefinition('unique_tools_local',
        expr_builder    = lambda object_aggregation: pl.when(pl.col('aef_kind') == 'tool').then(pl.col('span_name')).otherwise(None).cumulative_eval(pl.element().n_unique()).over(object_aggregation, order_by = 'start_time_ns'),
        aggs_dynamic    = spread_agg)

    sem_text        : FeatureDefinition = FeatureDefinition('sem_text',
        expr_builder = lambda: pl.col('output_text').cast(pl.Utf8).fill_null(''))
    agent_prompt    : FeatureDefinition = FeatureDefinition('agent_prompt',
        expr_builder = lambda: pl.col('input_text').cast(pl.Utf8).fill_null(''))

    error_flag  : FeatureDefinition = FeatureDefinition('error_flag',
        expr_builder        = lambda feature_cfg: pl.col('sem_text').str.to_lowercase().str.contains(feature_cfg.error).cast(pl.Int32),
        aggs_static         = frozenset({'sum'}),
        aggs_dynamic        = frozenset({'sum'}),
        calculation_stage   = 2)

    char_count          : FeatureDefinition = FeatureDefinition('char_count',
        expr_builder        = lambda: pl.col('sem_text').str.len_chars().cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    word_count          : FeatureDefinition = FeatureDefinition('word_count',
        expr_builder        = lambda: pl.col('sem_text').str.split(' ').list.len().cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    sentence_count      : FeatureDefinition = FeatureDefinition('sentence_count',
        expr_builder        = lambda feature_cfg: pl.col('sem_text').str.count_matches(feature_cfg.sentence).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    punctuation_count   : FeatureDefinition = FeatureDefinition('punctuation_count',
        expr_builder        = lambda feature_cfg: pl.col('sem_text').str.count_matches(feature_cfg.punctuation).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    uppercase_count     : FeatureDefinition = FeatureDefinition('uppercase_count',
        expr_builder        = lambda feature_cfg: pl.col('sem_text').str.count_matches(feature_cfg.uppercase).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    digit_count         : FeatureDefinition = FeatureDefinition('digit_count',
        expr_builder        = lambda feature_cfg: pl.col('sem_text').str.count_matches(feature_cfg.digit).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    special_char_count  : FeatureDefinition = FeatureDefinition('special_char_count',
        expr_builder        = lambda feature_cfg: pl.col('sem_text').str.count_matches(feature_cfg.special_char).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)

    prompt_char_count           : FeatureDefinition = FeatureDefinition('prompt_char_count',
        expr_builder        = lambda: pl.col('agent_prompt').str.len_chars().cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    prompt_word_count           : FeatureDefinition = FeatureDefinition('prompt_word_count',
        expr_builder        = lambda: pl.col('agent_prompt').str.split(' ').list.len().cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    prompt_sentence_count       : FeatureDefinition = FeatureDefinition('prompt_sentence_count',
        expr_builder        = lambda feature_cfg: pl.col('agent_prompt').str.count_matches(feature_cfg.sentence).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    prompt_punctuation_count    : FeatureDefinition = FeatureDefinition('prompt_punctuation_count',
        expr_builder        = lambda feature_cfg: pl.col('agent_prompt').str.count_matches(feature_cfg.punctuation).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    prompt_uppercase_count      : FeatureDefinition = FeatureDefinition('prompt_uppercase_count',
        expr_builder        = lambda feature_cfg: pl.col('agent_prompt').str.count_matches(feature_cfg.uppercase).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    prompt_digit_count          : FeatureDefinition = FeatureDefinition('prompt_digit_count',
        expr_builder        = lambda feature_cfg: pl.col('agent_prompt').str.count_matches(feature_cfg.digit).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)
    prompt_special_char_count   : FeatureDefinition = FeatureDefinition('prompt_special_char_count',
        expr_builder        = lambda feature_cfg: pl.col('agent_prompt').str.count_matches(feature_cfg.special_char).cast(pl.Float64),
        log_transform       = True,
        aggs_static         = full_agg,
        aggs_dynamic        = full_agg,
        calculation_stage   = 2)

    exec_out_char_count         : FeatureDefinition = FeatureDefinition('exec_out_char_count',
        expr_builder    = lambda: pl.col('output_text').str.len_chars().cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    exec_out_word_count         : FeatureDefinition = FeatureDefinition('exec_out_word_count',
        expr_builder    = lambda: pl.col('output_text').str.split(' ').list.len().cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    exec_out_sentence_count     : FeatureDefinition = FeatureDefinition('exec_out_sentence_count',
        expr_builder    = lambda feature_cfg: pl.col('output_text').str.count_matches(feature_cfg.sentence).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    exec_out_punctuation_count  : FeatureDefinition = FeatureDefinition('exec_out_punctuation_count',
        expr_builder    = lambda feature_cfg: pl.col('output_text').str.count_matches(feature_cfg.punctuation).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    exec_out_uppercase_count    : FeatureDefinition = FeatureDefinition('exec_out_uppercase_count',
        expr_builder    = lambda feature_cfg: pl.col('output_text').str.count_matches(feature_cfg.uppercase).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    exec_out_digit_count        : FeatureDefinition = FeatureDefinition('exec_out_digit_count',
        expr_builder    = lambda feature_cfg: pl.col('output_text').str.count_matches(feature_cfg.digit).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)
    exec_out_special_char_count : FeatureDefinition = FeatureDefinition('exec_out_special_char_count',
        expr_builder    = lambda feature_cfg: pl.col('output_text').str.count_matches(feature_cfg.special_char).cast(pl.Float64),
        log_transform   = True,
        aggs_static     = full_agg,
        aggs_dynamic    = full_agg)

    is_llm      : FeatureDefinition = FeatureDefinition('is_llm',
        expr_builder    = lambda: (pl.col('aef_kind') == 'llm').cast(pl.Int32),
        aggs_static     = frozenset({'sum'}),
        aggs_dynamic    = frozenset({'sum'}))
    is_tool     : FeatureDefinition = FeatureDefinition('is_tool',
        expr_builder    = lambda: (pl.col('aef_kind') == 'tool').cast(pl.Int32),
        aggs_static     = frozenset({'sum'}),
        aggs_dynamic    = frozenset({'sum'}))
    is_chain    : FeatureDefinition = FeatureDefinition('is_chain',
        expr_builder    = lambda: (pl.col('aef_kind') == 'chain').cast(pl.Int32),
        aggs_static     = frozenset({'sum'}),
        aggs_dynamic    = frozenset({'sum'}))

    tool_compression    : FeatureDefinition = FeatureDefinition('tool_compression',
        # F-07: non-tool spans are "not applicable" (null, skipped by
        # aggregates), not the sentinel -1 pretending to be a measurement
        expr_builder    = lambda feature_cfg: pl.when(pl.col('aef_kind') == 'tool').then(pl.col('output_text').str.len_chars().truediv(pl.col('input_text').str.len_chars() + feature_cfg.eps)).otherwise(None),
        log_transform   = True,
        aggs_static     = spread_agg,
        aggs_dynamic    = spread_agg)

    avg_word_length_sem : FeatureDefinition = FeatureDefinition('avg_word_length_sem',
        expr_builder        = lambda: pl.col('sem_text').str.split(' ').list.eval(pl.element().str.len_chars()).list.mean(),
        log_transform       = True,
        aggs_static         = spread_agg,
        aggs_dynamic        = spread_agg,
        calculation_stage   = 2)

    final_output_length : FeatureDefinition = FeatureDefinition('final_output_length',
        expr_builder        = lambda: pl.col('output_text').last().str.len_chars().over('trace_id'),
        log_transform       = True,
        include_in_sequence = False)

    unique_agents   : FeatureDefinition = FeatureDefinition('unique_agents',
        expr_builder        = lambda: pl.col('agent_id').n_unique().over('trace_id'),
        include_in_sequence = False)

    actions_per_agent   : FeatureDefinition = FeatureDefinition('actions_per_agent',
        expr_builder        = lambda: pl.len().over('trace_id').truediv(pl.col('agent_id').n_unique().over('trace_id')),
        log_transform       = True,
        include_in_sequence = False)

    time_gap_variance   : FeatureDefinition = FeatureDefinition('time_gap_variance',
        expr_builder        = lambda: pl.col('delta_time').var().over('trace_id'),
        include_in_sequence = False,
        calculation_stage   = 2)

    iteration_duration_variance : FeatureDefinition = FeatureDefinition('iteration_duration_variance',
        expr_builder        = lambda: pl.col('duration').var().over('trace_id'),
        include_in_sequence = False,
        calculation_stage   = 2)

    unique_tools_global : FeatureDefinition = FeatureDefinition('unique_tools_global',
        expr_builder        = lambda: pl.col('span_name').filter(pl.col('aef_kind') == 'tool').n_unique().over('trace_id'),
        include_in_sequence = False)

    action_entropy  : FeatureDefinition = FeatureDefinition('action_entropy',
        expr_builder        = lambda: pl.col('aef_kind').value_counts(sort = True).struct.field('count').entropy(base = 2).over('trace_id'),
        include_in_sequence = False)

    is_repeat   : FeatureDefinition = FeatureDefinition('is_repeat',
        expr_builder = lambda: (
            (pl.col('aef_kind').cast(pl.Categorical).to_physical().shift(1).over('trace_id', order_by = 'start_time_ns')
                == pl.col('aef_kind').cast(pl.Categorical).to_physical())
            .cast(pl.Int32)
            .fill_null(0)),
        include_in_sequence = True)

    repetitive_actions  : FeatureDefinition = FeatureDefinition('repetitive_actions',
        expr_builder        = lambda: pl.col('is_repeat').sum().over('trace_id'),
        include_in_sequence = False,
        calculation_stage   = 2)

    processing_time_variance    : FeatureDefinition = FeatureDefinition('processing_time_variance',
        expr_builder        = lambda feature_cfg: pl.when(pl.col('aef_kind') == 'llm').then(pl.col('llm_duration').truediv(pl.col('llm_tokens') + feature_cfg.eps)).otherwise(None).std().over('trace_id'),
        include_in_sequence = False,
        calculation_stage   = 2)

    cv_duration : FeatureDefinition = FeatureDefinition('cv_duration',
        expr_builder        = lambda feature_cfg: pl.col('duration').std().over('trace_id').truediv(pl.col('duration').mean().over('trace_id') + feature_cfg.eps),
        log_transform       = True,
        include_in_sequence = False,
        calculation_stage   = 2)

    cv_delta    : FeatureDefinition = FeatureDefinition('cv_delta',
        expr_builder        = lambda feature_cfg: pl.col('delta_time').std().over('trace_id').truediv(pl.col('delta_time').mean().over('trace_id') + feature_cfg.eps),
        log_transform       = True,
        include_in_sequence = False,
        calculation_stage   = 2)

    def _apply_stage(self, feature_cfg: FeaturePatterns, df: pl.DataFrame, stage_feats: tuple[int, tuple[FeatureDefinition, ...]]) -> pl.DataFrame:
        feats   = stage_feats[1]
        local   = df.with_columns(map(
            lambda fd: fd.build_expr(object_aggregation = fd.object_aggregation, feature_cfg = feature_cfg),
            feats))

        return local.with_columns(chain.from_iterable(map(
            lambda fd: fd.build_aggregated_exprs(feature_cfg),
            feats)))

    def make_features(self, df: pl.DataFrame, feature_cfg: FeaturePatterns = FeaturePatterns()) -> pl.DataFrame:
        defs                = filter(lambda v: isinstance(v, FeatureDefinition), vars(self).values())
        stage               = lambda fd: fd.calculation_stage
        ordered             = sorted(defs, key = stage)
        stages              = starmap(lambda k, g: (k, tuple(g)), groupby(ordered, stage))
        keys, directions    = zip(*feature_cfg.objects_order)

        return reduce(partial(self._apply_stage, feature_cfg), stages, df.sort(keys, descending = directions))