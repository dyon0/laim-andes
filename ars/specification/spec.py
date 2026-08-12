from    typing                          import ClassVar
from    dataclasses                     import dataclass
from    itertools                       import chain

import  polars                          as pl

from    ars.specification.common_core   import Patterns, Categories, Sentinel
from    ars.specification.core          import SpanAttr, SpansDataSpec


SpansData = SpansDataSpec(attrs = (
    SpanAttr(
        name            = 'trace_id',
        group           = 'id',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = None,
        validation      = (
            pl.col('trace_id').str.contains(Patterns.base64)),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'span_id',
        group           = 'id',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = None,
        validation      = (
            pl.col('span_id').str.contains(Patterns.base64)),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'parent_span_id',
        group           = 'id',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.root,
        validation      = (
            pl.col('parent_span_id').str.contains(Patterns.base64) | (pl.col('parent_span_id') == 'root')),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'origin_span_id',
        group           = 'id',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.outside,
        validation      = (
            pl.col('origin_span_id').str.contains(Patterns.base64) | (pl.col('origin_span_id') == 'outside')),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'agent_id',
        group           = 'id',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = None,
        validation      = (
            pl.col('agent_id').str.contains(Patterns.codebase_elem)),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'session_id',
        group           = 'id',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = None,
        validation      = (
            pl.col('session_id').str.contains(Patterns.base64)),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'start_time_ns',
        group           = 'time',
        type_parquet    = 'INT64',
        type_polars     = pl.Int64,
        sentinel        = None,
        validation      = (
            pl.col('start_time_ns') >= 0),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'end_time_ns',
        group           = 'time',
        type_parquet    = 'INT64',
        type_polars     = pl.Int64,
        sentinel        = None,
        validation      = (
            pl.col('end_time_ns') >= pl.col('start_time_ns')),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'status_code',
        group           = 'status',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.Enum(categories = Categories.status_code + (Sentinel.token,)),
        sentinel        = Sentinel.unset,
        validation      = (
            pl.col('status_code').is_in(Categories.status_code)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'status_message',
        group           = 'status',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('status_code') == 'STATUS_CODE_ERROR')
                .then(pl.col('status_message').str.len_chars() > 0)
                .otherwise(pl.col('status_message') == '')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'aef_kind',
        group           = 'classification',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.Enum(categories = Categories.aef_kind + (Sentinel.token,)),
        sentinel        = None,
        validation      = (
            pl.col('aef_kind').is_in(Categories.aef_kind)),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'span_name',
        group           = 'classification',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = None,
        validation      = (
            pl.col('span_name').str.len_chars() > 0),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'input_text',
        group           = 'text_data',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            (pl.col('input_text') == '') | pl.col('input_text').str.contains(Patterns.json_like)),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'output_text',
        group           = 'text_data',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.lit(True)),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'llm_model',
        group           = 'LLM',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_model').str.len_chars() > 0)
                .otherwise(pl.col('llm_model') == '')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_prompt_tokens',
        group           = 'LLM',
        type_parquet    = 'INT64',
        type_polars     = pl.Int64,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_prompt_tokens') >= 0)
                .otherwise(pl.col('llm_prompt_tokens') == -1)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_completion_tokens',
        group           = 'LLM',
        type_parquet    = 'INT64',
        type_polars     = pl.Int64,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_completion_tokens') >= 0)
                .otherwise(pl.col('llm_completion_tokens') == -1)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_total_tokens',
        group           = 'LLM',
        type_parquet    = 'INT64',
        type_polars     = pl.Int64,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_total_tokens') >= 0)
                .otherwise(pl.col('llm_total_tokens') == -1)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_precached_prompt_tokens',
        group           = 'LLM',
        type_parquet    = 'INT64',
        type_polars     = pl.Int64,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_precached_prompt_tokens') >= 0)
                .otherwise(pl.col('llm_precached_prompt_tokens') == -1)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_temperature',
        group           = 'LLM',
        type_parquet    = 'FLOAT',
        type_polars     = pl.Float32,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_temperature') >= 0.0)
                .otherwise(pl.col('llm_temperature') == -1.0)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_top_p',
        group           = 'LLM',
        type_parquet    = 'FLOAT',
        type_polars     = pl.Float32,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_top_p').is_between(0.0, 1.0))
                .otherwise(pl.col('llm_top_p') == -1.0)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_max_tokens',
        group           = 'LLM',
        type_parquet    = 'INT64',
        type_polars     = pl.Int64,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_max_tokens') >= 0)
                .otherwise(pl.col('llm_max_tokens') == -1)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_repetition_penalty',
        group           = 'LLM',
        type_parquet    = 'FLOAT',
        type_polars     = pl.Float32,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_repetition_penalty') >= 1.0)
                .otherwise(pl.col('llm_repetition_penalty') == -1.0)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_profanity_check',
        group           = 'LLM',
        type_parquet    = 'BOOLEAN',
        type_polars     = pl.Boolean,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_profanity_check').is_in((True, False)))
                .otherwise(pl.col('llm_profanity_check') == False)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'llm_stream',
        group           = 'LLM',
        type_parquet    = 'BOOLEAN',
        type_polars     = pl.Boolean,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'llm')
                .then(pl.col('llm_stream').is_in((True, False)))
                .otherwise(pl.col('llm_stream') == False)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'http_method',
        group           = 'HTTP',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.Enum(categories = Categories.http_method + ('NONE', Sentinel.token)),
        sentinel        = Sentinel.none,
        validation      = (
            pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                .then(pl.col('http_method').is_in(Categories.http_method))
                .otherwise(pl.col('http_method') == 'NONE')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'http_path',
        group           = 'HTTP',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                .then(pl.col('http_path').str.len_chars() > 0)
                .otherwise(pl.col('http_path') == '')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'http_status_code',
        group           = 'HTTP',
        type_parquet    = 'INT64',
        type_polars     = pl.Int64,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                .then(pl.col('http_status_code').is_between(100, 599))
                .otherwise(pl.col('http_status_code') == -1)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'request_headers',
        group           = 'HTTP',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                .then(pl.col('request_headers').str.contains(Patterns.json_like))
                .otherwise(pl.col('request_headers') == '')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'response_headers',
        group           = 'HTTP',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind').is_in(('input_request', 'output_request')))
                .then(pl.col('response_headers').str.contains(Patterns.json_like))
                .otherwise(pl.col('response_headers') == '')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'kafka_topic',
        group           = 'Kafka',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind').is_in(('kafka_produce', 'kafka_consume')))
                .then(pl.col('kafka_topic').str.len_chars() > 0)
                .otherwise(pl.col('kafka_topic') == '')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'kafka_cluster',
        group           = 'Kafka',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind').is_in(('kafka_produce', 'kafka_consume')))
                .then(pl.col('kafka_cluster').str.len_chars() > 0)
                .otherwise(pl.col('kafka_cluster') == '')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'kafka_consumer_group',
        group           = 'Kafka',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind') == 'kafka_consume')
                .then(pl.col('kafka_consumer_group').str.len_chars() > 0)
                .otherwise(pl.col('kafka_consumer_group') == '')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'kafka_bootstrap_servers',
        group           = 'Kafka',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.when(pl.col('aef_kind').is_in(('kafka_produce', 'kafka_consume')))
                .then(pl.col('kafka_bootstrap_servers').str.contains(Patterns.json_str_array))
                .otherwise(pl.col('kafka_bootstrap_servers') == '')),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'meta_langgraph_step',
        group           = 'LangGraph_meta',
        type_parquet    = 'INT64',
        type_polars     = pl.Int64,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.col('meta_langgraph_step') >= -1),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'meta_langgraph_node',
        group           = 'LangGraph_meta',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.lit(True)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'meta_langgraph_triggers',
        group           = 'LangGraph_meta',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            (pl.col('meta_langgraph_triggers') == '') | pl.col('meta_langgraph_triggers').str.contains(Patterns.json_str_array)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'meta_langgraph_path',
        group           = 'LangGraph_meta',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            (pl.col('meta_langgraph_path') == '') | pl.col('meta_langgraph_path').str.contains(Patterns.json_str_array)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'meta_checkpoint_ns',
        group           = 'LangGraph_meta',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.lit(True)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'meta_tags',
        group           = 'LangGraph_meta',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            (pl.col('meta_tags') == '') | pl.col('meta_tags').str.contains(Patterns.json_str_array)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'meta_extra',
        group           = 'LangGraph_meta',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            (pl.col('meta_extra') == '') | pl.col('meta_extra').str.contains(Patterns.json_like)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'service_name',
        group           = 'service',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.lit(True)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'service_version',
        group           = 'service',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = Sentinel.unknown,
        validation      = (
            pl.lit(True)),
        mandatory = False,
    ),


    SpanAttr(
        name            = 'session_id_derived',
        group           = 'meta_attrs',
        type_parquet    = 'BOOLEAN',
        type_polars     = pl.Boolean,
        sentinel        = None,
        validation      = (
            ~(pl.col('session_id_derived') & pl.col('session_id_generated'))),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'session_id_generated',
        group           = 'meta_attrs',
        type_parquet    = 'BOOLEAN',
        type_polars     = pl.Boolean,
        sentinel        = None,
        validation      = (
            ~(pl.col('session_id_derived') & pl.col('session_id_generated'))),
        mandatory = True,
    ),


    SpanAttr(
        name            = 'nexus_distrib_ver',
        group           = 'service',
        type_parquet    = 'BYTE_ARRAY (UTF8)',
        type_polars     = pl.String,
        sentinel        = None,
        validation      = (
            pl.col('nexus_distrib_ver').str.contains(Patterns.nexus_version)),
        mandatory = False,
    ),
))


@dataclass(frozen = True)
class DataObject:
    trace_id        : ClassVar[str] = 'trace_id'
    span_id         : ClassVar[str] = 'span_id'
    parent_span_id  : ClassVar[str] = 'parent_span_id'
    agent_id        : ClassVar[str] = 'agent_id'
    session_id      : ClassVar[str] = 'session_id'

    order   : ClassVar[str] = 'start_time_ns'
    kind    : ClassVar[str] = 'aef_kind'
    status  : ClassVar[str] = 'status_code'

    input_text  : ClassVar[str] = 'input_text'
    output_text : ClassVar[str] = 'output_text'

    object_aggregation  : ClassVar[tuple[str, ...]] = ('trace_id', 'agent_id')
    sequence_key        : ClassVar[str]             = 'trace_id'

    objects_order   : ClassVar[tuple[tuple[str, bool], ...]]    = (
        ('agent_id', False), ('trace_id', False), ('start_time_ns', False))

    label           : ClassVar[str] = 'class'
    sublabel        : ClassVar[str] = 'anomaly_type'
    is_anomaly      : ClassVar[str] = 'is_anomaly'
    class_sentinel  : ClassVar[str] = 'unknown'

    def __post_init__(self):
        schema  = frozenset(SpansData.column_names)
        keyed   = tuple(map(lambda pair: pair[0], self.objects_order))
        roles   = (
            self.trace_id, self.span_id, self.parent_span_id, self.agent_id, self.session_id,
            self.order, self.kind, self.status, self.input_text, self.output_text, self.sequence_key)        
        
        if (unknown := next(filter(lambda c: c not in schema, chain(roles, self.object_aggregation, keyed)), None)) is not None:
            raise ValueError(f'роль объекта ссылается на отсутствующую в спецификации колонку: {unknown!r}')


type DType = type[pl.DataType] | pl.DataType


@dataclass(frozen = True, eq = False)
class Recast:
    nested  : ClassVar[tuple]   = (pl.List, pl.Array, pl.Struct, pl.Object)
    truthy  : ClassVar[tuple]   = ('true', 't', '1', 'yes', 'y')
    falsy   : ClassVar[tuple]   = ('false', 'f', '0', 'no', 'n')

    @staticmethod
    def scalar(source: DType) -> bool:
        return not isinstance(source, Recast.nested)

    @staticmethod
    def castable(source: DType, target: DType) -> bool:
        return source == target or Recast.scalar(source)

    @staticmethod
    def boolean(raw: pl.Expr) -> pl.Expr:
        token = raw.cast(pl.String, strict = False).str.strip_chars().str.to_lowercase()

        return (pl.when(token.is_in(Recast.truthy)).then(pl.lit(True))
                  .when(token.is_in(Recast.falsy)).then(pl.lit(False))
                  .otherwise(pl.lit(None, dtype = pl.Boolean)))

    @staticmethod
    def cast(raw: pl.Expr, source: DType, target: DType) -> pl.Expr:
        return (raw                      if source == target
                else Recast.boolean(raw) if target == pl.Boolean
                else raw.cast(target, strict = False))

    @staticmethod
    def column(a: SpanAttr, schema: pl.Schema) -> pl.Expr:
        target  = a.type_polars
        source  = schema.get(a.name)
        empty   = pl.lit(a.empty, dtype = target)
        if source is None or not Recast.scalar(source):
            return empty.alias(a.name)
        raw     = pl.col(a.name)
        casted  = Recast.cast(raw, source, target)
        garbage = raw.is_not_null() & casted.is_null()
        trash   = pl.lit(a.trash, dtype = target) if a.trash is not None else empty

        return pl.when(garbage).then(trash).otherwise(casted.fill_null(empty)).alias(a.name)

    @staticmethod
    def frame(lf: pl.LazyFrame, schema: pl.Schema, spec: SpansDataSpec = SpansData) -> pl.LazyFrame:
        return lf.select(map(lambda a: Recast.column(a, schema), spec.attrs))

    @staticmethod
    def overlay(lf: pl.LazyFrame, schema: pl.Schema, spec: SpansDataSpec = SpansData) -> pl.LazyFrame:
        present = frozenset(schema.names())

        return lf.with_columns(map(
            lambda a: Recast.column(a, schema),
            filter(lambda a: a.name in present, spec.attrs)))


def recast(frame: pl.DataFrame | pl.LazyFrame, spec: SpansDataSpec = SpansData) -> pl.LazyFrame:
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame

    return Recast.frame(lf, lf.collect_schema(), spec)