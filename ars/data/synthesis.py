from    typing                          import ClassVar, Literal
from    dataclasses                     import dataclass, field, replace
from    functools                       import reduce
from    itertools                       import accumulate, chain

from    datetime                        import datetime, timezone
from    pathlib                         import PurePath
from    operator                        import add
from    tempfile                        import mkdtemp
from    json                            import dumps
from    collections                     import deque

import  pyarrow.parquet                 as pq

import  polars                          as pl

from    ars.configuration.c0__device    import Device
from    ars.specification.spec          import SpansData
from    ars.specification.core          import SpansDataSpec
from    ars.data.validation             import Quality, Readiness, Policy, By, scan
from    ars.tools.performance           import perf
from    ars.tools.visualisations        import viz
from    ars.tools.tui.tui               import ColorSchemeDataScience
from    ars.tools.tui.tui_data          import ColorSchemeDataScienceSakura


@dataclass(frozen = True)
class Step:
    token       : str
    aef_kind    : str
    span_name   : str


@dataclass(frozen = True)
class Scenario:
    name    : str
    weight  : float
    steps   : tuple[str, ...]


@dataclass(frozen = True)
class Override:
    column  : str
    op      : Literal['literal', 'before_start']
    operand : object


@dataclass(frozen = True)
class Defect:
    name        : str
    weight      : float
    overrides   : tuple[Override, ...]


@dataclass(frozen = True)
class Agent:
    id          : str
    name        : str
    traces      : int
    trend       : float
    corrupt     : float
    versions    : tuple[str, ...]


@dataclass(frozen = True)
class Salt:
    scenario    : ClassVar[int] = 101
    version     : ClassVar[int] = 102
    day         : ClassVar[int] = 103
    intra       : ClassVar[int] = 104
    defect      : ClassVar[int] = 105
    pick        : ClassVar[int] = 106
    victim      : ClassVar[int] = 107
    dur         : ClassVar[int] = 108
    gap         : ClassVar[int] = 109
    status      : ClassVar[int] = 110
    message     : ClassVar[int] = 111
    task        : ClassVar[int] = 112
    ptok        : ClassVar[int] = 113
    ctok        : ClassVar[int] = 114
    precache    : ClassVar[int] = 115
    temp        : ClassVar[int] = 116
    topp        : ClassVar[int] = 117
    maxtok      : ClassVar[int] = 118
    rep         : ClassVar[int] = 119
    prof        : ClassVar[int] = 120
    stream      : ClassVar[int] = 121
    method      : ClassVar[int] = 122
    path        : ClassVar[int] = 123
    code        : ClassVar[int] = 124
    session     : ClassVar[int] = 125
    results     : ClassVar[int] = 126


@dataclass(frozen = True)
class Catalog:
    sec_ns  : ClassVar[int] = 1_000_000_000

    @staticmethod
    def steps() -> tuple[Step, ...]:
        return (
            Step('llm',             'llm',              'llm.generate'),
            Step('llm2',            'llm',              'llm.summarize'),
            Step('tool_flights',    'tool',             'tool.search_flights'),
            Step('tool_hotels',     'tool',             'tool.search_hotels'),
            Step('tool_weather',    'tool',             'tool.get_weather'),
            Step('retriever',       'retriever',        'retriever.search'),
            Step('chain',           'chain',            'chain.step'),
            Step('http_get',        'output_request',   'http.get'),
            Step('http_post',       'output_request',   'http.post'),
            Step('kafka_pub',       'kafka_produce',    'kafka.produce'),
            Step('kafka_sub',       'kafka_consume',    'kafka.consume'),
            Step('guard',           'guard',            'guard.safety'))

    @staticmethod
    def scenarios() -> tuple[Scenario, ...]:
        return (
            Scenario('simple',      0.15,   ('llm', 'llm2')),
            Scenario('qa',          0.20,   ('retriever', 'llm', 'llm2')),
            Scenario('planning',    0.30,   ('llm', 'tool_flights', 'tool_hotels', 'tool_weather', 'retriever', 'llm2', 'guard')),
            Scenario('rich',        0.15,   ('llm', 'retriever', 'tool_flights', 'tool_hotels', 'http_get', 'http_post', 'kafka_pub', 'guard', 'llm2', 'chain', 'llm', 'tool_weather')),
            Scenario('http',        0.10,   ('http_get', 'http_get', 'http_post', 'http_get', 'llm', 'llm2')),
            Scenario('kafka',       0.10,   ('kafka_sub', 'llm', 'tool_weather', 'retriever', 'kafka_pub', 'llm2', 'guard')))

    @staticmethod
    def agents() -> tuple[Agent, ...]:
        return (
            Agent('CI1', 'travel-planner',  16,  0.8, 0.00, ('2.3.1', '2.4.0')),
            Agent('CI2', 'booking-agent',   10,  0.0, 0.80, ('1.8.0', '1.8.1')),
            Agent('CE7', 'enrichment-svc',   8, -0.6, 2.00, ('0.9.4', '?')))

    @staticmethod
    def defects() -> tuple[Defect, ...]:
        return (
            Defect('time_reversed',         1.0, (Override('end_time_ns', 'before_start', 1_000_000),)),
            Defect('bad_input_json',        1.0, (Override('input_text', 'literal', '{"city": "Paris"'),)),
            Defect('both_session_flags',    1.0, (Override('session_id_derived', 'literal', True), Override('session_id_generated', 'literal', True))),
            Defect('null_output',           1.0, (Override('output_text', 'literal', None),)),
            Defect('bad_id',                1.0, (Override('span_id', 'literal', 'span id not base64!'),)),
            Defect('bad_temp',              1.0, (Override('llm_temperature', 'literal', -0.5),)),
            Defect('bad_http_status',       1.0, (Override('http_status_code', 'literal', 999),)))

    @staticmethod
    def root() -> Step:
        return Step('start', 'start_agent', 'agent.invoke')

    @staticmethod
    def epoch_ns(year: int = 2026, month: int = 5, day: int = 15) -> int:
        return int(datetime(year, month, day, tzinfo = timezone.utc).timestamp()) * Catalog.sec_ns


@dataclass(frozen = True)
class GenConfig:
    seed            : int           = 20_250_601
    target_spans    : None | int    = 1000000
    base_ns         : int           = field(default_factory = Catalog.epoch_ns)
    day_ns          : int           = 86_400_000_000_000
    window_days     : int           = 40

    alphabet            : str   = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    id_length           : int   = 10
    traces_per_session  : int   = 3

    root        : Step                  = field(default_factory = Catalog.root)
    steps       : tuple[Step, ...]      = field(default_factory = Catalog.steps)
    scenarios   : tuple[Scenario, ...]  = field(default_factory = Catalog.scenarios)
    agents      : tuple[Agent, ...]     = field(default_factory = Catalog.agents)
    defects     : tuple[Defect, ...]    = field(default_factory = Catalog.defects)

    corrupt_rate    : float = 0.10
    error_rate      : float = 0.05
    unset_rate      : float = 0.01

    dur_min_ns  : int   = 2_000_000
    dur_max_ns  : int   = 90_000_000
    gap_max_ns  : int   = 12_000_000
    tokens_min  : int   = 200
    tokens_max  : int   = 4_000

    models      : tuple[str, ...]   = ('gpt-4o', 'claude-3.7-sonnet', 'llama-3.1-70b')
    tasks       : tuple[str, ...]   = (
        'Plan a 5-day trip to Tokyo',
        'Find cheap flights to Rome',
        'Weekend near Berlin',
        'Honeymoon in Bali',
        'Budget tour of Portugal',
        'Business trip to Singapore')
    paths       : tuple[str, ...]   = ('/flights', '/hotels', '/weather')
    messages    : tuple[str, ...]   = ('timeout', 'tool failed', 'upstream 500')
    http_codes  : tuple[int, ...]   = (200, 200, 200, 201, 404, 500)

    headers_json    : str   = '{"Content-Type": "application/json"}'
    bootstrap_json  : str   = '["kafka-1:9092", "kafka-2:9092"]'
    lg_path_json    : str   = '["__start__", "planner"]'
    lg_tags_json    : str   = '["llm", "node"]'

    recast              : bool                      = False
    device              : Device                    = field(default_factory = Device)
    chunk_traces        : int                       = 1_000_000
    output_color_scheme : ColorSchemeDataScience    = field(default_factory = ColorSchemeDataScience)


@dataclass(frozen = True)
class Rand:
    @staticmethod
    def u01(key: pl.Expr, salt: int) -> pl.Expr:
        return (key.hash(seed = salt) % 1_000_000) / 1_000_000.0

    @staticmethod
    def randint(key: pl.Expr, salt: int, lo: int, hi: int) -> pl.Expr:
        return (lo + key.hash(seed = salt) % (hi - lo + 1)).cast(pl.Int64)

    @staticmethod
    def index(key: pl.Expr, salt: int, size: int) -> pl.Expr:
        return (key.hash(seed = salt) % size).cast(pl.Int64)

    @staticmethod
    def pick(key: pl.Expr, salt: int, options: tuple, dtype: type[pl.DataType] | pl.DataType) -> pl.Expr:
        return Rand.index(key, salt, len(options)).replace_strict(
            list(range(len(options))), list(options), return_dtype = dtype)

    @staticmethod
    def b64(ordinal: pl.Expr, length: int, alphabet: str) -> pl.Expr:
        digit = lambda k: ((ordinal // (64 ** k)) % 64).cast(pl.Int64).replace_strict(
            list(range(64)), list(alphabet), return_dtype = pl.String)

        return pl.concat_str(map(digit, range(length)), separator = '')


@dataclass(frozen = True)
class Build:
    @staticmethod
    def kind(kinds: tuple[str, ...], value: pl.Expr, sentinel: object) -> pl.Expr:
        return pl.when(pl.col('aef_kind').is_in(kinds)).then(value).otherwise(pl.lit(sentinel))

    @staticmethod
    def step_table(config: GenConfig) -> pl.DataFrame:
        by_token    = dict(map(lambda st: (st.token, st), config.steps))
        root_row    = lambda sc: (sc.name, 0, config.root.aef_kind, config.root.span_name)
        body        = lambda sc: map(
            lambda it: (sc.name, it[0] + 1, by_token[it[1]].aef_kind, by_token[it[1]].span_name),
            enumerate(sc.steps))
        rows        = chain.from_iterable(map(lambda sc: chain((root_row(sc),), body(sc)), config.scenarios))

        return pl.DataFrame(list(rows), schema = ('scenario', 'step', 'aef_kind', 'span_name'), orient = 'row')

    @staticmethod
    def scenario_edges(config: GenConfig) -> tuple[float, ...]:
        total = sum(map(lambda s: s.weight, config.scenarios))

        return tuple(accumulate(map(lambda s: s.weight / total, config.scenarios)))[:-1]

    @staticmethod
    def agent_traces(config: GenConfig, agent: Agent, offset: int) -> pl.LazyFrame:
        uid             = pl.col('trace_uid')
        names           = tuple(map(lambda s: s.name, config.scenarios))
        sizes           = tuple(map(lambda s: 1 + len(s.steps), config.scenarios))
        edges           = Build.scenario_edges(config)
        u_scn           = Rand.u01(uid, Salt.scenario)
        s_idx           = reduce(add, map(lambda e: (u_scn >= e).cast(pl.Int64), edges), pl.lit(0, dtype = pl.Int64))
        u_day           = Rand.u01(uid, Salt.day).pow(1.0 / (1.0 + agent.trend))
        defect_names    = tuple(map(lambda d: d.name, config.defects))
        defect_edges    = tuple(accumulate(map(lambda d: d.weight / sum(map(lambda x: x.weight, config.defects)), config.defects)))[:-1]
        u_def           = Rand.u01(uid, Salt.pick)
        d_idx           = reduce(add, map(lambda e: (u_def >= e).cast(pl.Int64), defect_edges), pl.lit(0, dtype = pl.Int64))
        session_ordinal = (uid // config.traces_per_session)

        return (
            pl.select(pl.int_range(offset, offset + agent.traces, dtype = pl.Int64).alias('trace_uid')).lazy()
            .with_columns(
                pl.lit(agent.id).alias('agent_id'),
                pl.lit(agent.name).alias('service_name'),
                pl.lit(agent.versions[0]).alias('service_version'),
                Rand.pick(uid, Salt.version, agent.versions, pl.String).alias('nexus_distrib_ver'),
                s_idx.replace_strict(list(range(len(names))), list(names), return_dtype = pl.String).alias('scenario'),
                s_idx.replace_strict(list(range(len(sizes))), list(sizes), return_dtype = pl.Int64).alias('length'),
                Rand.b64(session_ordinal, config.id_length, config.alphabet).alias('session_id'),
                (Rand.u01(uid, Salt.session) < 0.6).alias('session_id_derived'),
                ((Rand.u01(uid, Salt.session) >= 0.6) & (Rand.u01(uid, Salt.session) < 0.8)).alias('session_id_generated'),
                (pl.lit(config.base_ns)
                    + (u_day * config.window_days).floor().cast(pl.Int64) * config.day_ns
                    + Rand.randint(uid, Salt.intra, 0, 64_800) * 1_000_000_000).alias('trace_start_ns'),
                pl.when(Rand.u01(uid, Salt.defect) < config.corrupt_rate * agent.corrupt)
                  .then(d_idx.replace_strict(list(range(len(defect_names))), list(defect_names), return_dtype = pl.String))
                  .otherwise(pl.lit('none')).alias('defect'))
            .with_columns(
                (uid.hash(seed = Salt.victim) % pl.col('length')).cast(pl.Int64).alias('victim_step')))

    @staticmethod
    def avg_spans(config: GenConfig) -> float:
        total = sum(map(lambda s: s.weight, config.scenarios))

        return sum(map(lambda s: s.weight / total * (1 + len(s.steps)), config.scenarios))

    @staticmethod
    def effective_agents(config: GenConfig) -> tuple[Agent, ...]:
        if config.target_spans is None: return config.agents

        base    = sum(map(lambda a: a.traces, config.agents))
        total   = max(len(config.agents), round(config.target_spans / Build.avg_spans(config)))
        rescale = lambda a: replace(a, traces = max(1, round(total * a.traces / base)))

        return tuple(map(rescale, config.agents))

    @staticmethod
    def traces(config: GenConfig) -> pl.LazyFrame:
        agents  = Build.effective_agents(config)
        counts  = tuple(map(lambda a: a.traces, agents))
        offsets = (0,) + tuple(accumulate(counts))[:-1]
        blocks  = map(lambda ao: Build.agent_traces(config, ao[0], ao[1]), zip(agents, offsets))

        return pl.concat(blocks, how = 'vertical')

    @staticmethod
    def numbered(config: GenConfig) -> pl.LazyFrame:
        return Build.traces(config).with_columns(
            (pl.col('length').cum_sum() - pl.col('length')).alias('span_base'))

    @staticmethod
    def chunks(total: int, size: int) -> tuple[tuple[int, int], ...]:
        return tuple(map(lambda s: (s, min(s + size, total)), range(0, total, size)))

    @staticmethod
    def spans(config: GenConfig, traces: pl.LazyFrame) -> pl.LazyFrame:
        alpha, L    = config.alphabet, config.id_length
        span        = pl.col('span_uid')
        llm         = ('llm',)
        http        = ('output_request',)
        kafka       = ('kafka_produce', 'kafka_consume')
        lg          = ('llm', 'chain', 'tool', 'retriever')
        p_raw       = Rand.randint(span, Salt.ptok, config.tokens_min, config.tokens_max)
        c_raw       = Rand.randint(span, Salt.ctok, 50, 1500)
        task        = Rand.pick(span, Salt.task, config.tasks, pl.String)
        dur         = Rand.randint(span, Salt.dur, config.dur_min_ns, config.dur_max_ns)
        gap         = pl.when(pl.col('step') == 0).then(0).otherwise(Rand.randint(span, Salt.gap, 0, config.gap_max_ns))
        u_status    = Rand.u01(span, Salt.status)

        exploded = (traces
            .with_columns(pl.int_ranges(0, pl.col('length')).alias('step'))
            .explode('step')
            .with_columns((pl.col('span_base') + pl.col('step')).alias('span_uid'))
            .join(Build.step_table(config).lazy(), on = ('scenario', 'step'), how = 'left'))

        identified = exploded.with_columns(
            Rand.b64(pl.col('trace_uid'), L, alpha).alias('trace_id'),
            Rand.b64(span, L, alpha).alias('span_id'))

        timed = identified.with_columns(
            ((dur + gap).cum_sum().over('trace_uid', order_by = 'step') - (dur + gap)).alias('_offset'),
            dur.alias('_dur')).with_columns(
            (pl.col('trace_start_ns') + pl.col('_offset')).alias('start_time_ns')).with_columns(
            (pl.col('start_time_ns') + pl.col('_dur')).alias('end_time_ns'))

        root_id = Rand.b64(pl.col('span_base'), L, alpha)

        classified = timed.with_columns(
            pl.when(pl.col('step') == 0).then(pl.lit('root')).otherwise(root_id).alias('parent_span_id'),
            pl.when(pl.col('step') == 0).then(pl.lit('outside')).otherwise(root_id).alias('origin_span_id'),
            pl.when(u_status < config.error_rate).then(pl.lit('STATUS_CODE_ERROR'))
              .when(u_status < config.error_rate + config.unset_rate).then(pl.lit('STATUS_CODE_UNSET'))
              .otherwise(pl.lit('STATUS_CODE_OK')).alias('status_code'))

        return classified.with_columns(
            Build.kind(llm, Rand.pick(span, Salt.maxtok, config.models, pl.String), '').alias('llm_model'),
            Build.kind(llm, p_raw, -1).alias('llm_prompt_tokens'),
            Build.kind(llm, c_raw, -1).alias('llm_completion_tokens'),
            Build.kind(llm, p_raw + c_raw, -1).alias('llm_total_tokens'),
            Build.kind(llm, Rand.randint(span, Salt.precache, 0, 200), -1).alias('llm_precached_prompt_tokens'),
            Build.kind(llm, Rand.u01(span, Salt.temp).round(3), -1.0).alias('llm_temperature'),
            Build.kind(llm, (0.85 + Rand.u01(span, Salt.topp) * 0.15).round(3), -1.0).alias('llm_top_p'),
            Build.kind(llm, Rand.pick(span, Salt.maxtok, (512, 1024, 2048, 4096), pl.Int64), -1).alias('llm_max_tokens'),
            Build.kind(llm, (1.0 + Rand.u01(span, Salt.rep) * 0.3).round(3), -1.0).alias('llm_repetition_penalty'),
            Build.kind(llm, Rand.u01(span, Salt.prof) < 0.6, False).alias('llm_profanity_check'),
            Build.kind(llm, Rand.u01(span, Salt.stream) < 0.5, False).alias('llm_stream'),
            Build.kind(http, Rand.pick(span, Salt.method, ('GET', 'POST', 'PUT'), pl.String), 'NONE').alias('http_method'),
            Build.kind(http, Rand.pick(span, Salt.path, config.paths, pl.String), '').alias('http_path'),
            Build.kind(http, Rand.pick(span, Salt.code, config.http_codes, pl.Int64), -1).alias('http_status_code'),
            Build.kind(http, pl.lit(config.headers_json), '').alias('request_headers'),
            Build.kind(http, pl.lit(config.headers_json), '').alias('response_headers'),
            Build.kind(kafka, pl.lit('itineraries'), '').alias('kafka_topic'),
            Build.kind(kafka, pl.lit('prod'), '').alias('kafka_cluster'),
            Build.kind(('kafka_consume',), pl.lit('planner-cg'), '').alias('kafka_consumer_group'),
            Build.kind(kafka, pl.lit(config.bootstrap_json), '').alias('kafka_bootstrap_servers'),
            Build.kind(lg, pl.col('step'), -1).alias('meta_langgraph_step'),
            Build.kind(lg, pl.col('span_name'), '').alias('meta_langgraph_node'),
            pl.lit('').alias('meta_langgraph_triggers'),
            Build.kind(lg, pl.lit(config.lg_path_json), '').alias('meta_langgraph_path'),
            pl.lit('').alias('meta_checkpoint_ns'),
            Build.kind(lg, pl.lit(config.lg_tags_json), '').alias('meta_tags'),
            pl.lit('').alias('meta_extra'),
            (pl.when(pl.col('aef_kind') == 'llm')
               .then(pl.concat_str((pl.lit('[{"role": "user", "content": "'), task, pl.lit('"}]'))))
               .when(pl.col('aef_kind').is_in(('tool', 'retriever')))
               .then(pl.concat_str((pl.lit('{"query": "'), pl.col('span_name'), pl.lit('"}'))))
               .when(pl.col('aef_kind').is_in(http))
               .then(pl.concat_str((pl.lit('{"path": "'), Rand.pick(span, Salt.path, config.paths, pl.String), pl.lit('"}'))))
               .when(pl.col('aef_kind').is_in(kafka))
               .then(pl.concat_str((pl.lit('{"event": "'), pl.col('span_name'), pl.lit('"}'))))
               .when(pl.col('aef_kind') == 'start_agent')
               .then(pl.concat_str((pl.lit('{"task": "'), task, pl.lit('"}'))))
               .otherwise(pl.lit(''))).alias('input_text'),
            (pl.when(pl.col('aef_kind') == 'llm').then(pl.lit('Here is the plan with flights and hotels.'))
               .when(pl.col('aef_kind') == 'tool')
               .then(pl.concat_str((pl.lit('{"results": '), Rand.randint(span, Salt.results, 0, 25).cast(pl.String), pl.lit('}'))))
               .when(pl.col('aef_kind') == 'guard').then(pl.lit('passed'))
               .when(pl.col('aef_kind') == 'start_agent').then(pl.lit('Final itinerary delivered to the user.'))
               .otherwise(pl.lit(''))).alias('output_text'),
            pl.when(pl.col('status_code') == 'STATUS_CODE_ERROR')
              .then(Rand.pick(span, Salt.message, config.messages, pl.String))
              .otherwise(pl.lit('')).alias('status_message'))

    @staticmethod
    def override_value(override: Override) -> pl.Expr:
        match override.op:
            case 'before_start': return pl.col('start_time_ns') - pl.lit(override.operand)
            case _:              return pl.lit(override.operand)

    @staticmethod
    def corrupt(config: GenConfig, spans: pl.LazyFrame) -> pl.LazyFrame:
        victim      = (pl.col('step') == pl.col('victim_step')) & (pl.col('defect') != 'none')
        pairs       = tuple(chain.from_iterable(map(
            lambda d: map(lambda ov: (ov.column, d.name, Build.override_value(ov)), d.overrides), config.defects)))
        columns     = dict.fromkeys(map(lambda p: p[0], pairs))
        override    = lambda col: reduce(
            lambda acc, p: pl.when(victim & (pl.col('defect') == p[1])).then(p[2]).otherwise(acc),
            filter(lambda p: p[0] == col, pairs), pl.col(col)).alias(col)

        return spans.with_columns(map(override, columns))

    @staticmethod
    def finalize(spans: pl.LazyFrame, spec: SpansDataSpec = SpansData, recast: bool = False) -> pl.LazyFrame:
        strict  = lambda a: pl.col(a.name).cast(a.type_polars).alias(a.name)
        lenient = lambda a: pl.col(a.name).cast(a.type_polars, strict = False).fill_null(a.empty).alias(a.name)

        return spans.select(map(lenient if recast else strict, spec.attrs))


def synthesize(config: GenConfig = GenConfig()) -> pl.LazyFrame:
    return Build.finalize(Build.corrupt(config, Build.spans(config, Build.numbered(config))), recast = config.recast)


@perf.benchmark('синтез + запись parquet')
def write(path: PurePath, config: GenConfig = GenConfig()) -> None:
    frame   = Build.numbered(config).collect(engine = config.device.engine)
    bounds  = Build.chunks(frame.height, config.chunk_traces)
    tables  = map(lambda lh: Build.finalize(
        Build.corrupt(config, Build.spans(config, frame[lh[0]:lh[1]].lazy())),
        recast = config.recast).collect(engine = config.device.engine).to_arrow(), bounds)

    head    = next(tables)
    writer  = pq.ParquetWriter(path.as_posix(), head.schema)

    writer.write_table(head)
    deque(map(writer.write_table, tables), maxlen = 0)
    writer.close()


@dataclass(frozen = True)
class Report:
    @staticmethod
    def summary(frame: pl.DataFrame) -> str:
        return dumps(
            {
                'spans'  : frame.height,
                'traces' : frame.get_column('trace_id').n_unique(),
                'agents' : frame.get_column('agent_id').n_unique(),
                'kinds'  : frame.get_column('aef_kind').n_unique()},
            ensure_ascii = False)

    @staticmethod
    def html(frame: pl.DataFrame) -> str:
        work        = PurePath(mkdtemp())
        volume      = (frame
            .group_by('agent_id')
            .agg(pl.len().alias('spans'), pl.col('trace_id').n_unique().alias('traces'))
            .sort('spans', descending = True))
        by_agent    = viz.save_value_counts(frame.get_column('agent_id'),  work, 'by_agent', 'Спаны по агентам')
        by_kind     = viz.save_value_counts(frame.get_column('aef_kind'),  work, 'by_kind',  'Спаны по типу (aef_kind)')
        by_span     = viz.save_value_counts(frame.get_column('span_name'), work, 'by_span',  'Топ имён спанов', top = 20)
        blocks      = (
            viz.block_cards('Сводка', (
                ('Спанов',  str(frame.height)),
                ('Трасс',   str(frame.get_column('trace_id').n_unique())),
                ('Агентов', str(frame.get_column('agent_id').n_unique())),
                ('Типов',   str(frame.get_column('aef_kind').n_unique())))),
            viz.block_image('Спаны по агентам', by_agent),
            viz.block_image('Спаны по типам',   by_kind),
            viz.block_image('Топ имён спанов',   by_span),
            viz.block_table('Объём по агентам',  volume))

        return viz.Html.doc('Отчёт генерации синтетики', f'спанов: {frame.height}', str().join(blocks))


def main(**params: str) -> dict:
    out_path    = params.pop('out_path', '/mnt/data/traces/raw/parquet/spans_example.parquet')
    config      = GenConfig(
        target_spans        = int(params['target_spans']),
        seed                = int(params['seed']),
        window_days         = int(params['window_days']),
        traces_per_session  = int(params['traces_per_session']),
        id_length           = int(params['id_length']),
        corrupt_rate        = float(params['corrupt_rate']),
        error_rate          = float(params['error_rate']),
        unset_rate          = float(params['unset_rate']),
        recast              = str(params.get('recast', 'false')).strip().lower() in ('true', '1', 'yes', 'on'))
    config      = replace(config, device = Device.of(params.get('device', ''), config.device))
    dont_colorize = str(params.get('dont_colorize_log', 'true')).strip().lower() in ('true', '1', 'yes', 'on')
    config      = replace(config, output_color_scheme = ColorSchemeDataScience() if dont_colorize else ColorSchemeDataScienceSakura())
    config.output_color_scheme.print_section('Синтез данных')
    frame, elapsed, cpu, gpu = perf.measure_during_call(synthesize(config).collect, (), {'engine': config.device.engine})
    config.output_color_scheme.print_performance_metrics('синтез + collect', elapsed, cpu, gpu)
    frame.write_parquet(PurePath(out_path).as_posix())

    return {
        'report_html'       : Report.html(frame),
        'parquet_path'      : PurePath(out_path).as_posix(),
        'summary'           : Report.summary(frame)}


if __name__ == '__main__':
    out     = PurePath('/mnt/data/traces/raw/synthetic__trip_planner.parquet')
    config  = GenConfig()
    config  = replace(config, device = Device.of('', config.device))
    cs      = config.output_color_scheme
    perf.inject_color_scheme(globals(), cs)

    cs.print_section('Генерация синтетики')
    write(out, config)

    policy = Policy(
        window_days         = 60,
        volume_target       = 8,
        volume_steepness    = 2.0,
        rate_target         = 1.0,
        min_eff_traces      = 4,
        max_loss            = 0.30,
        min_quality         = 0.75,
        ready_at            = 0.60,
        pilot_at            = 0.40)

    cs.print_section('Диагностика (полные данные)')

    scanned = scan(out)

    overview    = scanned.select(
        pl.len().alias('spans'),
        pl.col('trace_id').n_unique().alias('traces'),
        pl.col('agent_id').n_unique().alias('agents'),
        pl.col('aef_kind').n_unique().alias('kinds')).collect(engine = config.device.engine)
    volume      = (scanned
        .group_by('agent_id')
        .agg(pl.len().alias('spans'), pl.col('trace_id').n_unique().alias('traces'))
        .sort('spans', descending = True).collect(engine = config.device.engine))
    stats       = overview.row(0, named = True)
    spans_n     = stats['spans']
    cols_n      = len(scanned.collect_schema())

    quality = Quality(scanned)
    ready   = Readiness(quality, policy)

    with pl.Config(tbl_rows = 80, tbl_cols = 40, tbl_width_chars = 10000, fmt_str_lengths = 140):
        print('строк:', spans_n, '| колонок:', cols_n)
        cs.print_subsection('объём по агентам')
        print(volume.sort('agent_id'))
        cs.print_subsection('вердикт')
        print(quality.verdict().collect(engine = config.device.engine))
        cs.print_subsection('отбраковка по агентам')
        print(quality.rejection(By.agent).collect(engine = config.device.engine))
        cs.print_subsection('готовность')
        print(ready.assess(By.agent).collect(engine = config.device.engine))
        cs.print_subsection('виновные атрибуты')
        print(ready.blame(By.agent).collect(engine = config.device.engine))

    cs.print_section('Графики и отчёт')

    _out        = PurePath('/mnt/data/reports/')
    _kinds      = scanned.group_by('aef_kind').agg(pl.len().alias('count')).sort('count', descending = True).collect(engine = config.device.engine)
    _spans      = scanned.group_by('span_name').agg(pl.len().alias('count')).sort('count', descending = True).head(20).collect(engine = config.device.engine)
    _by_agent   = viz.save_bars(volume, 'agent_id', 'spans', _out, 'by_agent', 'Спаны по агентам')
    _by_kind    = viz.save_bars(_kinds, 'aef_kind', 'count', _out, 'by_kind', 'Спаны по типу')
    _by_span    = viz.save_bars(_spans, 'span_name', 'count', _out, 'by_span', 'Топ-20 имён спанов')
    viz.write_report(
        _out / 'synthesis_report__trip_planner.html',
        'Отчёт генерации синтетики',
        f'спанов: {spans_n}',
        (
            viz.block_cards('Сводка', (
                ('Спанов',  str(stats['spans'])),
                ('Трасс',   str(stats['traces'])),
                ('Агентов', str(stats['agents'])),
                ('Типов',   str(stats['kinds'])))),
            viz.block_image('Спаны по агентам',         _by_agent),
            viz.block_image('Спаны по типам',           _by_kind),
            viz.block_image('Топ имён спанов',          _by_span),
            viz.block_table('Объём по агентам',         volume)))