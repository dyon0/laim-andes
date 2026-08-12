from    typing                  import Literal, ClassVar, Callable
from    dataclasses             import dataclass
from    functools               import reduce
from    itertools               import chain, starmap

from    datetime                import datetime, timezone
from    re                      import Pattern, compile

from    ars.tools.sql.dialect   import Dialect, Dialects
from    ars.tools.sql.template  import Template


@dataclass(frozen = True)
class Patterns:
    base64          : ClassVar[str] = r'^[A-Za-z0-9+/]+={0,2}$'
    codebase_elem   : ClassVar[str] = r'^C[IE][0-9]+$'
    nexus_version   : ClassVar[str] = r'^(\d+|\?)\.(\d+|\?)\.(\d+|\?)$'
    json_str_array  : ClassVar[str] = r'^\[\s*"([^"\\]|\\.)*"(\s*,\s*"([^"\\]|\\.)*")*\s*\]$'
    json_like       : ClassVar[str] = r'^\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*$'


@dataclass(frozen = True)
class Schema:
    agent   : str   = 'agent_id'
    trace   : str   = 'trace_id'
    span    : str   = 'span_id'

    begin   : str   = 'start_time_ns'
    end     : str   = 'end_time_ns'


@dataclass(frozen = True)
class Agent:
    @staticmethod
    def check(agents: tuple[str, ...]) -> tuple[str, ...]:
        return agents


@dataclass(frozen = True)
class Time:
    ns_per_sec  : ClassVar[int]     = 1_000_000_000
    ns_per_min  : ClassVar[int]     = 60 * ns_per_sec
    ns_digits   : ClassVar[int]     = 11
    pattern     : ClassVar[Pattern] = compile(r'^(\d{4})\.(\d{2})\.(\d{2})-(\d{2}):(\d{2}):' + rf'(\d{{{ns_digits}}})$')

    @staticmethod
    def epoch_ns(text: str) -> int:
        if (_match := Time.pattern.match(text)) is None:
            raise ValueError(f'время должно быть в формате yyyy.mm.dd-hh:mm:ns с ровно {Time.ns_digits} цифрами наносекунд, получено: {text!r}')

        year, month, day, hour, minute, ns = map(int, _match.groups())
        if ns >= Time.ns_per_min:
            raise ValueError(f'поле ns = {ns} должно быть < {Time.ns_per_min} (количество наносекунд внутри минуты)')

        return int(datetime(year, month, day, hour, minute, tzinfo = timezone.utc).timestamp()) * Time.ns_per_sec + ns


@dataclass(frozen = True)
class Ident:
    pattern : ClassVar[Pattern] = compile(r'^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$')

    @staticmethod
    def check(name: str) -> str:
        if Ident.pattern.match(name) is None:
            raise ValueError(f'недопустимое имя таблицы: {name!r}')

        return name


@dataclass(frozen = True)
class Selection:
    agents  : tuple[str, ...]   = ()

    t_min   : None | str    = None
    t_max   : None | str    = None

    limit   : None | int    = None

    eligible    : bool  = False
    inclusive   : bool  = True

    holdout_method  : Literal['recent', 'random']   = 'random'
    holdout_percent : float                         = 0.0

    seed    : None | int    = None

    def __post_init__(self):
        Agent.check(self.agents)

        lo  = Time.epoch_ns(self.t_min) if self.t_min is not None else None
        hi  = Time.epoch_ns(self.t_max) if self.t_max is not None else None
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f't_min должен быть не позже t_max, получено: {self.t_min!r} > {self.t_max!r}')

        if self.limit is not None and self.limit <= 0:
            raise ValueError(f'limit должен быть положительным или None, получено: {self.limit}')

        if self.holdout_method not in ('recent', 'random'):
            raise ValueError(f'holdout_method должен быть "recent" или "random", получено: {self.holdout_method!r}')

        if not 0.0 <= self.holdout_percent <= 100.0:
            raise ValueError(f'holdout_percent должен быть в [0, 100], получено: {self.holdout_percent}')

        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f'seed должен быть int или None, получено: {self.seed!r}')


@dataclass(frozen = True)
class Params:
    agents  : tuple[str, ...]

    t_min   : None | int
    t_max   : None | int

    limit   : None | int

    holdout_frac    : float

    seed    : None | int

    @property
    def mapping(self) -> dict:
        scalars = (
            ('t_min',           self.t_min),
            ('t_max',           self.t_max),
            ('limit',           self.limit),
            ('holdout_frac',    self.holdout_frac),
            ('seed',            self.seed))

        return dict(
            chain(filter(lambda p: p[1] is not None, scalars),
            zip(map('agent_{}'.format, range(len(self.agents))), self.agents)))


@dataclass(frozen = True)
class Lit:
    tick    : ClassVar[str] = '\''

    @staticmethod
    def render(value: None | int | float | str) -> str:
        return (str(value) if isinstance(value, (int, float))
                else Lit.tick + str(value).replace(Lit.tick, Lit.tick * 2) + Lit.tick)


@dataclass(frozen = True)
class Holdout:
    col     : ClassVar[str]     = 'm.is_holdout'
    preds   : ClassVar[dict]    = {'full': '', 'train': f'NOT {col}', 'test': col}

    @staticmethod
    def where(subset: str) -> str:
        return f'\nWHERE {Holdout.preds[subset]}' if Holdout.preds[subset] else ''

    @staticmethod
    def conj(subset: str) -> str:
        return f' AND {Holdout.preds[subset]}' if Holdout.preds[subset] else ''


@dataclass(frozen = True)
class Clause:
    lcg_a   : ClassVar[str] = '1103515245'
    lcg_c   : ClassVar[str] = '12345'
    lcg_mod : ClassVar[str] = '2147483648'

    id  : ClassVar[str] = '(1 = 1)'

    @staticmethod
    def conj(*preds: None | str) -> str:
        live = tuple(filter(None, preds))

        return ' AND '.join(live) if live else Clause.id

    @staticmethod
    def agent(schema: Schema, dialect: Dialect, agents: tuple[str, ...]) -> str:
        names = ', '.join(map(dialect.param.format, map('agent_{}'.format, range(len(agents)))))

        return f'({schema.agent} IS NOT NULL AND {schema.agent} IN ({names}))' if agents else Clause.id

    @staticmethod
    def trace_time(schema: Schema, lo: None | str, hi: None | str) -> None | str:
        return Clause.conj(
            f'MIN({schema.begin}) >= {lo}' if lo else None,
            f'MAX({schema.end})   <= {hi}' if hi else None) if (lo or hi) else None

    @staticmethod
    def span_ok(schema: Schema, agent: str, lo: None | str, hi: None | str) -> str:
        return Clause.conj(
            agent if agent != Clause.id else None,
            f'{schema.begin} >= {lo}' if lo else None,
            f'{schema.begin} <= {hi}' if hi else None)

    @staticmethod
    def seeded_key(dialect: Dialect, pos: str, seed: str) -> str:
        mod = dialect.modulo.format
        h0  = mod(f'{pos} * {Clause.lcg_a} + {mod(seed, Clause.lcg_mod)}', Clause.lcg_mod)

        return mod(f'{h0} * {Clause.lcg_a} + {Clause.lcg_c}', Clause.lcg_mod)

    @staticmethod
    def holdout(dialect: Dialect, selection: Selection) -> tuple[str, str]:
        cut     = dialect.floor.format(f'(1.0 - {dialect.param.format('holdout_frac')}) * COUNT(*) OVER ()')
        order   = (
            'trace_begin DESC, trace DESC'                                                      if selection.holdout_method == 'recent'
            else f'{Clause.seeded_key(dialect, 'pos', dialect.param.format('seed'))}, trace'    if selection.seed is not None
            else dialect.random)

        return order, cut


@dataclass(frozen = True)
class Plan:
    table       : str
    dialect     : Dialect
    schema      : Schema
    selection   : Selection

    @property
    def lo(self) -> None | str:
        return self.dialect.param.format('t_min') if self.selection.t_min is not None else None

    @property
    def hi(self) -> None | str:
        return self.dialect.param.format('t_max') if self.selection.t_max is not None else None

    @property
    def limit(self) -> None | str:
        return self.dialect.param.format('limit') if self.selection.limit is not None else None

    @property
    def agent(self) -> str:
        return Clause.agent(self.schema, self.dialect, self.selection.agents)

    @property
    def span_ok(self) -> str:
        return Clause.span_ok(self.schema, self.agent, self.lo, self.hi)

    @property
    def holdout(self) -> tuple[str, str]:
        return Clause.holdout(self.dialect, self.selection)

    @property
    def capped(self) -> str:
        return Template.capped_all if self.limit is None else Template.capped_limit.format(limit = self.limit)

    @property
    def params(self) -> Params:
        return Params(
            agents          = self.selection.agents,
            t_min           = Time.epoch_ns(self.selection.t_min) if self.selection.t_min is not None else None,
            t_max           = Time.epoch_ns(self.selection.t_max) if self.selection.t_max is not None else None,
            limit           = self.selection.limit,
            holdout_frac    = self.selection.holdout_percent / 100.0,
            seed            = self.selection.seed if self.selection.holdout_method == 'random' else None)


@dataclass(frozen = True)
class Mode:
    @staticmethod
    def strict(plan: Plan) -> tuple[tuple[str, ...], Callable[[str], str]]:
        order, cut  = plan.holdout
        having      = Clause.conj(
            plan.dialect.bool_and.format(plan.agent) if plan.selection.agents else None,
            Clause.trace_time(plan.schema, plan.lo, plan.hi))
        ctes        = (
            Template.eligible.format(schema = plan.schema, table = plan.table, having = having),
            plan.capped,
            Template.ordered.format(source = 'capped'),
            Template.marked.format(order = order, cut = cut))
        tail        = lambda subset: Template.output_aligned.format(
            schema  = plan.schema,
            table   = plan.table,
            holdout = Holdout.where(subset))

        return ctes, tail

    @staticmethod
    def loose(plan: Plan) -> tuple[tuple[str, ...], Callable[[str], str]]:
        order, cut  = plan.holdout
        ctes        = (Template.eligible.format(
            schema  = plan.schema,
            table   = plan.table,
            having  = plan.dialect.bool_or.format(plan.span_ok)),
            plan.capped,
            Template.ordered.format(source = 'capped'),
            Template.marked.format(order = order, cut = cut))
        tail        = lambda subset: Template.output_aligned.format(
            schema  = plan.schema,
            table   = plan.table,
            holdout = Holdout.where(subset))

        return ctes, tail

    @staticmethod
    def none(plan: Plan) -> tuple[tuple[str, ...], Callable[[str], str]]:
        order, cut  = plan.holdout
        ctes        = (
            Template.present.format(schema = plan.schema, table = plan.table, span_ok = plan.span_ok),
            Template.ordered.format(source = 'present'),
            Template.marked.format(order = order, cut = cut))
        tail        = lambda subset: Template.output_open.format(
            schema  = plan.schema,
            table   = plan.table,
            span_ok = plan.span_ok,
            holdout = Holdout.conj(subset),
            limit   = Template.limit.format(param = plan.limit) if plan.limit is not None else '')

        return ctes, tail


@dataclass(frozen = True)
class Render:
    head    : ClassVar[str] = (
        '<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">'
        '<style>'
        'body{font-family:"Segoe UI",Tahoma,sans-serif;line-height:1.5;color:#2c3e50;background:#f4f6f9;max-width:1100px;margin:auto;padding:20px}'
        'h1{color:#1a2a3a;border-bottom:2px solid #2980b9;padding-bottom:8px}'
        'section{background:#fff;border:1px solid #dce1e8;border-radius:8px;padding:14px 18px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.05)}'
        'h2{color:#1a2a3a;font-size:1.05em;margin:0 0 10px}'
        'pre{background:#f8f9fa;border:1px solid #e0e3e8;border-radius:6px;padding:12px;overflow-x:auto;font-size:13px;line-height:1.4;margin:0}'
        'code{font-family:"Nimbus Mono PS","Courier New",monospace}'
        '</style></head><body><h1>Сгенерированные SQL-запросы</h1>')

    tail    : ClassVar[str] = '</body></html>'

    @staticmethod
    def inline(sql: str, dialect: Dialect, mapping: dict) -> str:
        bound = sorted(
            map(lambda kv: (dialect.param.format(kv[0]), Lit.render(kv[1])), mapping.items()),
            key     = lambda pair: len(pair[0]),
            reverse = True)

        return reduce(lambda acc, pair: acc.replace(*pair), bound, sql)

    @staticmethod
    def escape(text: str) -> str:
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    @staticmethod
    def report(queries: dict) -> str:
        block = lambda title, sql: f'<section><h2>{title}</h2><pre><code>{Render.escape(sql)}</code></pre></section>'

        return Render.head + str().join(starmap(block, (
            ('Полная выборка — все трассы с разметкой holdout',     queries['full']),
            ('Train — только тренировочные трассы (holdout = false)', queries['train']),
            ('Test — только тестовые трассы (holdout = true)',       queries['test']))) ) + Render.tail


def build(table: str, dialect: Dialect, schema: Schema, selection: Selection) -> tuple[dict[str, str], Params]:
    plan        = Plan(Ident.check(table), dialect, schema, selection)
    assemble    = (
        Mode.strict if selection.inclusive else
        Mode.loose  if selection.eligible  else
        Mode.none)
    ctes, tail  = assemble(plan)
    params      = plan.params
    queries     = dict(map(
        lambda subset: (subset, Render.inline(Template.query(ctes, tail(subset)), dialect, params.mapping)),
        ('full', 'train', 'test')))

    return queries, params


def fetch(connection, selection: Selection, table: str, schema: Schema, dialect: Dialect):
    queries, _ = build(table, dialect, schema, selection)

    return connection.execute(queries['full'])


def main(**params) -> dict[str, str]:
    table = params.get('table')
    if not table:
        raise ValueError('не указана таблица-источник')

    selection   = Selection(
        agents          = tuple(filter(None, map(str.strip, params.get('agents', '').split(',')))),
        t_min           = params.get('t_min') or None,
        t_max           = params.get('t_max') or None,
        limit           = int(params['limit']) if params.get('limit') else None,
        eligible        = bool(params.get('eligible', False)),
        inclusive       = bool(params.get('inclusive', False)),
        holdout_percent = float(params.get('holdout_percent') or 0.0),
        holdout_method  = params.get('holdout_method', 'random'),
        seed            = int(params['seed']) if params.get('seed') else None)
    dialect     = Dialects.of(params.get('dialect', 'FreeSQL'))
    queries, _  = build(table, dialect, Schema(), selection)

    return {
        'sql_query_full'    : queries['full'],
        'sql_query_train'   : queries['train'],
        'sql_query_test'    : queries['test'],
        'report_html'       : Render.report(queries)}


if __name__ == '__main__':
    out = main(
        table           = 'my_scheme.my_table',
        agents          = 'A01, B02',
        t_min           = '2025.05.25-00:00:00000000000',
        t_max           = '2025.06.01-23:59:59999999999',
        limit           = '',
        inclusive       = True,
        eligible        = False,
        holdout_percent = '20.0',
        holdout_method  = 'recent',
        seed            = '',
        dialect         = 'FreeSQL')
    
    print('\n------------------------- FULL  -------------------------\n')
    print(out['sql_query_full'])

    print('\n------------------------- TRAIN -------------------------\n')
    print(out['sql_query_train'])

    print('\n------------------------- TEST  -------------------------\n')
    print(out['sql_query_test'])