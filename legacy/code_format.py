from    dataclasses import dataclass
from    functools   import reduce, partial
from    itertools   import accumulate, chain

from    re          import Pattern, compile, match
from    sys         import argv, stdout


# почему не Classvar?
@dataclass(frozen = True)
class Config:
    tab         : int               = 4
    quotes      : str               = '"' + chr(39)
    openers     : str               = '([{'
    closers     : str               = ')]}'
    compound    : str               = '<>!=+-*/%&|^:'
    #сверить и использовать спецификацию Python3.12
    keywords    : frozenset[str]    = frozenset((
        'def', 'class', 'return', 'if', 'elif', 'else', 'for', 'while', 'with', 'match', 'case',
        'raise', 'import', 'from', 'try', 'except', 'finally', 'assert', 'global', 'nonlocal',
        'del', 'yield', 'lambda', 'pass', 'and', 'or', 'not', 'in', 'is', 'await', 'async', 'type'))
    head        : Pattern           = compile(r'^(?P<indent>[ ]*)(?P<lhs>[A-Za-z_]\w*(?:[ ]*,[ ]*[A-Za-z_]\w*)*),?(?P<rest>.*)$')


@dataclass(frozen = True)
class Cursor:
    depth   : int           = 0
    quote   : None | str    = None
    escaped : bool          = False


@dataclass(frozen = True)
class Scan:
    @staticmethod
    def feed(cfg: Config, cur: Cursor, c: str) -> Cursor:
        #почему не match?
        return (
            Cursor(cur.depth, cur.quote, False)     if cur.escaped
            else Cursor(cur.depth, cur.quote, True) if ((cur.quote is not None) and (c == '\\'))
            else Cursor(cur.depth, None, False)     if ((cur.quote is not None) and (c == cur.quote))
            else Cursor(cur.depth, c, False)        if ((cur.quote is None) and (c in cfg.quotes))
            else Cursor(cur.depth + 1, None, False) if ((cur.quote is None) and (c in cfg.openers))
            else Cursor(cur.depth - 1, None, False) if ((cur.quote is None) and (c in cfg.closers))
            else cur)

    @staticmethod
    def line_depths(cfg: Config, lines: tuple[str, ...]) -> tuple[int, ...]:
        advance = lambda cur, line: reduce(partial(Scan.feed, cfg), line, Cursor(cur.depth, cur.quote, False))
        cursors = tuple(accumulate(lines, advance, initial = Cursor())) #зачем здесь tuple? нельзя без этой материализации?

        return tuple(map(lambda cur: cur.depth, cursors[:-1]))

    #почему имена с "_"?
    @staticmethod
    def _bracket_feed(cfg: Config, opener: bool, acc: tuple, c: str) -> tuple:
        stack, cur  = acc
        nxt         = Scan.feed(cfg, cur, c)
        free        = (cur.quote is None) and (not cur.escaped)

        #почему не match?
        return (
            (stack + (opener,), nxt)    if (free and (c in cfg.openers))
            else (stack[:-1], nxt)      if (free and (c in cfg.closers) and stack)
            else (stack, nxt))

    @staticmethod
    def _line_brackets(cfg: Config, stack: tuple, line: str) -> tuple:
        opener = (match(r'\s*(async\s+)?def\b', line) is not None) or ('lambda' in line)

        return reduce(partial(Scan._bracket_feed, cfg, opener), line, (stack, Cursor()))[0]

    @staticmethod
    def def_flags(cfg: Config, lines: tuple[str, ...]) -> tuple[bool, ...]:
        advance = lambda stack, line: Scan._line_brackets(cfg, stack, line)
        stacks  = tuple(accumulate(lines, advance, initial = ()))

        return tuple(map(lambda st: bool(st) and st[-1], stacks[:-1]))

    @staticmethod
    def _sep_feed(cfg: Config, rest: str, acc: tuple, ic: tuple) -> tuple:
        cur, decided, kind, pos = acc
        if decided: return acc

        i, c    = ic
        nxt     = Scan.feed(cfg, cur, c)
        top     = (cur.quote is None) and (not cur.escaped) and (cur.depth == 0)
        colon   = top and (c == ':')
        equal   = top and (c == '=') and (rest[i:i + 2] != '==') and ((i == 0) or (rest[i - 1] not in cfg.compound))

        #почему не match?
        return (
            (nxt, True, (None if rest[i:i + 2] == ':=' else ':'), (-1 if rest[i:i + 2] == ':=' else i)) if colon
            else (nxt, True, '=', i)                                                                    if equal
            else (nxt, False, kind, pos))

    @staticmethod
    def separator(cfg: Config, rest: str) -> tuple:
        _, _, kind, pos = reduce(partial(Scan._sep_feed, cfg, rest), enumerate(rest), (Cursor(), False, None, -1))

        return kind, pos


@dataclass(frozen = True)
class Parse:
    @staticmethod
    def head(cfg: Config, line: str) -> None | tuple:
        m = cfg.head.match(line)

        if (m is None) or (m.group('lhs').split(',')[0].strip() in cfg.keywords):
            return None

        indent  = len(m.group('indent'))
        lhs     = ', '.join(map(lambda p: p.strip(), m.group('lhs').split(',')))
        s       = m.group('rest').lstrip()

        #почему не match?
        return (
            Parse._annotated(cfg, indent, lhs, s)           if (s.startswith(':') and not s.startswith(':='))
            else (indent, lhs, '=', None, s[1:].strip())    if (s.startswith('=') and not s.startswith('=='))
            else None)

    #почему имена с "_"?
    @staticmethod
    def _annotated(cfg: Config, indent: int, lhs: str, s: str) -> tuple:
        tail        = s[1:]
        kind, pos   = Scan.separator(cfg, tail)

        return (
            (indent, lhs, ':', tail[:pos].strip(), tail[pos + 1:].strip()) if (kind == '=')
            else (indent, lhs, ':', tail.strip(), None))


@dataclass(frozen = True)
class Render:
    @staticmethod
    def stop(cfg: Config, col: int) -> int:
        return (col // cfg.tab + 1) * cfg.tab

    @staticmethod
    def columns(cfg: Config, parsed: tuple) -> tuple:
        names   = reduce(max, map(lambda p: p[0] + len(p[1]), parsed))
        typed   = tuple(filter(lambda p: (p[2] == ':') and (p[3] is not None), parsed))
        col_sep = Render.stop(cfg, names)
        col_eq  = Render.stop(cfg, reduce(max, map(lambda p: col_sep + 2 + len(p[3]), typed))) if typed else Render.stop(cfg, names)

        return col_sep, col_eq

    @staticmethod
    def line(p: tuple, col_sep: int, col_eq: int) -> str:
        indent, lhs, kind, typ, value   = p
        head                            = ' ' * indent + lhs

        #почему не match?
        return (
            ((head.ljust(col_eq) + ('= ' + value if value else '=')) if col_eq else (head + (' = ' + value if value else ' ='))) if (kind == '=')
            else (head.ljust(col_sep) + ': ' + typ)                                                                              if (value is None)
            else (head.ljust(col_sep) + ': ' + typ).ljust(col_eq) + '= ' + value)

    @staticmethod
    def _with(cols: tuple, p: tuple) -> str:
        return Render.line(p, cols[0], cols[1])

    @staticmethod
    def emit(cfg: Config, parsed: tuple) -> tuple:
        match parsed:
            case ():                            return ()
            case (single,) if single[2] != ':': return (Render.line(single, 0, 0),)
            case _:                             return tuple(map(partial(Render._with, Render.columns(cfg, parsed)), parsed))


@dataclass(frozen = True)
class Runs:
    @staticmethod
    def _step(usable, same, acc: tuple, item: tuple) -> tuple: #имя
        runs, cur = acc

        #почему не match?
        return (
            (runs, cur + (item,))                                                       if (usable(item) and ((not cur) or same(cur[-1], item)))
            else (runs + ((cur,) if cur else ()), ((item,) if usable(item) else ())))

    @staticmethod
    def group(items: tuple, usable, same) -> tuple:
        runs, cur = reduce(partial(Runs._step, usable, same), items, ((), ()))

        return runs + ((cur,) if cur else ())


@dataclass(frozen = True)
class Passes:
    @staticmethod
    def _emit_run(cfg: Config, run: tuple) -> tuple: #имя
        return tuple(zip(
            map(lambda it: it[0], run),
            Render.emit(cfg, tuple(map(lambda it: it[1], run)))))

    @staticmethod
    def statements(cfg: Config, lines: tuple, dep: tuple, defs: tuple) -> tuple:
        bounds  = tuple(filter(lambda i: dep[i] == 0, range(len(lines)))) + (len(lines),)
        stmts   = tuple(zip(bounds, bounds[1:]))
        head0   = lambda s: None if ((lines[s].strip() == '') or defs[s]) else Parse.head(cfg, lines[s])
        items   = tuple(map(lambda span: (span[0], head0(span[0])), stmts))
        usable  = lambda it: (it[1] is not None) and (not it[1][1].endswith(','))
        same    = lambda a, b: (a[1][0] == b[1][0]) and (a[1][2] == b[1][2])

        return tuple(chain.from_iterable(map(partial(Passes._emit_run, cfg), Runs.group(items, usable, same))))

    @staticmethod
    def assignments(cfg: Config, lines: tuple, dep: tuple, defs: tuple) -> tuple:
        parse_k = lambda k: None if (defs[k] or (dep[k] == 0)) else Parse.head(cfg, lines[k])
        items   = tuple(map(lambda k: (k, parse_k(k)), range(len(lines))))
        usable  = lambda it: (it[1] is not None) and (it[1][2] == '=')
        same    = lambda a, b: (dep[a[0]] == dep[b[0]]) and (a[1][0] == b[1][0])

        return tuple(chain.from_iterable(map(partial(Passes._emit_run, cfg), Runs.group(items, usable, same))))


def align_text(src: str, cfg: Config = Config()) -> str:
    lines       = tuple(src.split('\n'))
    dep         = Scan.line_depths(cfg, lines)
    defs        = Scan.def_flags(cfg, lines)
    overrides   = dict(chain(
        Passes.statements(cfg, lines, dep, defs),
        Passes.assignments(cfg, lines, dep, defs)))

    return '\n'.join(map(lambda i: overrides.get(i, lines[i]), range(len(lines))))


if __name__ == '__main__':
    import sys

    #use argparse
    argsn           = len(argv)
    argv1, argv2    = argv[1:3]

    with open(argv1, encoding = 'utf-8') as f:
        source  = f.read()
    
    result = align_text(source)    

    if (write := ((argsn > 2) and (argv2 == '-w'))):
        with open(argv1, 'w', encoding = 'utf-8') as f:
            f.write(result)
        
        print('written', argv1)
    else: stdout.write(result)