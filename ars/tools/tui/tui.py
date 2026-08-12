from    typing                              import Any, Callable, ClassVar
from    dataclasses                         import dataclass, fields, replace
from    functools                           import reduce
from    itertools                           import starmap

from    pathlib                             import PurePath, Path
from    re                                  import compile as re_compile
from    shutil                              import get_terminal_size
from    os                                  import (
    close, dup, dup2, open as os_open, O_WRONLY, O_CREAT, O_APPEND, environ)

import  sys

import  polars                              as pl

from    tqdm                                import tqdm

from    ars.tools.abstraction.composition   import composable
from    ars.tools.tui.tui_core              import AnsiCodes, ColorScheme


def sprint(*args, style_code: str = '', sep: str = ' ', end: str = '\n', file = None, flush: bool = False) -> None:
    body = sep.join(map(str, args))

    print(f'{style_code}{body}{AnsiCodes.reset}' if style_code else body, sep = '', end = end, file = file, flush = flush)


def styler(code: str) -> Callable[[str], str]:
    @composable
    def apply(text: str) -> str:
        return f'{code}{text}{AnsiCodes.reset}' if code else text

    return apply


@dataclass(frozen = True)
class ColorSchemeSystem(ColorScheme):
    subheader   : str   = ''
    progress    : str   = ''

    def print_success(self, text: str) -> None:
        sprint(text, style_code = self.success)

    def print_error(self, text: str) -> None:
        sprint(text, style_code = self.error)

    def print_warning(self, text: str) -> None:
        sprint(text, style_code = self.warning)

    def print_info(self, text: str) -> None:
        sprint(text, style_code = self.info)

    def print_debug(self, text: str) -> None:
        sprint(text, style_code = self.debug)

    def print_header(self, text: str) -> None:
        sprint(text, style_code = self.header)

    def print_prompt(self, text: str) -> None:
        sprint(text, style_code = self.prompt)


@dataclass(frozen = True)
class ColorSchemeDataScience(ColorSchemeSystem):
    table_border        : str   = ''
    table_header        : str   = ''
    table_cell          : str   = ''
    table_highlight     : str   = ''
    best_metric         : str   = ''
    perf_metric         : str   = ''
    good_metric         : str   = ''
    bad_metric          : str   = ''
    progress_bar        : str   = ''
    progress_desc       : str   = ''
    progress_color_hex  : str   = ''

    def print_section(self, title: str) -> None:
        sprint(Layout.section(self, title), style_code = self.header)

    def print_subsection(self, title: str) -> None:
        sprint(Layout.subsection(self, title), style_code = self.subheader)

    def print_metric(self, label: str, value: Any, unit: str = '', pad: int = 30) -> tuple[str, str]:
        text = f'{value} {unit}'.strip()
        sprint(f'\t{label:<{pad}}\t: {text}', style_code = self.info)

        return label, str(value)

    def print_inline_tuple(self, title: str, items: tuple) -> tuple[str, str]:
        sprint(f'{title}:', style_code = self.subheader)
        sprint('\t' + ', '.join(map(str, items)), style_code = self.info)

        return title, str(items)

    def print_stats_table(self, title: str, df: pl.DataFrame) -> tuple[str, str]:
        sprint(f'\n{self.subheader}{title}:', style_code = self.subheader)
        print_table(df, cols = ('Параметр', 'Значение'), scheme = self)

        return title, str(df.to_dicts())

    def print_missings_table(self, title: str, df: pl.DataFrame) -> tuple[str, str]:
        sprint(f'\n{self.subheader}{title}', style_code = self.subheader)
        print_table(df, cols = ('Признак', 'Пропусков', 'Процент', 'Заполнено'), float_fmt = '.2f', scheme = self)

        return title, str(df.to_dicts())

    def print_performance_metrics(
            self,
            label       : str,
            telapsed    : float,
            cpu_peak    : None | float = None,
            gpu_peak    : None | float = None,
            ram_pos_l   : float = 0.50,
            time_pos_r  : float = 0.99,
            ram_str_max : int = 25,
            time_prec   : int = 5,
    ) -> None:
        sprint(
            Layout.performance_line(label, telapsed, cpu_peak, gpu_peak, ram_pos_l, time_pos_r, ram_str_max, time_prec),
            style_code = self.perf_metric)

    def tqdm_kwargs(self) -> dict[str, Any]:
        if not self.progress_bar and not self.progress_color_hex:
            return {'colour': None}

        return {
            'colour'     : self.progress_color_hex,
            'bar_format' : f'{self.progress_bar}{{l_bar}}{self.progress_bar}{{bar}}{self.progress_bar}{{r_bar}}{AnsiCodes.reset}'}


@dataclass(frozen = True)
class Progress:
    @staticmethod
    def bar(iterable = None, *, total = None, desc = None, unit = 'it', file = sys.__stdout__, **extra) -> tqdm:
        match environ.get('ARS_PROGRESS', 'on').strip().lower():
            case 'off' | 'false' | '0' | 'no':
                return tqdm(iterable, total = total, desc = desc, unit = unit, file = file, disable = True)
            case _:
                every    = float(environ.get('ARS_PROGRESS_EVERY', '0') or '0')
                miniters = max(1, int(total * every)) if (every > 0 and total) else None

                return tqdm(
                    iterable, total = total, desc = desc, unit = unit, file = file,
                    mininterval = (0.0 if miniters else 0.1), miniters = miniters, **extra)


@dataclass(frozen = True)
class Layout:
    @staticmethod
    def width(default: int = 100) -> int:
        return get_terminal_size((default, 24)).columns

    @staticmethod
    def section(scheme: ColorSchemeSystem, title: str) -> str:
        pad = Layout.width() - len(title) - 10

        return f'\n{scheme.header}\t{title}{' ' * max(pad, 0)}'

    @staticmethod
    def subsection(scheme: ColorSchemeSystem, title: str) -> str:
        pad = Layout.width() - len(title)

        return f'\n{scheme.subheader}{title}{' ' * max(pad, 0)}'

    @staticmethod
    def fit(text: str, width: int, cut: str = '...') -> str:
        return (text[:width - len(cut)] + cut) if (len(text) > width and width >= len(cut)) else text[:width]

    @staticmethod
    def performance_line(
            label       : str,
            elapsed     : float,
            cpu_peak    : None | float,
            gpu_peak    : None | float,
            ram_pos_l   : float,
            time_pos_r  : float,
            ram_str_max : int,
            time_prec   : int,
    ) -> str:
        total           = Layout.width()
        r_pos           = max(1, min(int(total * ram_pos_l), total - 1))
        t_pos           = max(r_pos + 2, min(int(total * time_pos_r), total))

        ram             = 'пик RAM '  + (f'{cpu_peak:.2f} MB' if cpu_peak is not None else 'недоступно')
        vram            = 'пик VRAM ' + (f'{gpu_peak:.2f} MB' if gpu_peak is not None else 'недоступно')
        elapsed_text    = f'{elapsed:.{time_prec}f} сек'

        head            = Layout.fit(f'>> {label}', r_pos).ljust(r_pos)
        free            = max(r_pos, t_pos - len(elapsed_text)) - r_pos
        block           = Layout.fit(f'{ram.ljust(ram_str_max)} {vram}', free) if free > 0 else ''

        return (head + block + ' ' * (free - len(block)) + elapsed_text).ljust(total)

    @staticmethod
    def best(indexed: pl.DataFrame, col: str, direction: str) -> dict[str, Any]:
        cast    = pl.col(col).cast(pl.Float64)
        clean   = indexed.select(cast.drop_nans().drop_nulls()).to_series()
        value   = getattr(clean, direction)()
        rows    = indexed.filter(cast == value).select('_row_idx').to_series().to_list()

        return {'indices': frozenset(rows), 'value': value}


def format_table(
        df              : pl.DataFrame,
        cols            : None | tuple[str, ...]                        = None,
        highlight_cols  : None | str | tuple[str, ...] | dict[str, str] = None,
        float_fmt       : str = '.4f',
        col_styles      : None | dict[str, str] = None,
        scheme          : ColorSchemeDataScience = ColorSchemeDataScience(),
) -> str:
    columns = cols or tuple(df.columns)

    match highlight_cols:
        case str():     marked = {highlight_cols: 'max'}
        case tuple():   marked = dict(map(lambda c: (c, 'max'), highlight_cols))
        case dict():    marked = highlight_cols
        case _:         marked = {}

    indexed = df.with_row_index('_row_idx')
    records = indexed.to_dicts()
    if not records: return ''

    shown   = lambda v: f'{v:{float_fmt}}' if isinstance(v, float) else str(v)
    body    = tuple(map(lambda r: dict(map(lambda c: (c, shown(r.get(c, ''))), columns)), records))
    widths  = dict(map(lambda c: (c, reduce(max, map(lambda r: len(r[c]), body), len(c))), columns))
    best    = dict(map(
        lambda c: (c, Layout.best(indexed, c, marked[c])),
        filter(lambda c: c in columns and c in df.columns, marked.keys())))

    numeric = lambda c: isinstance(records[0].get(c), (int, float))
    style   = lambda i, c: (
        scheme.best_metric  if (c in best and i in best[c]['indices'])
        else col_styles[c]  if (col_styles and c in col_styles)
        else scheme.table_cell)
    sized   = lambda i, c: body[i][c].rjust(widths[c]) if numeric(c) else body[i][c].ljust(widths[c])
    cell    = lambda i, c: f'{style(i, c)}{sized(i, c)}{AnsiCodes.reset}{scheme.table_border}'
    head    = lambda c: f'{scheme.table_header}{c.center(widths[c])}{scheme.table_border}'
    line    = lambda i: '| ' + ' | '.join(map(lambda c: cell(i, c), columns)) + ' |'

    border  = scheme.table_border + '+'  + '+'.join(map(lambda c: '-' * (widths[c] + 2), columns)) + '+' + AnsiCodes.reset
    heading = scheme.table_border + '| ' + ' | '.join(map(head, columns)) + ' |' + AnsiCodes.reset
    data    = scheme.table_border + '\n'.join(map(line, range(len(body)))) + AnsiCodes.reset

    return '\n'.join((border, heading, border, data, border))


def print_table(
        df              : pl.DataFrame,
        cols            : None | tuple[str, ...]                        = None,
        highlight_cols  : None | str | tuple[str, ...] | dict[str, str] = None,
        float_fmt       : str = '.4f',
        col_styles      : None | dict[str, str] = None,
        scheme          : ColorSchemeDataScience = ColorSchemeDataScience(),
) -> None:
    print(format_table(df, cols, highlight_cols, float_fmt, col_styles, scheme))


def neutrilizeCS(scheme: ColorScheme) -> ColorScheme:
    return replace(scheme, **dict(map(lambda f: (f.name, ''), filter(lambda f: f.type == str, fields(scheme)))))


@dataclass(frozen = True)
class Summary:
    metric_cols : ClassVar[tuple[str, ...]] = (
        'Experiment', 'Accuracy', 'Precision', 'Recall', 'Specificity',
        'F1', 'Youden', 'EPI train MSE', 'EPI val MSE', 'Threshold', 'Time (s)')
    config_cols : ClassVar[tuple[str, ...]] = (
        'Experiment', 'LR', 'Batch', 'EPI Latent', 'SEM Latent',
        'EPI Decoder', 'SEM Decoder', 'EPI Loss', 'SEM Loss', 'Combined Loss')
    highlight   : ClassVar[dict[str, str]]  = {
        'Accuracy'      : 'max',
        'Precision'     : 'max',
        'Recall'        : 'max',
        'Specificity'   : 'max',
        'F1'            : 'max',
        'Youden'        : 'max',
        'EPI train MSE' : 'min',
        'EPI val MSE'   : 'min'}

    @staticmethod
    def metrics_row(res: dict[str, Any]) -> dict[str, Any]:
        metrics = res['test_metrics']
        spent   = res.get('time_epi', 0.0) + res.get('time_sem', 0.0) + res.get('time_combined', 0.0)

        return {
            'Experiment'    : res['experiment'],
            'Accuracy'      : float(metrics['accuracy']),
            'Precision'     : float(metrics['precision']),
            'Recall'        : float(metrics['recall']),
            'Specificity'   : float(metrics['specificity']),
            'F1'            : float(metrics['f1']),
            'Youden'        : float(metrics['youden']),
            'EPI train MSE' : float(res.get('epi_train_mse', 0.0)),
            'EPI val MSE'   : float(res.get('epi_val_mse',   0.0)),
            'Threshold'     : float(res['best_threshold']),
            'Time (s)'      : float(spent)}

    @staticmethod
    def config_row(res: dict[str, Any]) -> dict[str, Any]:
        conf = res.get('config', {})

        return {
            'Experiment'    : res['experiment'],
            'LR'            : conf.get('learning_rate', ''),
            'Batch'         : conf.get('batch_size',    ''),
            'EPI Latent'    : conf.get('epi_sz_latent', ''),
            'SEM Latent'    : conf.get('sem_sz_latent', ''),
            'EPI Decoder'   : conf.get('epi_decoder',   ''),
            'SEM Decoder'   : conf.get('sem_decoder',   ''),
            'EPI Loss'      : conf.get('epi_loss',      ''),
            'SEM Loss'      : conf.get('sem_loss',      ''),
            'Combined Loss' : conf.get('combined_loss', '')}


def print_summary_table(scheme: ColorSchemeDataScience, all_results: tuple[dict[str, Any], ...]) -> None:
    if not all_results: return

    scheme.print_section('ИТОГОВАЯ СВОДКА ПО ЭКСПЕРИМЕНТАМ')
    print_table(
        pl.DataFrame(map(Summary.metrics_row, all_results)),
        cols            = Summary.metric_cols,
        highlight_cols  = Summary.highlight,
        float_fmt       = '.4f',
        scheme          = scheme)


def print_config_table(scheme: ColorSchemeDataScience, all_results: tuple[dict[str, Any], ...]) -> None:
    if not all_results: return

    scheme.print_section('КОНФИГУРАЦИИ ЭКСПЕРИМЕНТОВ')
    print_table(
        pl.DataFrame(map(Summary.config_row, all_results)),
        cols        = Summary.config_cols,
        float_fmt   = '.4f',
        scheme      = scheme)


def print_best_summary(scheme: ColorSchemeDataScience, best: dict[str, Any]) -> None:
    scheme.print_section('ЛУЧШАЯ МОДЕЛЬ')
    sprint(f'Эксперимент: {best['experiment']}', style_code = scheme.success)

    _ = tuple(starmap(
        lambda k, v: sprint(f'  {k}: {float(v):.4f}', style_code = scheme.success),
        best['test_metrics'].items()))


class _StdoutHook:
    def __init__(self, on_line: Callable[[str], None]):
        self._on_line   = on_line
        self._buf       = ''

    def write(self, s: str) -> int:
        parts = (self._buf + s).split('\n')
        tuple(map(self._on_line, parts[:-1]))
        self._buf       = parts[-1]

        return len(s)

    def flush(self) -> None:
        pass

    def close_remaining(self) -> None:
        if self._buf:
            self._on_line(self._buf)
            self._buf = ''

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return sys.__stdout__.fileno() if sys.__stdout__ is not None else 1


class capture_epoch_progress:
    pattern : ClassVar  = re_compile(
        r'Epoch\s+(\d+),\s*(?:Combined\s+)?Loss:\s*([\d.eE+\-]+),\s*Val Loss:\s*([\d.eE+\-]+)')

    def __init__(self, bar):
        self._bar   = bar
        self._saved : Any                = None
        self._hook  : None | _StdoutHook = None

    def __enter__(self):
        self._saved = sys.stdout
        self._hook  = _StdoutHook(self.handle)
        sys.stdout  = self._hook

        return self

    def __exit__(self, *exc):
        sys.stdout = self._saved
        if self._hook is not None:
            self._hook.close_remaining()

    def handle(self, line: str) -> None:
        found = self.pattern.search(line)

        if found is not None:
            self._bar.set_postfix(loss = float(found.group(2)), val = float(found.group(3)))
            self._bar.update(1)
        else:
            if ('Stopping' in line) or ('reached' in line): self._bar.write(line)

class redirect_native_stderr:
    def __init__(self, path: PurePath):
        self._path  = path
        self._saved : None | int = None

    def __enter__(self):
        Path(self._path).parent.mkdir(parents = True, exist_ok = True)
        sys.stdout.flush()
        sys.stderr.flush()

        self._saved = dup(2)
        log_fd      = os_open(self._path, O_WRONLY | O_CREAT | O_APPEND, 0o644)
        dup2(log_fd, 2)
        close(log_fd)

        return self

    def __exit__(self, *_):
        sys.stderr.flush()
        if self._saved is not None:
            dup2(self._saved, 2)
            close(self._saved)