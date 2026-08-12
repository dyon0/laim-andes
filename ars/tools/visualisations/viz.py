from    typing                          import Any, ClassVar, Mapping, cast
from    dataclasses                     import dataclass
from    functools                       import reduce
from    itertools                       import chain, starmap

from    pathlib                         import PurePath, Path
from    datetime                        import datetime
from    base64                          import b64encode
from    importlib.util                  import find_spec

import  polars                          as pl

import  jax                             as jx

import  altair                          as al

# F-30: altair's default 5000-row cap raised MaxRowsError mid-run on any test
# set with >5k spans, killing training AFTER the models were fitted
al.data_transformers.disable_max_rows()

from    ars.configuration.c2__detector  import S2Config


type Frame = pl.DataFrame
type Chart = Any


@dataclass(frozen = True)
class VizTheme:
    normal          : str               = '#4C72B0'
    anomaly         : str               = '#C44E52'
    accent          : str               = '#55A868'
    background      : str               = '#FFFFFF'
    grid            : str               = '#E6E8EC'
    text            : str               = '#1F2A37'
    muted           : str               = '#6B7280'
    palette         : tuple[str, ...]   = ('#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3')
    width           : int               = 640
    height          : int               = 360
    square          : int               = 380
    facet_width     : int               = 130
    area_opacity    : float             = 0.45
    heat_scheme     : str               = 'blues'
    diverging       : str               = 'blueorange'
    stroke          : float             = 2.0
    stroke_bold     : float             = 2.5
    rule_stroke     : float             = 1.5
    interpolate     : str               = 'monotone'
    corner          : int               = 3
    corner_small    : int               = 2
    text_size       : int               = 18
    text_weight     : int               = 700
    label_size      : int               = 11
    title_size      : int               = 12
    head_size       : int               = 15
    head_weight     : int               = 600
    title_dy        : int               = -4
    tilt            : int               = -40
    dash            : tuple[int, int]   = (6, 4)
    dash_ref        : tuple[int, int]   = (4, 4)
    dash_train      : tuple[int, int]   = (1, 0)
    dash_val        : tuple[int, int]   = (6, 4)
    fmt             : str               = '.3f'
    fmt_fine        : str               = '.5f'
    fmt_rate        : str               = '.4f'
    fmt_corr        : str               = '.2f'


@dataclass(frozen = True)
class Html:
    style   : ClassVar[str] = (
        '*{box-sizing:border-box}'
        'body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;padding:40px 48px;background:#F5F6F8;color:#1F2A37;line-height:1.5}'
        'h1{font-size:28px;font-weight:700;margin:0 0 2px;letter-spacing:-.02em}'
        'h2{font-size:18px;font-weight:600;margin:34px 0 14px;color:#374151}'
        '.sub{color:#6B7280;font-size:14px;margin:0 0 8px}'
        '.cards{display:flex;flex-wrap:wrap;gap:14px;margin:8px 0 4px}'
        '.card{flex:1 1 150px;background:#FFFFFF;border:1px solid #E5E7EB;border-radius:14px;padding:15px 18px;box-shadow:0 1px 2px rgba(16,24,40,.04)}'
        '.card .k{color:#6B7280;font-size:12px;text-transform:uppercase;letter-spacing:.04em}'
        '.card .v{color:#111827;font-size:22px;font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums}'
        '.tw{overflow-x:auto;border-radius:12px;box-shadow:0 1px 2px rgba(16,24,40,.04);margin:6px 0 8px;-webkit-overflow-scrolling:touch}'
        'table{border-collapse:collapse;width:100%;min-width:max-content;background:#FFFFFF}'
        'th,td{padding:10px 14px;text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}'
        'th{background:#F9FAFB;color:#6B7280;font-size:12px;text-transform:uppercase;letter-spacing:.03em;border-bottom:1px solid #E5E7EB}'
        'td{border-bottom:1px solid #F3F4F6;font-size:14px}'
        'th:first-child,td:first-child{text-align:left}'
        'tr:last-child td{border-bottom:none}'
        'img{max-width:100%;border-radius:12px;border:1px solid #E5E7EB;background:#FFFFFF}'
        '.fig{margin:8px 0 20px}')

    @staticmethod
    def doc(title: str, subtitle: str, body: str) -> str:
        return (
            f'<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{title}</title><style>{Html.style}</style></head><body>'
            f'<h1>{title}</h1><p class="sub">{subtitle}</p>{body}</body></html>')

    @staticmethod
    def section(title: str, content: str) -> str:
        return f'<h2>{title}</h2>{content}' if content else ''

    @staticmethod
    def image(title: str, b64: str) -> str:
        return f'<div class="fig"><img alt="{title}" src="data:image/png;base64,{b64}"></div>' if b64 else ''

    @staticmethod
    def cards(items: tuple[tuple[str, str], ...]) -> str:
        card = lambda kv: f'<div class="card"><div class="k">{kv[0]}</div><div class="v">{kv[1]}</div></div>'

        return f'<div class="cards">{str().join(map(card, items))}</div>' if items else ''

    @staticmethod
    def frame_table(frame: Frame, float_fmt: str = '.4f') -> str:
        shown   = lambda v: f'{v:{float_fmt}}' if isinstance(v, float) else str(v)
        head    = '<tr>' + str().join(map(lambda c: f'<th>{c}</th>', frame.columns)) + '</tr>'
        line    = lambda row: '<tr>' + str().join(map(lambda c: f'<td>{shown(row[c])}</td>', frame.columns)) + '</tr>'

        return f'<div class="tw"><table>{head}{str().join(map(line, frame.to_dicts()))}</table></div>'


@dataclass(frozen = True)
class Frames:
    @staticmethod
    def as_list(values: Any) -> list:
        match values:
            case jx.Array():    return values.tolist()
            case pl.Series():   return values.to_list()
            case _:             return list(values)

    @staticmethod
    def losses(history: dict[str, dict[str, tuple[float, ...]]]) -> Frame:
        one     = lambda branch, split, values: pl.DataFrame({
            'epoch'     : range(1, len(values) + 1),
            'loss'      : map(float, values),
            'branch'    : branch,
            'split'     : split})
        nested  = starmap(
            lambda branch, splits: starmap(lambda split, values: one(branch, split, values), splits.items()),
            history.items())
        present = tuple(filter(lambda f: f.height > 0, chain.from_iterable(nested)))

        return (pl.concat(present)
                if present
                else pl.DataFrame(schema = {'epoch': pl.Int64, 'loss': pl.Float64, 'branch': pl.Utf8, 'split': pl.Utf8}))

    @staticmethod
    def errors(values: Any, labels: Any) -> Frame:
        return pl.DataFrame({
            'error' : map(float, Frames.as_list(values)),
            'label' : map(lambda y: 'anomaly' if y else 'normal', Frames.as_list(labels))})

    @staticmethod
    def metric_bars(results: tuple[dict[str, Any], ...], metrics: tuple[str, ...] = ('f1', 'recall', 'precision')) -> Frame:
        row = lambda res: map(
            lambda m: {'experiment': res['experiment'], 'metric': m, 'score': float(res.get('test_metrics', {}).get(m, 0.0))},
            metrics)

        return pl.DataFrame(chain.from_iterable(map(row, results)))

    @staticmethod
    def confusion(tp: int, fp: int, fn: int, tn: int) -> Frame:
        return pl.DataFrame({
            'actual'    : ('anomaly', 'anomaly', 'normal', 'normal'),
            'predicted' : ('anomaly', 'normal', 'anomaly', 'normal'),
            'count'     : (tp, fn, fp, tn)})

    @staticmethod
    def curve(x_name: str, y_name: str, xs: Any, ys: Any) -> Frame:
        return pl.DataFrame({x_name: map(float, Frames.as_list(xs)), y_name: map(float, Frames.as_list(ys))})

    @staticmethod
    def sweep(thresholds: Any, metric_scores: dict[str, Any]) -> Frame:
        grid    = tuple(map(float, Frames.as_list(thresholds)))
        block   = lambda name, scores: pl.DataFrame({'threshold': grid, 'metric': name, 'score': map(float, Frames.as_list(scores))})

        return pl.concat(starmap(block, metric_scores.items()))

    @staticmethod
    def value_counts(values: Any, top: None | int = None) -> Frame:
        counted = pl.Series('category', Frames.as_list(values)).value_counts(sort = True)

        return counted.head(top) if top is not None else counted

    @staticmethod
    def mapping(items: dict[str, float]) -> Frame:
        return pl.DataFrame({'category': items.keys(), 'value': map(float, items.values())})

    @staticmethod
    def missingness(frame: Frame) -> Frame:
        height = max(frame.height, 1)

        return (frame
                    .null_count()
                    .unpivot(variable_name = 'column', value_name = 'nulls')
                    .with_columns((pl.col('nulls') / height * 100).alias('pct'))
                    .filter(pl.col('nulls') > 0)
                    .sort('pct', descending = True))

    @staticmethod
    def correlation(matrix: Frame, labels: tuple[str, ...]) -> Frame:
        return (matrix
            .with_columns(pl.Series('row', labels))
            .unpivot(index = 'row', variable_name = 'col', value_name = 'corr'))

    @staticmethod
    def to_long(frame: Frame, index: str, on: tuple[str, ...]) -> Frame:
        return frame.unpivot(index = index, on = on, variable_name = 'metric', value_name = 'value')

    @staticmethod
    def samples(groups: dict[str, Any]) -> Frame:
        block = lambda name, values: pl.DataFrame({'value': map(float, Frames.as_list(values)), 'group': name})

        return pl.concat(starmap(block, groups.items()))

    @staticmethod
    def confusion_counts(labels: Any, preds: Any) -> tuple[int, int, int, int]:
        pair    = pl.DataFrame({'y': map(int, Frames.as_list(labels)), 'p': map(int, Frames.as_list(preds))})
        counts  = pair.select(
            ((pl.col('p') == 1) & (pl.col('y') == 1)).sum().alias('tp'),
            ((pl.col('p') == 1) & (pl.col('y') == 0)).sum().alias('fp'),
            ((pl.col('p') == 0) & (pl.col('y') == 1)).sum().alias('fn'),
            ((pl.col('p') == 0) & (pl.col('y') == 0)).sum().alias('tn')).row(0)

        tp, fp, fn, tn = map(int, counts)

        return tp, fp, fn, tn

    @staticmethod
    def auc(fpr: Any, tpr: Any) -> float:
        ordered = pl.DataFrame({'fpr': map(float, Frames.as_list(fpr)), 'tpr': map(float, Frames.as_list(tpr))}).sort('fpr')
        n       = ordered.height
        dx      = ordered.get_column('fpr').diff().slice(1)
        ys      = ordered.get_column('tpr')

        return float((dx * (ys.slice(1) + ys.slice(0, n - 1)) / 2.0).sum()) if n > 1 else 0.0


@dataclass(frozen = True)
class Charts:
    @staticmethod
    def themed(theme: VizTheme, chart: Chart) -> Chart:
        return (chart
                    .configure_view(stroke = None, fill = theme.background)
                    .configure_axis(
                        gridColor       = theme.grid,
                        domainColor     = theme.grid,
                        tickColor       = theme.grid,
                        labelColor      = theme.muted,
                        titleColor      = theme.text,
                        labelFontSize   = theme.label_size,
                        titleFontSize   = theme.title_size)
                    .configure_title(
                        color       = theme.text,
                        fontSize    = theme.head_size,
                        fontWeight  = theme.head_weight,
                        anchor      = 'start',
                        dy          = theme.title_dy)
                    .configure_legend(
                        labelColor      = theme.text,
                        titleColor      = theme.muted,
                        labelFontSize   = theme.label_size,
                        titleFontSize   = theme.label_size)
                    .configure_header(
                        labelColor  = theme.muted,
                        titleColor  = theme.text))

    @staticmethod
    def loss_curves(theme: VizTheme, frame: Frame, title: str = 'Кривые обучения') -> Chart:
        return (al.Chart(frame).mark_line(strokeWidth = theme.stroke, interpolate = cast(Any, theme.interpolate))
            .encode(
                x           = al.X('epoch:Q', title = 'Эпоха'),
                y           = al.Y('loss:Q', title = 'Loss', scale = al.Scale(zero = False)),
                color       = al.Color('branch:N', title = 'Ветвь', scale = al.Scale(range = theme.palette)),
                strokeDash  = al.StrokeDash('split:N', title = 'Выборка', scale = al.Scale(domain = ('train', 'val'), range = (theme.dash_train, theme.dash_val))),
                tooltip     = ('branch:N', 'split:N', al.Tooltip('epoch:Q'), al.Tooltip('loss:Q', format = theme.fmt_fine)))
            .properties(title = title, width = theme.width, height = theme.height)
            .interactive())

    @staticmethod
    def error_density(theme: VizTheme, frame: Frame, threshold: None | float = None, title: str = 'Распределение ошибки реконструкции') -> Chart:
        area    = (al.Chart(frame)
            .transform_density('error', groupby = ['label'], as_ = ['error', 'density'])
            .mark_area(opacity = theme.area_opacity, line = {'strokeWidth': theme.stroke})
            .encode(
                x       = al.X('error:Q', title = 'Ошибка реконструкции'),
                y       = al.Y('density:Q', title = 'Плотность'),
                color   = al.Color('label:N', title = None, scale = al.Scale(domain = ('normal', 'anomaly'), range = (theme.normal, theme.anomaly)))))
        ruled   = (al.Chart(pl.DataFrame({'threshold': (float(threshold or 0.0),)}))
            .mark_rule(color = theme.text, strokeDash = theme.dash, strokeWidth = theme.rule_stroke)
            .encode(x = 'threshold:Q', tooltip = al.Tooltip('threshold:Q', title = 'Порог', format = theme.fmt_fine)))
        layered = al.layer(area, ruled) if threshold is not None else area

        return layered.properties(title = title, width = theme.width, height = theme.height)

    @staticmethod
    def roc(theme: VizTheme, frame: Frame, auc: None | float = None, title: str = 'ROC-кривая') -> Chart:
        diag    = (al.Chart(pl.DataFrame({'fpr': (0.0, 1.0), 'tpr': (0.0, 1.0)}))
            .mark_line(color = theme.grid, strokeDash = theme.dash_ref).encode(x = 'fpr:Q', y = 'tpr:Q'))
        curve   = (al.Chart(frame).mark_line(strokeWidth = theme.stroke_bold, color = theme.normal)
            .encode(
                x       = al.X('fpr:Q', title = 'FPR', scale = al.Scale(domain = (0, 1))),
                y       = al.Y('tpr:Q', title = 'TPR', scale = al.Scale(domain = (0, 1))),
                tooltip = (al.Tooltip('fpr:Q', format = theme.fmt), al.Tooltip('tpr:Q', format = theme.fmt))))

        return al.layer(diag, curve).properties(
            title = title if auc is None else f'{title}  ·  AUC = {auc:{theme.fmt}}', width = theme.square, height = theme.square)

    @staticmethod
    def confusion(theme: VizTheme, frame: Frame, title: str = 'Матрица ошибок') -> Chart:
        base    = al.Chart(frame).encode(
            x   = al.X('predicted:N', title = 'Предсказано', sort = ('anomaly', 'normal')),
            y   = al.Y('actual:N',    title = 'Истина',      sort = ('anomaly', 'normal')))
        cells   = base.mark_rect().encode(color = al.Color('count:Q', title = None, scale = al.Scale(scheme = cast(Any, theme.heat_scheme))))
        labels  = base.mark_text(fontSize = theme.text_size, fontWeight  = cast(Any, theme.text_weight)).encode(
            text    = al.Text('count:Q'),
            color   = al.condition('datum.count > 0', al.value(theme.background), al.value(theme.muted)))

        return al.layer(cells, labels).properties(title = title, width = theme.square, height = theme.square)

    @staticmethod
    def metric_bars(theme: VizTheme, frame: Frame, title: str = 'Метрики по экспериментам') -> Chart:
        return (al.Chart(frame).mark_bar(cornerRadiusEnd = theme.corner)
            .encode(
                x       = al.X('metric:N', title = None, axis = al.Axis(labelAngle = 0)),
                y       = al.Y('score:Q', title = 'Значение', scale = al.Scale(domain = (0, 1))),
                color   = al.Color('metric:N', title = 'Метрика', scale = al.Scale(range = theme.palette)),
                column  = al.Column('experiment:N', title = 'Эксперимент'),
                tooltip = ('experiment:N', 'metric:N', al.Tooltip('score:Q', format = theme.fmt)))
            .properties(title = title, width = theme.facet_width, height = theme.height))

    @staticmethod
    def threshold_sweep(theme: VizTheme, frame: Frame, chosen: None | float = None, title: str = 'Метрики и порог') -> Chart:
        lines   = (al.Chart(frame).mark_line(strokeWidth = theme.stroke, interpolate = cast(Any, theme.interpolate))
            .encode(
                x       = al.X('threshold:Q', title = 'Порог'),
                y       = al.Y('score:Q', title = 'Значение', scale = al.Scale(domain = (0, 1))),
                color   = al.Color('metric:N', title = 'Метрика', scale = al.Scale(range = theme.palette)),
                tooltip = ('metric:N', al.Tooltip('threshold:Q', format = theme.fmt_rate), al.Tooltip('score:Q', format = theme.fmt))))
        ruled   = (al.Chart(pl.DataFrame({'threshold': (float(chosen or 0.0),)}))
            .mark_rule(color = theme.text, strokeDash = theme.dash, strokeWidth = theme.rule_stroke).encode(x = 'threshold:Q'))
        layered = al.layer(lines, ruled) if chosen is not None else lines

        return layered.properties(title = title, width = theme.width, height = theme.height)

    @staticmethod
    def dashboard(charts: tuple[Chart, ...], columns: int = 2) -> Chart:
        chunks  = map(lambda i: charts[i:i + columns], range(0, len(charts), columns))
        rows    = map(lambda chunk: reduce(lambda acc, c: acc | c, chunk), chunks)

        return reduce(lambda acc, r: acc & r, rows)

    @staticmethod
    def bars(theme: VizTheme, frame: Frame, category: str, value: str, title: str, baseline: None | float = None, descending: bool = True) -> Chart:
        bars    = (al.Chart(frame).mark_bar(cornerRadiusEnd = theme.corner, color = theme.normal)
            .encode(
                x       = al.X(f'{category}:N', title = None, sort = '-y' if descending else None, axis = al.Axis(labelAngle = theme.tilt)),
                y       = al.Y(f'{value}:Q', title = value),
                tooltip = (f'{category}:N', al.Tooltip(f'{value}:Q', format = theme.fmt_rate))))
        ruled   = (al.Chart(pl.DataFrame({'y': (float(baseline or 0.0),)}))
            .mark_rule(color = theme.anomaly, strokeDash = theme.dash).encode(y = 'y:Q'))
        layered = al.layer(bars, ruled) if baseline is not None else bars

        return layered.properties(title = title, width = theme.width, height = theme.height)

    @staticmethod
    def grouped_bars(theme: VizTheme, frame: Frame, category: str, value: str, group: str, title: str) -> Chart:
        return (al.Chart(frame).mark_bar(cornerRadiusEnd = theme.corner_small)
            .encode(
                x       = al.X(f'{category}:N', title = None, axis = al.Axis(labelAngle = theme.tilt)),
                y       = al.Y(f'{value}:Q', title = value),
                xOffset = f'{group}:N',
                color   = al.Color(f'{group}:N', title = group, scale = al.Scale(range = theme.palette)),
                tooltip = (f'{category}:N', f'{group}:N', al.Tooltip(f'{value}:Q', format = theme.fmt_rate)))
            .properties(title = title, width = theme.width, height = theme.height))

    @staticmethod
    def distribution(theme: VizTheme, frame: Frame, title: str = 'Распределение', value: str = 'value', group: str = 'group', marker: None | float = None) -> Chart:
        area    = (al.Chart(frame)
            .transform_density(value, groupby = [group], as_ = [value, 'density'])
            .mark_area(opacity = theme.area_opacity, line = {'strokeWidth': theme.stroke})
            .encode(
                x       = al.X(f'{value}:Q', title = value),
                y       = al.Y('density:Q', title = 'Плотность'),
                color   = al.Color(f'{group}:N', title = None, scale = al.Scale(range = theme.palette))))
        ruled   = (al.Chart(pl.DataFrame({'m': (float(marker or 0.0),)}))
            .mark_rule(color = theme.text, strokeDash = theme.dash, strokeWidth = theme.rule_stroke).encode(x = 'm:Q'))
        layered = al.layer(area, ruled) if marker is not None else area

        return layered.properties(title = title, width = theme.width, height = theme.height)

    @staticmethod
    def correlation_heatmap(theme: VizTheme, frame: Frame, title: str = 'Корреляции признаков') -> Chart:
        return (al.Chart(frame).mark_rect()
            .encode(
                x       = al.X('col:N', title = None, axis = al.Axis(labelAngle = theme.tilt)),
                y       = al.Y('row:N', title = None),
                color   = al.Color('corr:Q', title = 'r', scale = al.Scale(scheme = cast(Any, theme.diverging), domain = (-1, 1))),
                tooltip = ('row:N', 'col:N', al.Tooltip('corr:Q', format = theme.fmt_corr)))
            .properties(title = title, width = theme.height, height = theme.height))

    @staticmethod
    def reliability(theme: VizTheme, frame: Frame, title: str = 'Калибровка (reliability)') -> Chart:
        diag    = (al.Chart(pl.DataFrame({'pred': (0.0, 1.0), 'actual': (0.0, 1.0)}))
            .mark_line(color = theme.grid, strokeDash = theme.dash_ref).encode(x = 'pred:Q', y = 'actual:Q'))
        line    = (al.Chart(frame).mark_line(strokeWidth = theme.stroke, point = True, color = theme.normal)
            .encode(
                x       = al.X('pred:Q', title = 'Предсказанная вероятность', scale = al.Scale(domain = (0, 1))),
                y       = al.Y('actual:Q', title = 'Эмпирическая доля', scale = al.Scale(domain = (0, 1))),
                size    = al.Size('n:Q', title = 'N'),
                tooltip = (al.Tooltip('n:Q'), al.Tooltip('pred:Q', format = theme.fmt), al.Tooltip('actual:Q', format = theme.fmt))))

        return al.layer(diag, line).properties(title = title, width = theme.square, height = theme.square)

    @staticmethod
    def pr(theme: VizTheme, frame: Frame, title: str = 'Precision-Recall') -> Chart:
        return (al.Chart(frame).mark_line(strokeWidth = theme.stroke_bold, color = theme.accent)
            .encode(
                x       = al.X('recall:Q', title = 'Recall', scale = al.Scale(domain = (0, 1))),
                y       = al.Y('precision:Q', title = 'Precision', scale = al.Scale(domain = (0, 1))),
                tooltip = (al.Tooltip('recall:Q', format = theme.fmt), al.Tooltip('precision:Q', format = theme.fmt)))
            .properties(title = title, width = theme.square, height = theme.square))


def ensure(path: PurePath) -> PurePath:
    Path(path).parent.mkdir(parents = True, exist_ok = True)

    return path


def img_to_base64(path: None | PurePath) -> str:
    return b64encode(p.read_bytes()).decode('utf-8') if (path and (p := Path(path)).exists()) else ''


def save_chart(chart: Chart, path: PurePath) -> None | PurePath:
    target = path.with_suffix('.png')
    if find_spec('vl_convert') is None: return None

    # F-30: a render failure (fonts, memory, vega) must degrade to a missing
    # image in the report, never abort the pipeline that already trained models
    try:
        chart.save(ensure(target).as_posix(), scale_factor = 2.0)
    except Exception as render_error:
        print(f'viz: не удалось отрисовать {target.name}: {render_error!r}')
        return None

    return target


def save_loss_plot(
        exp_name        : str,
        epi_losses      : tuple[float, ...],
        sem_losses      : tuple[float, ...],
        combined_losses : tuple[float, ...],
        output_dir      : PurePath,
        theme           : VizTheme = VizTheme(),
) -> None | PurePath:
    history = {
        'EPI'       : {'train': epi_losses},
        'SEM'       : {'train': sem_losses},
        'Combined'  : {'train': combined_losses}}
    frame   = Frames.losses(history)

    return None if frame.height == 0 else save_chart(
        Charts.themed(theme, Charts.loss_curves(theme, frame, f'Кривые обучения — {exp_name}')),
        output_dir / f'{exp_name}_losses')


def save_training(
        exp_name    : str,
        history     : dict[str, dict[str, tuple[float, ...]]],
        output_dir  : PurePath,
        theme       : VizTheme = VizTheme()
) -> None | PurePath:
    frame = Frames.losses(history)

    return None if frame.height == 0 else save_chart(
        Charts.themed(theme, Charts.loss_curves(theme, frame, f'Кривые обучения — {exp_name}')),
        output_dir / f'{exp_name}_training')


def save_error_distribution(
        errors      : Any,
        labels      : Any,
        title       : str,
        output_dir  : PurePath,
        name        : str,
        threshold   : None | float  = None,
        theme       : VizTheme      = VizTheme(),
) -> None | PurePath:
    frame = Frames.errors(errors, labels)
    frame.write_json(ensure(output_dir / f'{name}.json').as_posix())

    return save_chart(Charts.themed(theme, Charts.error_density(theme, frame, threshold, title)), output_dir / name)


def save_roc(
        fpr         : Any,
        tpr         : Any,
        output_dir  : PurePath,
        name        : str           = 'roc',
        auc         : None | float  = None,
        theme       : VizTheme      = VizTheme()
) -> None | PurePath:
    frame   = Frames.curve('fpr', 'tpr', fpr, tpr).sort('fpr')
    area    = auc if auc is not None else Frames.auc(fpr, tpr)

    return save_chart(Charts.themed(theme, Charts.roc(theme, frame, area)), output_dir / name)


def save_confusion(tp: int, fp: int, fn: int, tn: int, output_dir: PurePath, name: str = 'confusion', theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.confusion(theme, Frames.confusion(tp, fp, fn, tn))), output_dir / name)


def save_threshold_sweep(thresholds: Any, metric_scores: dict[str, Any], output_dir: PurePath, name: str = 'threshold_sweep', chosen: None | float = None, theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.threshold_sweep(theme, Frames.sweep(thresholds, metric_scores), chosen)), output_dir / name)


def save_metrics_bar(all_results: tuple[dict[str, Any], ...], output_dir: PurePath, theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.metric_bars(theme, Frames.metric_bars(all_results))), output_dir / 'metrics_bar')


def generate_experiment_report(
        experiment_dir  : Path,
        experiment_name : str,
        exp_config      : dict[str, Any],
        results         : dict[str, Any],
        viz_paths       : None | Mapping[str, None | PurePath],
) -> Path:
    train_keys  = (
        'epi_final_train_mse', 'epi_final_val_mse',
        'sem_final_train_mse', 'sem_final_val_mse',
        'combined_final_train_mse', 'combined_final_val_mse')
    shown       = lambda v: f'{v:.4f}' if isinstance(v, float) else str(v)
    paths       = viz_paths or {}

    config_cards    = Html.cards(tuple(map(lambda kv: (str(kv[0]), shown(kv[1])), exp_config.items())))
    train_cards     = Html.cards(tuple(map(lambda k: (k.replace('_', ' '), shown(results.get(k, 'N/A'))), train_keys)))
    metric_cards    = Html.cards(tuple(starmap(lambda k, v: (k, f'{float(v):.4f}'), results.get('test_metrics', {}).items())))
    figures         = ''.join(map(
        lambda kv: Html.image(kv[0], img_to_base64(kv[1])),
        (('Кривые обучения',          paths.get('loss')),
         ('Распределение ошибок',     paths.get('error')),
         ('ROC-кривая',               paths.get('roc')),
         ('Метрики и порог',          paths.get('sweep')),
         ('Матрица ошибок',           paths.get('confusion')),
         ('Калибровка',               paths.get('reliability')))))

    body    = (
          Html.section('Конфигурация',          config_cards)
        + Html.section('Результаты обучения',   train_cards)
        + Html.section('Графики',               figures)
        + Html.section('Метрики на тесте',      metric_cards))
    report  = Path(experiment_dir) / f'report_{experiment_name}.html'
    Path(report).parent.mkdir(parents = True, exist_ok = True)
    report.write_text(
        Html.doc(f'Отчёт — {experiment_name}', f'{datetime.now():%Y-%m-%d %H:%M:%S}', body),
        encoding = 'utf-8')

    return report


def generate_summary_report(cfg: S2Config, all_results: tuple[dict[str, Any], ...], theme: VizTheme = VizTheme()) -> Path:
    scheme          = cfg.output_color_scheme
    output_dir      = Path(cfg.output_dir)
    report          = output_dir / 'summary_report.html'
    metric_names    = ('accuracy', 'precision', 'recall', 'f1', 'youden')

    metric_row  = lambda r: {
        'experiment'        : r['experiment'],
        'best_threshold'    : float(r.get('best_threshold', 0.0)),
        **dict(map(lambda m: (m, float(r.get('test_metrics', {}).get(m, 0.0))), metric_names)),
        'total_time'        : float(r.get('time_epi', 0.0) + r.get('time_sem', 0.0) + r.get('time_combined', 0.0))}
    config_row  = lambda r: {'experiment': r['experiment'], **dict(map(lambda kv: (str(kv[0]), kv[1]), r.get('config', {}).items()))}

    metrics_frame   = pl.DataFrame(map(metric_row, all_results))
    config_frame    = pl.DataFrame(map(config_row, all_results))
    bar_path        = save_metrics_bar(all_results, output_dir / 'visualizations', theme)

    body = (
          Html.section('Сводка метрик',                 Html.frame_table(metrics_frame))
        + Html.section('Сравнение метрик',              Html.image('F1 / Recall / Precision', img_to_base64(bar_path)))
        + Html.section('Конфигурации экспериментов',    Html.frame_table(config_frame)))
    
    Path(report).parent.mkdir(parents = True, exist_ok = True)
    report.write_text(
        Html.doc('Общий отчёт экспериментов', f'Метрика выбора: {cfg.select_metric}  ·  {datetime.now():%Y-%m-%d %H:%M:%S}', body),
        encoding = 'utf-8')
    
    scheme.print_metric('Сводный отчёт сохранён', report.as_posix())

    return report


def save_value_counts(values: Any, output_dir: PurePath, name: str, title: str = 'Распределение', top: None | int = None, baseline: None | float = None, theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.bars(theme, Frames.value_counts(values, top), 'category', 'count', title, baseline)), output_dir / name)


def save_bars(frame: Frame, category: str, value: str, output_dir: PurePath, name: str, title: str = '', baseline: None | float = None, theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.bars(theme, frame, category, value, title or value, baseline)), output_dir / name)


def save_grouped_bars(frame: Frame, category: str, value: str, group: str, output_dir: PurePath, name: str, title: str = '', theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.grouped_bars(theme, frame, category, value, group, title or value)), output_dir / name)


def save_distribution(groups: dict[str, Any], output_dir: PurePath, name: str, title: str = 'Распределение', marker: None | float = None, theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.distribution(theme, Frames.samples(groups), title, 'value', 'group', marker)), output_dir / name)


def save_correlation(matrix: Frame, labels: tuple[str, ...], output_dir: PurePath, name: str = 'correlation', title: str = 'Корреляции признаков', theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.correlation_heatmap(theme, Frames.correlation(matrix, labels), title)), output_dir / name)


def save_missingness(frame: Frame, output_dir: PurePath, name: str = 'missingness', title: str = 'Пропуски, %', theme: VizTheme = VizTheme()) -> None | PurePath:
    miss = Frames.missingness(frame)

    return None if miss.height == 0 else save_chart(Charts.themed(theme, Charts.bars(theme, miss, 'column', 'pct', title)), output_dir / name)


def save_reliability(frame: Frame, output_dir: PurePath, name: str = 'reliability', title: str = 'Калибровка (reliability)', theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.reliability(theme, frame, title)), output_dir / name)


def save_pr(recall: Any, precision: Any, output_dir: PurePath, name: str = 'pr', theme: VizTheme = VizTheme()) -> None | PurePath:
    frame = pl.DataFrame({'recall': map(float, Frames.as_list(recall)), 'precision': map(float, Frames.as_list(precision))})

    return save_chart(Charts.themed(theme, Charts.pr(theme, frame)), output_dir / name)


def save_confusion_preds(labels: Any, preds: Any, output_dir: PurePath, name: str = 'confusion', theme: VizTheme = VizTheme()) -> None | PurePath:
    tp, fp, fn, tn = Frames.confusion_counts(labels, preds)

    return save_confusion(tp, fp, fn, tn, output_dir, name, theme)


def save_mapping_bars(items: dict[str, float], output_dir: PurePath, name: str, title: str = '', baseline: None | float = None, theme: VizTheme = VizTheme()) -> None | PurePath:
    return save_chart(Charts.themed(theme, Charts.bars(theme, Frames.mapping(items), 'category', 'value', title, baseline, descending = False)), output_dir / name)


def block_table(title: str, frame: Frame) -> str:
    return Html.section(title, Html.frame_table(frame))


def block_cards(title: str, items: tuple[tuple[str, str], ...]) -> str:
    return Html.section(title, Html.cards(items))


def block_image(title: str, path: None | PurePath) -> str:
    return Html.section(title, Html.image(title, img_to_base64(path)))


def write_report(path: PurePath, title: str, subtitle: str, blocks: tuple[str, ...]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents = True, exist_ok = True)
    target.write_text(Html.doc(title, subtitle, str().join(blocks)), encoding = 'utf-8')

    return target