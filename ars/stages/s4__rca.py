import  json
import  math
import  re

from    datetime                import datetime, timezone
from    typing                  import Tuple

import  polars                  as pl

from    ars.data.features       import feature_provenance
from    ars.data.stages_meta    import S1Meta, S2Meta, S3Meta
from    ars.specification.spec  import DataObject


DETECTOR_RCA_SCHEMA = 'laim.detector_rca/1'
EXCERPT_CHARS       = 200

# span columns worth showing to an analyst, in output order (absent ones are skipped)
SPAN_DETAIL_COLUMNS = (
    'span_name', 'aef_kind', 'status_code', 'status_message', 'parent_span_id',
    'start_time_ns', 'end_time_ns', 'llm_model', 'llm_total_tokens',
    'http_method', 'http_path', 'http_status_code', 'kafka_topic',
    'meta_langgraph_node', 'service_name', 'input_text', 'output_text')
_RENAMED    = {'span_name': 'name', 'aef_kind': 'kind', 'status_code': 'status',
               'meta_langgraph_node': 'graph_node', 'service_name': 'service'}
# data-contract sentinels mean "absent" — they carry nothing for an analyst
_SENTINELS  = frozenset({'', 'NONE', -1, -1.0})
# aggregations of a log1p feature that still name a value in natural units
_LOCATION   = re.compile(r'(?:rolling_)?(?:max|min|mean|q\d+)')


def analyze_anomalies(data: pl.LazyFrame, meta: Tuple[S1Meta, S2Meta, S3Meta]) -> pl.LazyFrame:
    '''RCA seam (gap M11): formats the detector's attribution surface into a
    human-readable per-trace summary. The actual root-cause analysis (LMA per
    the LumiMAS paper) plugs in here; until then the report names the worst
    spans and the EPI features driving the reconstruction error.'''
    s1_meta, _s2_meta, _s3_meta = meta
    cols = data.collect_schema().names()
    if 'rca_top_span_indices' not in cols:
        return data.with_columns(rca_report_str = pl.lit(''))

    feature_names = list(s1_meta.epi_features)

    def _summarize(row: dict) -> str:
        spans = ', '.join(
            f'span#{i} (err={e:.4f})'
            for i, e in zip(row['rca_top_span_indices'], row['rca_top_span_errors']))
        feats = ', '.join(
            f'{feature_names[i] if i < len(feature_names) else f"f{i}"} (err={e:.4f})'
            for i, e in zip(row['rca_top_feature_indices'], row['rca_top_feature_errors']))
        return f'worst spans: {spans}; driving features: {feats}'

    return data.with_columns(
        pl.struct('rca_top_span_indices', 'rca_top_span_errors',
                  'rca_top_feature_indices', 'rca_top_feature_errors')
          .map_elements(_summarize, return_dtype = pl.String)
          .alias('rca_report_str'))


def _num(value: None | float, digits: int = 6) -> None | float:
    if value is None:
        return None
    value = float(value)
    return float(f'{value:.{digits}g}') if math.isfinite(value) else None


def _excerpt(text: object, limit: int) -> None | str:
    if not isinstance(text, str):
        return None
    flat = ' '.join(text.split())
    if not flat:
        return None
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + '…'


def _iso(ns: object) -> None | str:
    try:
        if int(ns) < 0:
            return None
        return datetime.fromtimestamp(int(ns) / 1e9, tz = timezone.utc).isoformat(timespec = 'milliseconds').replace('+00:00', 'Z')
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _scalar(value: object) -> None | str | int | float | bool:
    '''JSON-safe span attribute; None for sentinels and unrepresentable values'''
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        value = _num(value)
    elif not isinstance(value, (str, int)):
        return None
    return None if value is None or value in _SENTINELS else value


def _span_details(row: dict, excerpt_chars: int) -> dict:
    '''one span as an analyst reads it: identity, timing, status, excerpts'''
    details = {}
    for col in SPAN_DETAIL_COLUMNS:
        if col in ('start_time_ns', 'end_time_ns', 'input_text', 'output_text'):
            continue
        if (value := _scalar(row.get(col))) is not None:
            details[_RENAMED.get(col, col)] = value
    start, end = row.get('start_time_ns'), row.get('end_time_ns')
    if (started := _iso(start)) is not None:
        details['start_time'] = started
    try:
        if int(start) >= 0 and int(end) >= int(start):
            details['duration_s'] = _num((int(end) - int(start)) / 1e9)
    except (TypeError, ValueError):
        pass
    for col, key in (('input_text', 'input_excerpt'), ('output_text', 'output_excerpt')):
        if (text := _excerpt(row.get(col), excerpt_chars)) is not None:
            details[key] = text
    return details


def _natural(value: float, log1p: bool, aggregation: None | str) -> None | float:
    '''the value in natural units (ns, chars, counts), when one exists'''
    if not log1p:
        return _num(value)
    if aggregation is not None and not _LOCATION.fullmatch(aggregation):
        return None     # std/sum of log1p values have no natural-unit reading
    magnitude = abs(value)
    return _num(math.copysign(math.expm1(magnitude), value)) if magnitude < 700 else None


def _feature_name(index: int, names: Tuple[str, ...]) -> str:
    return names[index] if index < len(names) else f'f{index}'


def _feature_view(index: int, obs: None | float, exp: None | float, names: Tuple[str, ...], norm: dict) -> dict:
    '''a feature at one span: observed vs reconstructed ("expected") value in
    the feature's own scale and, when defined, in natural units (ns, chars...);
    provenance lives once per feature in detector_rca.feature_info'''
    name    = _feature_name(index, names)
    view    = {'feature': name}
    if obs is None or exp is None:
        return view
    prov    = feature_provenance(name)
    view['direction'] = 'higher' if obs > exp else 'lower' if obs < exp else 'as_expected'
    shift, scale = norm.get('shift'), norm.get('scale')
    if shift is not None and scale is not None and index < len(shift):
        observed, expected = obs * scale[index] + shift[index], exp * scale[index] + shift[index]
        view |= {'observed': _num(observed, 4), 'expected': _num(expected, 4)}
        if (observed_raw := _natural(observed, prov['log1p'], prov['aggregation'])) is not None:
            view |= {'observed_raw': _num(observed_raw, 4),
                     'expected_raw': _num(_natural(expected, prov['log1p'], prov['aggregation']), 4)}
    z_clip = float(norm.get('z_clip') or 0.0)
    if z_clip > 0 and abs(obs) >= z_clip - 1e-6:
        view['clipped'] = True      # observed sits at the normalization clip: true value is further out
    return view


def _detector_rca(row: dict, attribution: dict, spans: dict, names: Tuple[str, ...], norm: dict,
                  excerpt_chars: int) -> dict:
    trace_id    = str(row[DataObject.trace_id])
    scores      = attribution.get('scores', {})
    logit       = scores.get('logit', {})
    share       = scores.get('comb_share_epi')

    def span_ref(entry: dict, rank: int) -> dict:
        return {'rank': rank, 'index': entry['i'], 'span_id': entry.get('id'),
                'error': _num(entry.get('err'), 4), 'error_share': _num(entry.get('share'), 4),
                'vs_typical': _num(entry.get('vs_typical'), 4)}

    behavior_spans = [
        {**span_ref(entry, rank),
         'drivers': [{**_feature_view(d['f'], d.get('obs'), d.get('exp'), names, norm), 'error': _num(d.get('err'), 4)}
                     for d in entry.get('drivers', [])]}
        for rank, entry in enumerate(attribution.get('epi_spans', []), start = 1)]
    semantic_spans = [span_ref(entry, rank) for rank, entry in enumerate(attribution.get('sem_spans', []), start = 1)]
    features = [
        {'rank': rank, **_feature_view(f['f'], f.get('obs'), f.get('exp'), names, norm),
         'error': _num(f.get('err'), 4), 'error_share': _num(f.get('share'), 4),
         'vs_typical': _num(f.get('vs_typical'), 4),
         'peak_span_index': f.get('peak'), 'peak_span_id': f.get('peak_id')}
        for rank, f in enumerate(attribution.get('epi_features', []), start = 1)]

    # every referenced span and feature once: the lists above point into these
    referenced = dict.fromkeys(ref for ref in (
        *(s['span_id'] for s in behavior_spans), *(s['span_id'] for s in semantic_spans),
        *(f['peak_span_id'] for f in features)) if ref is not None)
    catalog = {ref: _span_details(spans[(trace_id, ref)], excerpt_chars)
               for ref in referenced if (trace_id, ref) in spans}
    feature_info = {name: feature_provenance(name) for name in dict.fromkeys(
        (*(f['feature'] for f in features), *(d['feature'] for s in behavior_spans for d in s['drivers'])))}

    return {
        'schema':   DETECTOR_RCA_SCHEMA,
        'agent_id': row.get(DataObject.agent_id),
        'scores': {
            'p_anomaly':            scores.get('p_anomaly'),
            'reconstruction_error': scores.get('e_comb'),
            'threshold':            scores.get('threshold'),
            'branch_z':             {'behavior': scores.get('z_epi'), 'semantic': scores.get('z_sem')},
            'flag_share':           {'behavior': share, 'semantic': _num(1.0 - share) if share is not None else None},
            'logit':                {'behavior': logit.get('epi'), 'semantic': logit.get('sem'),
                                     'combined': logit.get('comb'), 'bias': logit.get('bias')}},
        'sequence': {'spans': attribution.get('n_spans'), 'scored': attribution.get('n_scored'),
                     'truncated': attribution.get('truncated')},
        'spans':        catalog,
        'feature_info': feature_info,
        'behavior':     {'spans': behavior_spans, 'features': features},
        'semantic':     {'spans': semantic_spans}}


def export_detector_rca(scored: pl.DataFrame, spans: None | pl.LazyFrame, s1_meta: S1Meta,
                        excerpt_chars: int = EXCERPT_CHARS) -> pl.DataFrame:
    '''RCA-экспорт детектора (M11 → продукт): для каждой ФЛАГНУТОЙ трассы
    колонка detector_rca — самоописывающий JSON (схема DETECTOR_RCA_SCHEMA):
    agent_id; оценки (p_anomaly, порог, z ветвей, доли ветвей во флагующей
    ошибке, разложение логита); поведенческие (EPI) спаны с деталями спана и
    признаками-драйверами; EPI-признаки с наблюдаемым/ожидаемым значением
    (шкала признака и натуральные единицы); смысловые (SEM) спаны. Индексы
    переводятся в имена признаков (s1_meta) и детали спанов (spans).
    Нефлагнутые строки и строки без rca_attribution получают null.'''
    if 'rca_attribution' not in scored.columns:
        return scored
    flagged     = (scored.get_column('detector_is_anomaly').fill_null(False)
                   if 'detector_is_anomaly' in scored.columns else pl.Series([True] * scored.height))
    keys        = [c for c in (DataObject.trace_id, DataObject.agent_id) if c in scored.columns]
    rows        = scored.select(keys).to_dicts()
    parsed      = [json.loads(a) if (flag and a) else None
                   for flag, a in zip(flagged.to_list(), scored.get_column('rca_attribution').to_list())]

    wanted = {(str(row[DataObject.trace_id]), ref)
              for row, attribution in zip(rows, parsed) if attribution is not None
              for ref in (*(e.get('id') for e in (*attribution.get('epi_spans', []), *attribution.get('sem_spans', []))),
                          *(f.get('peak_id') for f in attribution.get('epi_features', [])))
              if ref is not None}
    span_rows: dict = {}
    if wanted and spans is not None:
        present = spans.collect_schema().names()
        columns = [c for c in SPAN_DETAIL_COLUMNS if c in present]
        found = (spans
            .select(pl.col(DataObject.trace_id).cast(pl.String), pl.col(DataObject.span_id).cast(pl.String), *columns)
            .join(pl.LazyFrame({DataObject.trace_id: [t for t, _ in wanted], DataObject.span_id: [s for _, s in wanted]},
                               schema = {DataObject.trace_id: pl.String, DataObject.span_id: pl.String}),
                  on = [DataObject.trace_id, DataObject.span_id], how = 'semi')
            .unique([DataObject.trace_id, DataObject.span_id], keep = 'first', maintain_order = True)
            .collect())
        span_rows = {(r[DataObject.trace_id], r[DataObject.span_id]): r for r in found.to_dicts()}

    names   = tuple(s1_meta.epi_features)
    norm    = dict(s1_meta.epi_normalization or {})
    exports = [json.dumps(_detector_rca(row, attribution, span_rows, names, norm, excerpt_chars),
                          ensure_ascii = False, allow_nan = False)
               if attribution is not None else None
               for row, attribution in zip(rows, parsed)]
    return scored.with_columns(pl.Series('detector_rca', exports, dtype = pl.String))
