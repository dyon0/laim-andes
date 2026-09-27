"""RCA export (M11 -> product): feature provenance, the s4 detector_rca block,
span-id alignment in s1, and the test_anomalies records contract."""
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from ars.data.features import FeatureDefinition, FeaturesSpan, feature_provenance
from ars.stages.s4__rca import DETECTOR_RCA_SCHEMA, export_detector_rca

REPO = Path(__file__).resolve().parents[1]


def _strict(text: str) -> dict:
    """json.loads that refuses NaN/Infinity — the export must be valid JSON."""
    def refuse(token):
        raise ValueError(f'non-finite JSON number: {token}')
    return json.loads(text, parse_constant=refuse)


@pytest.mark.parametrize('name,expected', [
    ('duration', ('duration', None, None, True, 'step')),
    ('duration_diff', ('duration_diff', None, None, True, 'step')),
    ('duration_diff_rolling_q75_w5', ('duration_diff', 'rolling_q75', 5, True, 'step')),
    ('duration_max', ('duration', 'max', None, True, 'sequence')),
    ('exec_out_char_count_q25', ('exec_out_char_count', 'q25', None, True, 'sequence')),
    ('tool_success_rolling_sum_w1000', ('tool_success', 'rolling_sum', 1000, False, 'step')),
    ('final_output_length', ('final_output_length', None, None, True, 'sequence')),   # over(trace_id)
    ('not_a_feature', ('not_a_feature', None, None, False, 'step')),
])
def test_feature_provenance(name, expected):
    got = feature_provenance(name)
    assert (got['base'], got['aggregation'], got['window'], got['log1p'], got['scope']) == expected


# ------------------------------------------------------------- s4 export

NAMES = ('duration', 'char_count_std', 'tool_success')
NORM = {'shift': [20.0, 1.0, 0.0], 'scale': [2.0, 0.5, 1.0], 'z_clip': 20.0}


def _attribution(**overrides) -> str:
    base = {
        'version': 1, 'n_spans': 9, 'n_scored': 8, 'truncated': True,
        'scores': {'p_anomaly': 0.97, 'e_comb': 0.8, 'threshold': 0.5, 'e_epi': 2.0, 'e_sem': 0.01,
                   'z_epi': 12.0, 'z_sem': 0.3, 'comb_share_epi': 0.75,
                   'logit': {'epi': 3.0, 'sem': -0.3, 'comb': 1.0, 'bias': 0.1}},
        'epi_spans': [{'i': 1, 'id': 'sp-b', 'err': 3.5, 'share': 0.6, 'vs_typical': 12.5,
                       'drivers': [{'f': 0, 'err': 3.0, 'obs': 1.5, 'exp': -0.5},
                                   {'f': 1, 'err': 0.5, 'obs': -20.0, 'exp': 0.1}]}],
        'epi_features': [{'f': 0, 'err': 1.2, 'share': 0.5, 'vs_typical': 8.0, 'peak': 1, 'peak_id': 'sp-b',
                          'obs': 1.5, 'exp': -0.5},
                         {'f': 2, 'err': 0.4, 'share': 0.2, 'vs_typical': 2.0, 'peak': 0, 'peak_id': 'sp-a',
                          'obs': -1.0, 'exp': 0.0}],
        'sem_spans': [{'i': 0, 'id': 'sp-a', 'err': 0.02, 'share': 0.4, 'vs_typical': 1.1},
                      {'i': 2, 'id': 'sp-missing', 'err': 0.01, 'share': 0.2, 'vs_typical': 1.0}],
    }
    return json.dumps(base | overrides)


def _spans() -> pl.LazyFrame:
    return pl.LazyFrame({
        'trace_id': ['t1', 't1', 't1', 't2'],
        'span_id': ['sp-a', 'sp-b', 'sp-c', 'sp-b'],
        'span_name': ['plan', 'get_rate', 'answer', 'other-trace'],
        'aef_kind': ['chain', 'tool', 'llm', 'tool'],
        'status_code': ['STATUS_CODE_OK', 'STATUS_CODE_ERROR', 'STATUS_CODE_OK', 'STATUS_CODE_OK'],
        'status_message': ['', 'upstream timeout', '', ''],
        'start_time_ns': [1_779_268_740_000_000_000, 1_779_268_741_000_000_000, 1_779_268_760_000_000_000, 0],
        'end_time_ns': [1_779_268_740_500_000_000, 1_779_268_753_300_000_000, 1_779_268_761_000_000_000, 1],
        'llm_total_tokens': [-1, -1, 812, -1],
        'input_text': ['  план\n\nшагов ', '{"product": "кредит"}', 'x' * 500, ''],
        'output_text': ['', 'timeout', 'Ставка 22,5%', ''],
    })


def _export(scored: pl.DataFrame, spans=None) -> pl.DataFrame:
    meta = SimpleNamespace(epi_features=NAMES, epi_normalization=NORM)
    return export_detector_rca(scored, _spans() if spans is None else spans, meta)


def test_export_resolves_spans_features_and_scores():
    scored = pl.DataFrame({'trace_id': ['t1', 't2'], 'agent_id': ['agent-7', 'agent-9'],
                           'detector_is_anomaly': [True, False],
                           'rca_attribution': [_attribution(), _attribution()]})
    out = _export(scored)

    assert out['detector_rca'][1] is None                             # not flagged -> no block
    rca = _strict(out['detector_rca'][0])
    assert rca['schema'] == DETECTOR_RCA_SCHEMA and rca['agent_id'] == 'agent-7'
    assert rca['scores']['branch_z'] == {'behavior': 12.0, 'semantic': 0.3}
    assert rca['scores']['flag_share'] == {'behavior': 0.75, 'semantic': 0.25}
    assert rca['scores']['logit'] == {'behavior': 3.0, 'semantic': -0.3, 'combined': 1.0, 'bias': 0.1}
    assert rca['sequence'] == {'spans': 9, 'scored': 8, 'truncated': True}

    # span catalog: resolved within the SAME trace, sentinels dropped, excerpts cut
    assert set(rca['spans']) == {'sp-a', 'sp-b'}                      # sp-missing unknown, sp-c unreferenced
    tool = rca['spans']['sp-b']
    assert tool == {'name': 'get_rate', 'kind': 'tool', 'status': 'STATUS_CODE_ERROR',
                    'status_message': 'upstream timeout', 'start_time': '2026-05-20T09:19:01.000Z',
                    'duration_s': 12.3, 'input_excerpt': '{"product": "кредит"}', 'output_excerpt': 'timeout'}
    assert rca['spans']['sp-a']['input_excerpt'] == 'план шагов'          # whitespace collapsed
    assert 'llm_total_tokens' not in rca['spans']['sp-a']                 # -1 sentinel

    span = rca['behavior']['spans'][0]
    assert (span['rank'], span['index'], span['span_id'], span['vs_typical']) == (1, 1, 'sp-b', 12.5)
    duration, spread = span['drivers']
    # observed = z * scale + shift; log1p features also in natural units
    assert duration['feature'] == 'duration' and duration['direction'] == 'higher'
    assert (duration['observed'], duration['expected']) == (23.0, 19.0)
    assert duration['observed_raw'] == pytest.approx(math.expm1(23.0), rel=1e-3)
    assert duration['expected_raw'] == pytest.approx(math.expm1(19.0), rel=1e-3)
    # std of a log1p feature has no natural-unit reading; |z| at the clip is flagged
    assert 'observed_raw' not in spread and spread['clipped'] is True and spread['direction'] == 'lower'

    features = rca['behavior']['features']
    assert [f['feature'] for f in features] == ['duration', 'tool_success']
    assert features[1]['observed_raw'] == features[1]['observed'] == -1.0   # not log1p: raw == feature scale
    assert features[0]['peak_span_id'] == 'sp-b'
    assert rca['feature_info']['char_count_std'] == {'base': 'char_count', 'aggregation': 'std', 'window': None,
                                                     'log1p': True, 'scope': 'sequence'}
    assert [s['span_id'] for s in rca['semantic']['spans']] == ['sp-a', 'sp-missing']


def test_export_without_spans_or_attribution_degrades_gracefully():
    scored = pl.DataFrame({'trace_id': ['t1'], 'agent_id': ['a'], 'detector_is_anomaly': [True],
                           'rca_attribution': [_attribution()]})
    rca = _strict(export_detector_rca(scored, None, SimpleNamespace(epi_features=NAMES, epi_normalization=NORM))
                  ['detector_rca'][0])
    assert rca['spans'] == {} and rca['behavior']['spans'][0]['span_id'] == 'sp-b'
    plain = pl.DataFrame({'trace_id': ['t1'], 'detector_is_anomaly': [True]})
    assert _export(plain).equals(plain)                                   # attribution off: untouched


def test_export_json_never_carries_non_finite_numbers():
    attribution = _attribution(scores={'p_anomaly': float('nan'), 'comb_share_epi': None})
    attribution = attribution.replace('NaN', 'null')          # s2 writes JSON with allow_nan=False
    scored = pl.DataFrame({'trace_id': ['t1'], 'agent_id': ['a'], 'detector_is_anomaly': [True],
                           'rca_attribution': [attribution]})
    rca = _strict(_export(scored)['detector_rca'][0])
    assert rca['scores']['p_anomaly'] is None and rca['scores']['flag_share'] == {'behavior': None, 'semantic': None}


def test_build_traces_carries_span_ids_in_sequence_order(tmp_path):
    from ars.configuration.c1__data import S1Config
    from ars.stages.s1__data import build_traces
    cfg = S1Config(input_parquet_files=(), output_dir=tmp_path, output_prefix='ids')
    spans = pl.DataFrame({
        'trace_id': ['t1', 't1', 't1', 't2'], 'agent_id': ['a', 'a', 'b', 'a'],
        'span_id': ['s3', 's1', 's2', 's9'], 'start_time_ns': [30, 10, 20, 5],
        'epi_vector': [[3.0], [1.0], [2.0], [9.0]],
        'is_anomaly': pl.Series([None] * 4, dtype=pl.Int8), 'anomaly_type': ['unknown'] * 4,
    }).sort('agent_id', 'trace_id', 'start_time_ns')
    traces = build_traces(spans, (), cfg, carry_span_ids=True).sort('trace_id', 'agent_id')
    rows = {(r['trace_id'], r['agent_id']): r for r in traces.to_dicts()}
    assert rows[('t1', 'a')]['span_ids'] == ['s1', 's3']
    assert [v[0] for v in rows[('t1', 'a')]['epi_sequence']] == [1.0, 3.0]  # index i <-> span_ids[i]
    assert 'span_ids' not in build_traces(spans, (), cfg).columns          # training path unchanged


def test_records_carry_detector_rca_only_when_present():
    from ars.main import Anomalies
    base = {'trace_id': ['t1'], 'starttime': ['s'], 'endtime': ['e'], 'confidence': [97]}
    legacy = Anomalies.records(pl.DataFrame(base))[0]
    assert 'detector_rca' not in legacy and legacy['rca_results'] == ''
    rich = Anomalies.records(pl.DataFrame(base | {'detector_rca': ['{"schema": "laim.detector_rca/1"}']}))[0]
    assert rich['detector_rca'] == {'schema': 'laim.detector_rca/1'}
    assert list(rich)[:11] == list(legacy)                                  # legacy fields first, unchanged


def test_rca_node_glossary_covers_every_detector_feature():
    """Contract guard: a new laim feature must get a description in the RCA
    node's glossary, or the node explains it only by its raw name."""
    path = REPO / 'rca' / 'laim_rca' / 'glossary.py'
    if not path.exists():
        pytest.skip('RCA node not in this checkout')
    spec = importlib.util.spec_from_file_location('rca_glossary', path)
    glossary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(glossary)
    numeric = {fd.name for fd in vars(FeaturesSpan()).values() if isinstance(fd, FeatureDefinition)} \
        - {'sem_text', 'agent_prompt'}                                      # text carriers, never EPI features
    assert numeric - set(glossary.FEATURES) == set()


def test_product_confidence_is_the_anomaly_probability():
    """detector_confidence is max(p, 1-p): a flagged trace with p_anomaly 0.3
    used to be reported with confidence 70. The product field is p_anomaly."""
    from ars.main import Anomalies
    detected = pl.DataFrame({'trace_id': ['t1', 't2'], 'detector_p_anomaly': [0.3, 0.914],
                             'detector_confidence': [0.7, 0.914]})
    bounds = pl.DataFrame({'trace_id': ['t1', 't2'], '_t0': [0, 0], '_t1': [1, 1]})
    assert Anomalies.enrich(detected, bounds)['confidence'].to_list() == [30, 91]
