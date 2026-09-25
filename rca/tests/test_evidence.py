"""Сигнал детектора -> доказательства: разбор detector_rca, гипотеза, фразы, ограничения."""
import json

import pytest

from conftest import detector_rca, load_fixture
from laim_rca import evidence, glossary


def parse(**kwargs) -> evidence.Evidence:
    ev = evidence.parse({'detector_rca': detector_rca(**kwargs)})
    assert ev is not None
    return ev


@pytest.mark.parametrize('record', [
    {}, {'detector_rca': None}, {'detector_rca': '{broken'}, {'detector_rca': {'schema': 'other/1'}},
    {'detector_rca': []},
])
def test_no_or_foreign_signal_gives_no_evidence(record):
    assert evidence.parse(record) is None


def test_signal_as_json_string_is_accepted():
    ev = evidence.parse({'detector_rca': json.dumps(detector_rca())})
    assert ev is not None and ev.agent_id == 'agent-1'


def test_default_signal_reads_as_tool_latency():
    ev = parse()
    assert (ev.dominant, ev.strength, ev.category) == ('behavior', 'strong', 'latency')
    assert ev.families[0][0] == 'timing'
    assert ev.hypothesis.startswith('Аномальные задержки: длительность вызова инструмента: 12.3 с при ожидаемых ~400 мс')
    assert 'шаг «get_rate» (tool), длительность 12.3 с' in ev.hypothesis
    assert [s.span_id for s in ev.spans] == ['s-tool', 's-llm']         # смысловые шаги — только при смысловом сигнале
    assert ev.location() == {'agent_id': 'agent-1', 'span_id': 's-tool', 'span_name': 'get_rate',
                             'span_kind': 'tool', 'source': 'detector'}


@pytest.mark.parametrize('z_behavior,z_semantic,share,expected', [
    (6.0, 0.5, 0.8, 'behavior'),
    (0.5, 4.0, 0.8, 'semantic'),       # z решает раньше раскладки флага
    (5.0, 3.0, 0.5, 'mixed'),
    (1.0, 1.0, 0.9, 'behavior'),       # ни одна ветвь не выделяется — решает раскладка
    (1.0, 1.0, 0.2, 'semantic'),
    (1.0, 1.0, 0.5, 'mixed'),
])
def test_dominant_branch(z_behavior, z_semantic, share, expected):
    assert parse(z_behavior=z_behavior, z_semantic=z_semantic, share=share).dominant == expected


def test_semantic_signal_points_to_the_semantic_span():
    ev = parse(z_behavior=0.3, z_semantic=5.0)
    assert ev.category == 'semantic'
    assert 'текста шага «answer» (llm)' in ev.hypothesis and 'галлюцинац' in ev.hypothesis
    assert ev.span('s-llm').roles == ['behavior', 'semantic']            # один шаг — обе роли, без дублей


def test_error_status_span_wins_the_hypothesis():
    spans = {'s-tool': {'name': 'get_rate', 'kind': 'tool', 'status': 'STATUS_CODE_ERROR',
                        'status_message': 'upstream timeout'},
             's-llm': {'name': 'answer', 'kind': 'llm', 'status': 'STATUS_CODE_OK'}}
    behavior_spans = [{'rank': 1, 'index': 5, 'span_id': 's-llm', 'vs_typical': 9.0, 'drivers': []},
                      {'rank': 2, 'index': 3, 'span_id': 's-tool', 'vs_typical': 3.0, 'drivers': []}]
    ev = parse(spans=spans, behavior_spans=behavior_spans)
    assert ev.category == 'technical_error'
    assert ev.hypothesis == 'Шаг «get_rate» (tool) завершился ошибкой: upstream timeout.'
    assert ev.spans[0].span_id == 's-tool'                                # шаг с ошибкой — первым


@pytest.mark.parametrize('p,strength,caveat', [
    (0.95, 'strong', False), (0.6, 'moderate', False), (0.3, 'weak', True)])
def test_strength_and_low_probability_caveat(p, strength, caveat):
    ev = parse(p=p)
    assert ev.strength == strength
    assert any('вероятно это ложное срабатывание' in c for c in ev.caveats) is caveat


def test_caveats_truncation_sequence_scope_and_clipping():
    features = [{'rank': 1, 'feature': 'duration_max', 'direction': 'higher', 'observed_raw': 1e10,
                 'expected_raw': 1e9, 'clipped': True, 'error_share': 0.5}]
    ev = parse(truncated=True, features=features)
    assert 'детектор оценил только первые 30 из 40 шагов' in ev.caveats
    assert any('всей последовательности' in c for c in ev.caveats)
    assert any('границу нормализации' in c for c in ev.caveats)
    assert ev.features == ['длительность шага (максимум по шагам агента): ≥ 10.0 с при ожидаемых ~1.0 с (выше нормы)']


@pytest.mark.parametrize('view,info,phrase', [
    ({'feature': 'exec_gap', 'direction': 'higher', 'observed_raw': 9e10, 'expected_raw': 3e7},
     {'base': 'exec_gap'}, 'пауза между концом предыдущего шага и началом текущего: 1.5 мин при ожидаемых ~30 мс (выше нормы)'),
    ({'feature': 'char_count_std', 'direction': 'lower'},
     {'base': 'char_count', 'aggregation': 'std'}, 'длина ответа шага, символов (разброс по шагам агента): ниже нормы'),
    ({'feature': 'prompt_char_count_rolling_q75_w5', 'direction': 'higher', 'observed_raw': 5231.4, 'expected_raw': 812.2},
     {'base': 'prompt_char_count', 'aggregation': 'rolling_q75', 'window': 5},
     'длина входа шага, символов (скользящий 75-й перцентиль по окну 5 шагов): 5231 при ожидаемых ~812 (выше нормы)'),
    ({'feature': 'brand_new_feature', 'direction': 'lower', 'observed_raw': 0.123, 'expected_raw': 0.5},
     {}, 'brand_new_feature: 0.12 при ожидаемых ~0.50 (ниже нормы)'),
])
def test_feature_phrases(view, info, phrase):
    assert evidence._feature_phrase(view, info) == phrase


def test_prompt_view_is_compact():
    long = 'ж' * 1000
    spans = {f's{i}': {'name': f'step{i}', 'kind': 'chain', 'input_excerpt': long, 'output_excerpt': long}
             for i in range(5)}
    behavior_spans = [{'rank': i + 1, 'index': i, 'span_id': f's{i}', 'vs_typical': 5.0 - i, 'drivers': []}
                      for i in range(5)]
    view = parse(spans=spans, behavior_spans=behavior_spans).prompt_view()
    assert len(view['suspicious_spans']) == 3
    assert all(len(s['input']) <= 200 and len(s['output']) <= 200 for s in view['suspicious_spans'])
    assert len(view['deviating_features']) <= 4


def test_real_laim_payload_is_understood():
    """Записи из настоящего прогона laim (smoke-модель на data/traces_1k_sample.parquet)."""
    for record in load_fixture()['anomalies']:
        ev = evidence.parse(record)
        assert ev is not None and ev.agent_id == record['detector_rca']['agent_id']
        assert ev.dominant == 'behavior' and ev.strength == 'strong'
        assert ev.hypothesis and ev.features and ev.spans
        assert all(s.span_id in record['detector_rca']['spans'] for s in ev.spans)
        assert ev.spans[0].details.get('name')                            # детали шага приехали из каталога
        json.dumps(ev.prompt_view(), ensure_ascii=False, allow_nan=False)


def test_glossary_formats():
    assert glossary.format_value(2.5e9, 'ns') == '2.5 с'
    assert glossary.format_value(4.2e8, 'ns') == '420 мс'
    assert glossary.format_value(1.8e11, 'ns') == '3.0 мин'
    assert glossary.format_value(1234.4, 'chars') == '1234'
    assert glossary.format_value(2.34, 'count') == '2.3'
    assert glossary.aggregation_phrase('q95', None) == '95-й перцентиль по шагам агента'
    assert glossary.aggregation_phrase('rolling_mean', 10) == 'скользящее среднее по окну 10 шагов'
    assert glossary.describe('unknown') == ('other', 'unknown', '')
