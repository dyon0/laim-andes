"""Нода LAIM anomaly report: типы аномалий зависят от классификатора (s3),
без шапки «Автономный мониторинг…» и без сообщений о разметке; RCA обоих форматов."""
import inspect
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as report  # noqa: E402

RCA_OBJECT = {
    'verdict': 'uncertain', 'verdict_confidence': 70, 'severity': 'high',
    'rca': {'category': 'Галлюцинация', 'root_cause': 'Неверная ставка', 'evidence': ['22,5%/12 ≠ 0,1875']},
    'location': {'agent_id': 'agent-1', 'span_id': 's1', 'span_name': 'get_rate', 'span_kind': 'tool',
                 'source': 'llm'},
    'detector_evidence': {'hypothesis': 'Аномальные задержки: длительность вызова инструмента выше нормы.'},
    'analyzed_by': 'llm:minimax-m2.5',
}


def record(i: int, anomaly_type: str = '', **fields) -> dict:
    return {'trace_id': f't{i}', 'starttime': '2026-05-20T09:18:40Z', 'endtime': '2026-05-20T09:19:00Z',
            'anomaly_type': anomaly_type, 'confidence': 90 - i, 'user_query': f'вопрос {i}',
            'agent_response': f'ответ {i}', 'business_description': '', 'tech_details': '', 'rca_results': ''} | fields


def render(records, **settings) -> tuple[str, dict]:
    out = report.main(json.dumps({'anomalies': records}, ensure_ascii=False), **settings)
    return out['anomaly_report'], out['all_results']


TYPE_BLOCKS = ('Описание показателей', 'Распределение по типам аномалий', 'Типов аномалий',
               'Преобладающий тип', 'class="anomaly-type-badge')


def test_without_classifier_type_blocks_are_hidden():
    html, results = render([record(0), record(1), record(2)])      # s3 выключен: типы пусты
    for block in TYPE_BLOCKS:
        assert block not in html, block
    assert 'Сводка' in html and 'Перечень аномалий' in html and 'Аномалий выявлено' in html
    assert html.count('class="anomaly-card"') == 3
    assert 'приведены уверенность детектора' in html and 'приведены тип' not in html
    assert results['anomaly_types_shown'] is False


def test_with_classifier_type_blocks_are_shown():
    html, results = render([record(0, 'hallucination'), record(1, 'bias'), record(2, 'dpi+bias')])
    for block in TYPE_BLOCKS:
        assert block in html, block
    assert html.count('class="anomaly-type-badge') >= 3 and results['anomaly_types_shown'] is True
    assert 'Галлюцинация' in html and 'Prompt Injection · DPI' in html


def test_drift_records_do_not_count_as_classifier_output():
    html, _ = render([record(0), record(1, 'drift_global')])
    assert 'Описание показателей' not in html


@pytest.mark.parametrize('mode,expected', [('show', True), ('hide', False), ('auto', False)])
def test_explicit_override(mode, expected):
    html, results = render([record(0)], anomaly_types=mode)
    assert ('Описание показателей' in html) is expected and results['anomaly_types_shown'] is expected
    html, _ = render([record(0, 'hallucination')], anomaly_types='hide')
    assert 'Описание показателей' not in html


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='anomaly_types'):
        render([record(0)], anomaly_types='maybe')


@pytest.mark.parametrize('typed', [False, True])
def test_no_header_and_no_markup_messages(typed):
    html, results = render([record(0, 'hallucination' if typed else '')], min_confidence=0)
    title = results['calculated_traffic_lights']['semaphore_title']
    for text in (html, title):
        for banned in ('Автономный мониторинг', 'Детектор аномалий</span>', 'разметк', 'Владел'):
            assert banned not in text, banned
    assert title == 'Детектор выявил 1 аномалию с confidence ≥ 0'
    assert 'eyebrow' not in html


def test_rca_object_from_laim_rca_node_is_readable():
    html, _ = render([record(0, rca_results=RCA_OBJECT)])
    text, tags = report.rca_text(RCA_OBJECT)
    assert text.startswith('Причина: Неверная ставка\nДоказательства:\n  • 22,5%/12')
    assert 'Категория' not in text and 'Галлюцинация' not in text          # категории не показываем
    assert 'Где: шаг «get_rate» (tool), агент agent-1' in text
    assert 'Сигнал детектора' not in text                         # числа детектора — не для отчёта
    assert tags == ['LLM не уверена в аномалии', 'критичность: высокая']
    assert 'Неверная ставка' in html and '"verdict"' not in html            # не сырой JSON


@pytest.mark.parametrize('value,expected', [
    ('Арифметическая ошибка в ставке', 'Арифметическая ошибка в ставке'),              # прежний RCA: строка
    ({'anomaly_category': '1 - Арифметика', 'quote_with_error': '0,1875'},
     'Цитата с ошибкой: 0,1875'),                                                    # объект по add_info
    ({'category': 'failure_propagation_or_guardrail', 'data_amiguity_or_lookup_result': 'Код не найден.'},
     'Код не найден.'),                                                             # без англ. меток
    ('ГАЛЛЮЦИНАЦИЯ: Агент дал неверное определение ГБК.', 'Агент дал неверное определение ГБК.'),
    ('Категория: data_lookup. Агент не нашёл код.', 'Агент не нашёл код.'),
    ({'verdict': 'anomaly', 'rca': 'Смешение продуктов'}, 'Смешение продуктов'),
    ('', ''),
])
def test_rca_formats(value, expected):
    assert report.rca_text(value)[0] == expected


def test_descriptor_matches_entry_point():
    descriptor = json.loads((ROOT / 'descriptor.json').read_text(encoding='utf-8'))
    run = descriptor['script']['runConfiguration']
    assert run['sourceFiles'] == ['main.py'] and run['functionName'] == 'main'
    params = [c['parameter'] for c in descriptor['ui']['settings'][0]['components'][0]['config']['components']]
    assert params == ['min_confidence', 'anomaly_types']
    signature = set(inspect.signature(report.main).parameters)
    assert set(params) <= signature
    assert {p['name'] for p in descriptor['ports'] if p['in']} <= signature
    assert {p['name'] for p in descriptor['ports'] if not p['in']} == set(report.main('[]'))


def test_trace_ids_in_rca_link_to_their_cards():
    records = [record(0, trace_id='aaaa1111bbbb2222cccc', rca_results='ПРОТИВОРЕЧИЕ МЕЖДУ ТРЕЙСАМИ: в трейсе '
                      'eeee5555ffff… агент нашёл ответ, а в 0123456789abcdef (не в отчёте) — нет.'),
               record(1, trace_id='eeee5555ffff6666aaaa', rca_results='ГАЛЛЮЦИНАЦИЯ: см. aaaa1111bbbb2222cccc')]
    html, _ = render(records)
    assert 'id="trace-aaaa1111bbbb2222cccc"' in html and 'id="trace-eeee5555ffff6666aaaa"' in html
    assert '<a class="trace-ref" href="#trace-eeee5555ffff6666aaaa">eeee5555ffff (#002)</a>' in html
    assert '<a class="trace-ref" href="#trace-aaaa1111bbbb2222cccc">aaaa1111bbbb2222cccc (#001)</a>' in html
    assert '0123456789abcdef (не в отчёте)' in html                  # чужой id без карточки — просто текст


def test_report_usage_is_visible_on_the_card():
    html, _ = render([record(0, rca_results={'verdict': 'anomaly', 'rca': 'Причина.', 'agent_report_used': True})])
    assert 'с учётом отчёта о разработке' in html
