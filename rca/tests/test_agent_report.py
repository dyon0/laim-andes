"""Опциональный порт agent_report: отчёт о разработке агента как контекст анализа."""
import io
import json
import zipfile
from xml.sax.saxutils import escape

import pytest

from conftest import anomaly, rca
from laim_rca import agent_report

_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def _p(text: str, heading: bool = False) -> str:
    style = '<w:pPr><w:pStyle w:val="Heading1"/></w:pPr>' if heading else ''
    return f'<w:p>{style}<w:r><w:t>{escape(text)}</w:t></w:r></w:p>'


def _row(*cells: str) -> str:
    return '<w:tr>' + ''.join(f'<w:tc>{_p(c)}</w:tc>' for c in cells) + '</w:tr>'


def make_docx(extra: str = '') -> bytes:
    """Мини-отчёт в структуре шаблона: заполненные и пустые пункты, таблица брифа, контакты."""
    body = ''.join([
        _p('Отчет о разработке'),
        '<w:tbl>' + _row('Название инициативы', 'AI агент гос. ограничения')
        + _row('Контакты заказчика', 'Иванов И.И. <ivanov@example.com>') + '</w:tbl>',
        _p('Бриф', heading=True),
        '<w:tbl>' + _row('Цель применения LLM', 'Определить статус ограничения и дать инструкцию по снятию') + '</w:tbl>',
        _p('Техническое задание', heading=True),
        _p('За что штрафуем:'),
        _p('Фразы из стоп-листа: «возьмите микрозайм».'),
        _p('- описание формул/логики расчета метрик (для нестандартных метрик)'),     # без ответа — шаблон
        _p('- доп. технические ограничения'),
        _p('Время ответа не более 10 секунд'),
        _p('Заполняется на этапе 2'),
        _p('Контрольные датасеты', heading=True),
        _p('300 кейсов, синтетика. ' * 40),
        _p('Архитектура AI-решения', heading=True),
        _p('Тул get_restrictions — API ФССК; при ошибке сервиса заглушка «По техническим причинам сейчас я не могу ответить»'),
        _p('Приложение', heading=True),
        _p('Промпт: ' + 'ж' * 3000),
        extra,
    ])
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{_NS}"><w:body>{body}</w:body></w:document>'
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('word/document.xml', xml)
    return buffer.getvalue()


def system_of(fake_llm) -> str:
    return fake_llm.calls[0][0][1]


def run_audit(**settings) -> dict:
    return json.loads(rca.main(json.dumps({'anomalies': [anomaly(0)]}), **settings)['rca_audit'])


def test_docx_becomes_clean_context():
    report = agent_report.load(make_docx(), max_chars=100_000)
    assert report.source == 'docx' and not report.truncated
    text = report.text
    assert '| Цель применения LLM | Определить статус ограничения' in text
    assert '«возьмите микрозайм»' in text and 'get_restrictions' in text and 'Время ответа не более 10 секунд' in text
    assert '## Техническое задание' in text
    assert 'Заполняется на этапе' not in text                      # служебные строки шаблона
    assert 'описание формул' not in text                           # вопрос шаблона без ответа
    assert 'ivanov@example.com' not in text                        # контакты — персональные данные


def test_budget_drops_validation_sections_before_agent_behavior():
    report = agent_report.load(make_docx(), max_chars=1_500)
    assert report.truncated and len(report.text) <= 1_700
    assert 'get_restrictions' in report.text and '«возьмите микрозайм»' in report.text
    assert '300 кейсов' not in report.text and 'жжж' not in report.text
    assert 'опущены разделы: Контрольные датасеты, Приложение' in report.text


def test_report_goes_to_the_system_prompt(fake_llm):
    out = run_audit(agent_report=make_docx())

    system = system_of(fake_llm)
    assert 'AGENT_REPORT:\n' in system and 'get_restrictions' in system
    assert 'штатная заглушка' in system                            # как пользоваться отчётом
    assert system.index('AGENT_REPORT') < len(system)
    assert 'get_restrictions' not in fake_llm.calls[0][1][1]         # в данных пакета отчёта нет
    assert out['agent_report']['source'] == 'docx' and out['agent_report']['truncated'] is False


def test_without_the_port_nothing_changes(fake_llm):
    out = run_audit()
    assert 'AGENT_REPORT' not in system_of(fake_llm) and out['agent_report'] is None
    run_audit(agent_report='   ')
    assert 'AGENT_REPORT' not in system_of(fake_llm)


@pytest.mark.parametrize('shape', ['path', 'bin_dict', 'text', 'doc_browser'])
def test_accepted_input_shapes(fake_llm, tmp_path, shape):
    if shape == 'path':
        path = tmp_path / 'unstructured_data'                      # платформа отдаёт файл без расширения
        path.write_bytes(make_docx())
        source, expected = str(path), 'get_restrictions'
    elif shape == 'bin_dict':                                      # вход g-aiva-doc-browser: {"bin", "ext"}
        source, expected = {'bin': make_docx(), 'ext': 'docx'}, 'get_restrictions'
    elif shape == 'text':
        source, expected = 'Агент консультирует по кредитам. Стоп-фразы: «займите у родственников».', 'займите'
    else:                                                          # выход g-aiva-doc-browser
        source = {'all_results': {'bp_card': 'Карточка: агент гос. ограничений'},
                  'extracted_fields': {'summary': 'Агент помогает оператору ЦКР', 'key_points': {'ml_task': 'Генерация'},
                                       'ml_architecture': 'MCP-сервер агрегирует данные API ФССК'}}
        expected = 'MCP-сервер агрегирует'
    out = run_audit(agent_report=source)
    assert expected in system_of(fake_llm)
    assert out['agent_report']['source'] == {'path': 'docx', 'bin_dict': 'docx', 'text': 'text',
                                              'doc_browser': 'doc_browser'}[shape]


def test_unreadable_report_does_not_fail_the_run(fake_llm):
    out = run_audit(agent_report=b'%PDF-1.7 ...')
    assert 'PDF' in out['agent_report']['error'] and out['counts']['output'] == 1
    assert 'AGENT_REPORT' not in system_of(fake_llm)
    out = run_audit(agent_report={'bin': b'PK not really a zip', 'ext': 'docx'})
    assert 'BadZipFile' in out['agent_report']['error']


# --- HTML-отчёты (Confluence, Word «веб-страница», MHTML-экспорт) --------------

HTML_REPORT = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Отчет</title>
<style>td {color: red}</style><script>var secret = 'не текст отчёта';</script></head><body>
<h1>Отчет о разработке</h1>
<table><tbody>
  <tr><th>Название инициативы</th><td>AI агент гос. ограничения</td></tr>
  <tr><th>Контакты заказчика</th><td>ivanov@example.com</td></tr>
  <tr><th>Цель применения LLM</th><td><p>Определить статус ограничения</p><p>и дать инструкцию по снятию</p></td></tr>
</tbody></table>
<h2>Техническое задание</h2>
<p>За что штрафуем:</p>
<ul><li>Фразы из стоп-листа: «возьмите микрозайм».</li></ul>
<p>- описание формул/логики расчета метрик</p>
<p>- доп. технические ограничения</p>
<p>Время ответа не более 10&nbsp;секунд</p>
<p>Заполняется на этапе 2</p>
<h2>Описание данных</h2>
<p>Источники данных:<br/><strong>- API ФССК (версия агента 1.0)<br/>- API Исп.П (версия агента 1.0)</strong></p>
<h2>Архитектура AI-решения</h2>
<div>Тул <code>get_restrictions</code> &mdash; при ошибке заглушка «По техническим причинам…»</div>
</body></html>"""


def test_html_report_parses_like_docx():
    report = agent_report.load(HTML_REPORT, max_chars=100_000)
    text = report.text
    assert report.source == 'html'
    assert '## Техническое задание' in text and '## Архитектура AI-решения' in text
    assert '| Цель применения LLM | Определить статус ограничения и дать инструкцию по снятию' in text
    assert '«возьмите микрозайм»' in text and 'Время ответа не более 10 секунд' in text
    assert '- API ФССК (версия агента 1.0)' in text and '- API Исп.П (версия агента 1.0)' in text
    assert 'get_restrictions — при ошибке заглушка' in text
    for absent in ('secret', 'color: red', 'ivanov@example.com', 'описание формул', 'Заполняется на этапе'):
        assert absent not in text, absent


def test_html_file_mhtml_and_legacy_encoding(tmp_path):
    page = tmp_path / 'report.html'
    page.write_text(HTML_REPORT, encoding='utf-8')
    assert 'get_restrictions' in agent_report.load(str(page)).text

    word_page = tmp_path / 'report.htm'                            # Word «веб-страница»: windows-1251
    word_page.write_bytes(HTML_REPORT.replace('charset="utf-8"', 'charset="windows-1251"').encode('cp1251'))
    assert '«возьмите микрозайм»' in agent_report.load(str(word_page)).text

    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    message = MIMEMultipart('related')                             # экспорт Confluence «в Word» (.doc)
    message.attach(MIMEText(HTML_REPORT, 'html', 'utf-8'))
    mhtml = tmp_path / 'export.doc'
    mhtml.write_bytes(message.as_bytes())
    report = agent_report.load(str(mhtml))
    assert report.source == 'html' and '- API ФССК (версия агента 1.0)' in report.text


def test_html_report_reaches_the_prompt(fake_llm):
    out = run_audit(agent_report={'bin': HTML_REPORT.encode(), 'ext': 'html'})
    assert 'get_restrictions' in system_of(fake_llm) and out['agent_report']['source'] == 'html'


def test_docx_line_breaks_keep_list_items():
    """«- API …» после w:br — содержание, а не пустые вопросы шаблона."""
    br = '<w:p><w:r><w:t>Источники данных:</w:t><w:br/><w:t>- API ФССК (версия 1.0)</w:t><w:br/><w:t>- API Исп.П</w:t></w:r></w:p>'
    text = agent_report.load(make_docx(br), max_chars=100_000).text
    assert '- API ФССК (версия 1.0)\n- API Исп.П' in text
