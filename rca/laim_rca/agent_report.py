"""Отчёт о разработке агента -> компактный контекст для RCA (опциональный вход ноды).

Основной путь — сам документ (.docx, HTML или MHTML-экспорт Confluence/Word): в нём дословно то, что нужно для RCA —
критерии корректного ответа и штрафы, стоп-фразы, границы компетенций,
инструменты (их имена совпадают с именами шагов в трейсах), штатные
заглушки, параметры генерации и промпты. Выход g-aiva-doc-browser тоже
принимается, но он собран для валидации модели (метрика, порог, выборки,
суммаризация) и эти детали теряет — это запасной вариант.

Документ читается стандартной библиотекой (zipfile + XML, html.parser, email
для MHTML): параграфы и таблицы в порядке следования, без незаполненных пунктов
шаблона отчёта.
"""
from __future__ import annotations

import io
import json
import pickle
import re
import zipfile
from dataclasses import dataclass
from email import policy as email_policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from laim_rca.log import log

_W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
CELL_CHARS = 1200               # длинные ячейки (промпты в приложении) режутся
_TEMPLATE = re.compile(
    r'^(Заполняется на этапе|Если в отчёте использовались|Требуемые значения метрик|Сокращение \| Пояснение$)',
    re.IGNORECASE)
# контакты — персональные данные, для RCA бесполезны
_CONTACTS = re.compile(r'^\| Контакты', re.IGNORECASE)

# разделы шаблона отчёта о разработке (заголовки бывают и стилями, и простым текстом)
SECTIONS = (
    'Глоссарий', 'Описание задачи', 'Бриф', 'Бизнес-процесс', 'Эффект от внедрения', 'Метрики',
    'Техническое задание', 'Дизайн пилота', 'Описание данных', 'Процесс порождения данных',
    'Контрольные датасеты', 'Тренировочные датасеты', 'Разметка данных', 'Настройка и обучение',
    'Подбор промптов', 'Дообучение', 'Эксплуатация модели', 'Архитектура AI-решения',
    'Описание внутреннего устройства', 'Инструкция по воспроизведению', 'Описание процесса разработки',
    'Результаты оценки решения', 'Тестирование решения на статус SOTA', 'Перечень артефактов', 'Приложение')
# что выбрасывать при нехватке бюджета — по порядку: сначала то, что о валидации модели,
# а не о поведении агента; приложение (тексты промптов) — последним
DROP_ORDER = (
    'Перечень артефактов', 'Тестирование решения на статус SOTA', 'Дизайн пилота', 'Тренировочные датасеты',
    'Разметка данных', 'Контрольные датасеты', 'Дообучение', 'Подбор промптов', 'Инструкция по воспроизведению',
    'Описание процесса разработки', 'Результаты оценки решения', 'Эффект от внедрения', 'Глоссарий', 'Приложение')


@dataclass(frozen=True)
class AgentReport:
    text: str
    source: str             # docx | html | text | doc_browser
    chars_total: int        # до усечения по бюджету
    truncated: bool
    sections: tuple[str, ...] = ()      # разделы шаблона, найденные в отчёте
    dropped: tuple[str, ...] = ()       # разделы, выброшенные по бюджету

    def audit(self) -> dict:
        return {'source': self.source, 'chars': len(self.text), 'chars_total': self.chars_total,
                'truncated': self.truncated, 'sections': list(self.sections), 'dropped': list(self.dropped)}


def _cut(text: str, limit: int) -> str:
    text = ' '.join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def _paragraph(p: ElementTree.Element) -> tuple[str, bool]:
    style = p.find(f'{_W}pPr/{_W}pStyle')
    name = (style.get(f'{_W}val') or '').lower() if style is not None else ''
    heading = name.startswith(('heading', 'заголовок', 'title')) or name in ('1', '2', '3')
    # w:br — перенос строки внутри абзаца: как <br> в HTML, отдельная строка
    text = ''.join((node.text or '') if node.tag == f'{_W}t' else '\n'
                   for node in p.iter() if node.tag in (f'{_W}t', f'{_W}br'))
    return text.strip(), heading


def _table(tbl: ElementTree.Element) -> list[str]:
    rows = []
    for tr in tbl.findall(f'{_W}tr'):
        cells: list[str] = []
        for tc in tr.findall(f'{_W}tc'):
            text = _cut(' '.join(_paragraph(p)[0].replace('\n', ' ') for p in tc.iter(f'{_W}p')), CELL_CHARS)
            if text and (not cells or cells[-1] != text):       # объединённые ячейки повторяются
                cells.append(text)
        if len(cells) > 1 or (cells and len(cells[0]) > 40):     # «ключ» без значения — шаблон
            rows.append('| ' + ' | '.join(cells))
    return rows


def _docx_lines(data: bytes) -> list[tuple[str, bool]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        body = ElementTree.fromstring(archive.read('word/document.xml')).find(f'{_W}body')
    lines: list[tuple[str, bool]] = []
    for child in body if body is not None else []:
        if child.tag == f'{_W}p':
            text, heading = _paragraph(child)
            lines.extend((' '.join(line.split()), heading) for line in text.split('\n') if line.strip())
        elif child.tag == f'{_W}tbl':
            lines.extend((row, False) for row in _table(child))
    return lines


class _HtmlLines(HTMLParser):
    """HTML -> те же строки, что и из .docx: (текст, заголовок?); таблицы — '| a | b'."""

    _BLOCKS = frozenset({'p', 'div', 'li', 'br', 'section', 'article', 'blockquote', 'pre', 'dt', 'dd', 'tr'})
    _SKIP = frozenset({'script', 'style', 'head', 'title', 'noscript', 'svg'})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[tuple[str, bool]] = []
        self._text: list[str] = []
        self._heading = False
        self._skip = 0
        self._cells: list[str] | None = None      # ячейки текущей строки таблицы
        self._cell: list[str] | None = None
        self._tables = 0

    def _flush(self) -> None:
        text = ' '.join(''.join(self._text).split())
        if text:
            self.lines.append((text, self._heading))
        self._text, self._heading = [], False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif tag == 'table':
            self._flush()
            self._tables += 1
        elif tag == 'tr' and self._tables:
            self._cells = []
        elif tag in ('td', 'th') and self._cells is not None:
            self._cell = []
        elif self._cell is not None:
            if tag in self._BLOCKS:
                self._cell.append(' ')
        elif re.fullmatch(r'h[1-6]', tag):
            self._flush()
            self._heading = True
        elif tag in self._BLOCKS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag in ('td', 'th') and self._cell is not None and self._cells is not None:
            text = _cut(''.join(self._cell), CELL_CHARS)
            if text and (not self._cells or self._cells[-1] != text):
                self._cells.append(text)
            self._cell = None
        elif tag == 'tr' and self._cells is not None:
            if len(self._cells) > 1 or (self._cells and len(self._cells[0]) > 40):
                self.lines.append(('| ' + ' | '.join(self._cells), False))
            self._cells = None
        elif tag == 'table':
            self._tables = max(0, self._tables - 1)
        elif self._cell is None and (re.fullmatch(r'h[1-6]', tag) or tag in self._BLOCKS):
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        (self._cell if self._cell is not None else self._text).append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def _html_lines(html: str) -> list[tuple[str, bool]]:
    parser = _HtmlLines()
    parser.feed(html)
    parser.close()
    return parser.lines


def _mhtml_html(data: bytes) -> str:
    """HTML-часть MHTML (экспорт Confluence «в Word», Word «веб-страница в одном файле»)."""
    message = BytesParser(policy=email_policy.default).parsebytes(data)
    part = message.get_body(preferencelist=('html',)) if message.is_multipart() else message
    if part is None:
        raise ValueError('в MHTML нет HTML-части')
    return part.get_content()


def _drop_template(lines: list[tuple[str, bool]]) -> list[str]:
    """Без пунктов шаблона: «Заполняется на этапе…» и вопросы «- …», на
    которые в отчёте нет ответа (за ними сразу следующий вопрос или раздел)."""
    kept: list[str] = []
    for i, (text, heading) in enumerate(lines):
        if _TEMPLATE.match(text) or _CONTACTS.match(text):
            continue
        if heading:
            kept.append(f'## {text}')
            continue
        # вопрос шаблона («- описание …», «- кто является …») — со строчной буквы; без
        # ответа (дальше снова пункт или раздел) он выбрасывается. Пункты-содержание
        # («- API ФССК (версия 1.0)») начинаются с заглавной и остаются.
        if re.match(r'-\s*[а-яёa-z]', text):
            following = next(((t, h) for t, h in lines[i + 1:] if not _TEMPLATE.match(t)), None)
            if following is None or following[1] or following[0].startswith('-'):
                continue
        kept.append(text)
    # разделы, от которых остался один заголовок, не нужны
    return [line for i, line in enumerate(kept)
            if not (line.startswith('## ') and (i + 1 == len(kept) or kept[i + 1].startswith('## ')))]


def _doc_browser_text(payload: dict) -> str | None:
    """Выход g-aiva-doc-browser: all_results.bp_card + extracted_fields."""
    fields = payload.get('extracted_fields', payload)
    card = payload.get('all_results', {}).get('bp_card') if isinstance(payload.get('all_results'), dict) \
        else payload.get('bp_card')
    parts = [f'## Карточка бизнес-процесса\n{card}'] if isinstance(card, str) and card.strip() else []
    if isinstance(fields, dict):
        for key, title in (('summary', 'Суммаризация'), ('key_points', 'Ключевые параметры'),
                           ('ml_architecture', 'Архитектура'), ('generation_hyperparams', 'Параметры генерации'),
                           ('sample_description', 'Выборки')):
            value = fields.get(key)
            if value in (None, '', {}, []):
                continue
            if isinstance(value, dict):
                value = '\n'.join(f'- {k}: {v}' for k, v in value.items())
            elif not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False)
            parts.append(f'## {title}\n{value}')
    return '\n'.join(parts) or None


_HTML_START = re.compile(r'^\s*(<!--.*?-->\s*)*<(!doctype\s+html|html|head|body|meta|table|div|h[1-6]|p)\b',
                         re.IGNORECASE | re.DOTALL)


def _decode(data: bytes) -> str:
    for encoding in ('utf-8-sig', 'cp1251'):          # Word «веб-страница» пишет в windows-1251
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


class _BuiltinsOnly(pickle.Unpickler):
    """pickle только из встроенных типов (dict/list/str/bytes/числа): так порт
    g-aiva-doc-browser отдаёт {"bin", "ext"}; любые классы запрещены — чужой
    pickle не может исполнить код."""

    def find_class(self, module: str, name: str):
        raise ValueError(f'pickle с объектом {module}.{name} не поддерживается: подайте .docx, HTML или текст')


def _looks_like_text(text: str) -> bool:
    if not text:
        return False
    noise = sum(1 for c in text if (ord(c) < 32 and c not in '\n\r\t') or c == '\ufffd')
    return noise <= 0.01 * len(text)


def _from_bytes(data: bytes, ext: str = '') -> tuple[str, str] | None:
    ext = ext.lstrip('.').lower()
    if data[:1] == b'\x80' or ext in ('pkl', 'pickle'):             # pickle протокола 2+
        log('отчёт', f'формат: pickle ({len(data)} байт) — читаем только встроенные типы')
        try:
            obj = _BuiltinsOnly(io.BytesIO(data)).load()
        except (pickle.UnpicklingError, EOFError, ValueError) as error:
            raise ValueError(f'отчёт в pickle не прочитан: {error}') from error
        return _from_object(obj)
    if data[:2] == b'PK' or ext == 'docx':
        log('отчёт', f'формат: .docx ({len(data)} байт)')
        return '\n'.join(_drop_template(_docx_lines(data))), 'docx'
    if data[:5] == b'%PDF-' or ext == 'pdf':
        raise ValueError('PDF не поддерживается: подайте отчёт в .docx, HTML или текстом')
    head = data[:4096].lstrip().lower()
    if ext in ('mht', 'mhtml') or (head.startswith((b'mime-version', b'from:', b'content-type'))
                                   and b'multipart/related' in data[:8192].lower()):
        log('отчёт', f'формат: MHTML ({len(data)} байт) — извлекаем HTML-часть')
        return '\n'.join(_drop_template(_html_lines(_mhtml_html(data)))), 'html'
    if ext in ('html', 'htm'):
        log('отчёт', f'формат: HTML ({len(data)} байт)')
        return '\n'.join(_drop_template(_html_lines(_decode(data)))), 'html'
    text = _decode(data)
    if not _looks_like_text(text):
        raise ValueError('данные порта agent_report не похожи на текст, .docx или HTML')
    return _from_text(text)


def _from_text(text: str) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if _HTML_START.match(stripped):
        log('отчёт', f'формат: HTML-текст ({len(stripped)} симв.)')
        return '\n'.join(_drop_template(_html_lines(stripped))), 'html'
    if stripped[:1] in '{[':
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            log('отчёт', 'формат: JSON-объект')
            return _from_object(payload)
    log('отчёт', f'формат: обычный текст ({len(stripped)} симв.)')
    return stripped, 'text'


def _from_object(source: Any) -> tuple[str, str] | None:
    if isinstance(source, dict):
        if isinstance(source.get('bin'), (bytes, bytearray)):          # вход doc-browser: {"bin", "ext"}
            log('отчёт', f'получен словарь {{"bin", "ext"}} (ext={source.get("ext")!r})')
            return _from_bytes(bytes(source['bin']), str(source.get('ext', '')))
        text = _doc_browser_text(source)
        log('отчёт', 'формат: выход g-aiva-doc-browser' if text else 'словарь без распознаваемых полей отчёта')
        return (text, 'doc_browser') if text else None
    if isinstance(source, (bytes, bytearray)):
        return _from_bytes(bytes(source))
    if isinstance(source, str):
        candidate = source.strip()
        if candidate and '\n' not in candidate and len(candidate) < 4096:
            try:
                path = Path(candidate)
                if path.is_file():                                   # порт отдал путь к файлу
                    log('отчёт', f'порт передал путь к файлу: {path} ({path.stat().st_size} байт)')
                    return _from_bytes(path.read_bytes(), path.suffix)
            except OSError:
                pass
        return _from_text(source)
    raise ValueError(f'неподдерживаемый формат отчёта: {type(source).__name__}')


def load(source: Any, max_chars: int = 20_000) -> AgentReport | None:
    """AgentReport из порта agent_report; None, если порт пуст.
    ValueError — если данные поданы, но прочитать их нельзя."""
    if source is None or (isinstance(source, str) and not source.strip()):
        log('отчёт', 'порт agent_report пуст — анализ без отчёта о разработке')
        return None
    log('отчёт', f'порт agent_report подан: {type(source).__name__}')
    parsed = _from_object(source)
    if parsed is None:
        return None
    text, kind = parsed
    # заголовки разделов, под которыми в отчёте ничего не заполнено
    lines = text.split('\n')
    text = '\n'.join(line for i, line in enumerate(lines)
                     if _section_of(line) is None or (i + 1 < len(lines) and _section_of(lines[i + 1]) is None))
    total = len(text)
    sections = tuple(dict.fromkeys(name for line in text.split('\n') if (name := _section_of(line))))
    dropped: tuple[str, ...] = ()
    if total > max_chars:
        text, dropped = _fit(text, max_chars)
    return AgentReport(text=text, source=kind, chars_total=total, truncated=total > max_chars,
                       sections=sections, dropped=dropped)


def _section_of(line: str) -> str | None:
    title = line.removeprefix('## ').strip()
    return next((s for s in SECTIONS if title.startswith(s) and len(title) < len(s) + 60), None)


def _fit(text: str, max_chars: int) -> tuple[str, tuple[str, ...]]:
    """В бюджет: сначала выбрасываются разделы про валидацию модели (DROP_ORDER),
    затем — если всё ещё много — хвост документа."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.split('\n'):
        if (name := _section_of(line)) is not None:
            sections.append((name, [line]))
        else:
            sections[-1][1].append(line)
    size = lambda: sum(len(line) + 1 for _, lines in sections for line in lines)
    dropped = []
    for name in DROP_ORDER:
        if size() <= max_chars:
            break
        if any(n == name for n, _ in sections):
            sections = [(n, lines) for n, lines in sections if n != name]
            dropped.append(name)
    fitted = '\n'.join(line for _, lines in sections for line in lines)
    if len(fitted) > max_chars:
        fitted = fitted[:max_chars].rsplit('\n', 1)[0] + '\n…'
    note = f'\n[отчёт сокращён до {max_chars} симв.' + (f'; опущены разделы: {", ".join(dropped)}' if dropped else '') + ']'
    return fitted + note, tuple(dropped)
