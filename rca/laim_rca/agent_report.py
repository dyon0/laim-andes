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

import base64
import binascii
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
    noise = sum(1 for c in text if (ord(c) < 32 and c not in '\n\r\t') or c == '�')
    return noise <= 0.01 * len(text)


# --- транспорт порта -----------------------------------------------------------
# DataArtifact в SberDS не имеет одного Python-представления: файл приходит
# локальным путём (часто без расширения), каталогом «as files and folders»
# (рядом служебные _SUCCESS / *.crc), bytes, словарём коннектора {"bin", "ext"}
# (иногда под другим ключом, иногда pickle), однострочным parquet / DataFrame
# с bytes или путём внутри. Сначала транспорт сводится к байтам документа,
# затем разбирается сам документ.
_BINARY_KEYS = ('bin', 'bytes', 'content', 'data', 'payload', 'value', 'unstructured_data', 'file_bytes')
_PATH_KEYS = ('path', 'local_path', 'file', 'file_path', '__file__')
_EXT_KEYS = ('ext', 'extension', 'suffix', 'filename', 'file_name', 'name')
_SERVICE_FILES = ('_SUCCESS', '_started', '_committed')
# при нескольких файлах в каталоге порта — по приоритету формата
_FILE_PRIORITY = ('.docx', '.html', '.htm', '.mht', '.mhtml', '.pkl', '.pickle', '', '.txt', '.md', '.json')
_MAX_DEPTH = 6
_BASE64 = re.compile(r'[A-Za-z0-9+/=\s]+')
_NOT_PROVIDED = frozenset({'', 'none', 'null', 'nan', 'nat'})


def _ext(value: Any) -> str:
    """'.docx' / 'docx' / 'report.docx' -> 'docx'; имя без расширения -> ''."""
    text = str(value or '').strip().lower()
    if '.' in text:
        return text.rsplit('.', 1)[-1]
    return text if text.isalnum() and len(text) <= 6 else ''


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float):
        return value == value                                        # NaN из parquet — пусто
    if isinstance(value, str):
        return value.strip().lower() not in _NOT_PROVIDED
    if isinstance(value, (bytes, bytearray, memoryview, dict, list, tuple)):
        return len(value) > 0
    return True


def describe(source: Any) -> str:
    """Что именно пришло в порт — для лога (без содержимого документа)."""
    kind = type(source).__name__
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        return f'{kind}, {len(data)} байт, начало {data[:8]!r}'
    if isinstance(source, dict):
        keys = ', '.join(f'{k}: {type(v).__name__}' for k, v in list(source.items())[:12])
        return f'dict {{{keys}}}'
    if isinstance(source, (str, Path)):
        text = str(source)
        if '\n' not in text and len(text) < 4096:
            try:
                path = Path(text.strip())
                if path.is_dir():
                    names = sorted(p.name for p in path.iterdir())
                    return f'путь к каталогу {path} (файлы: {", ".join(names[:10]) or "нет"})'
                if path.is_file():
                    return f'путь к файлу {path} ({path.stat().st_size} байт)'
            except OSError:
                pass
        return f'{kind}, {len(text)} симв., начало {text[:60]!r}'
    if hasattr(source, 'columns') and hasattr(source, 'to_dict'):
        return f'{kind} {getattr(source, "shape", "")}, колонки {list(source.columns)[:10]}'
    return kind


def _directory_file(directory: Path) -> Path:
    files = sorted(p for p in directory.rglob('*')
                   if p.is_file() and '__MACOSX' not in p.parts and not p.name.startswith(('.', '._'))
                   and not p.name.endswith('.crc') and not p.name.startswith(_SERVICE_FILES))
    if not files:
        raise ValueError(f'в каталоге порта нет файлов с отчётом: {directory}')
    if len(files) == 1:
        return files[0]
    ranked = sorted(files, key=lambda p: (_FILE_PRIORITY.index(p.suffix.lower())
                                          if p.suffix.lower() in _FILE_PRIORITY else len(_FILE_PRIORITY),
                                          -p.stat().st_size))
    log('отчёт', f'в каталоге {len(files)} файлов ({", ".join(p.name for p in files[:10])}) — берём {ranked[0].name}')
    return ranked[0]


def _frame_rows(frame: Any) -> list[dict]:
    try:
        return [dict(row) for row in frame.to_dict('records')]
    except TypeError:                                                  # pandas.Series / polars
        return [dict(frame.to_dict())]


def _from_parquet(data: bytes, ext: str, depth: int) -> tuple[str, str] | None:
    try:
        import pandas
        frame = pandas.read_parquet(io.BytesIO(data))
    except ImportError as error:
        raise ValueError('отчёт пришёл parquet-контейнером, а pandas/pyarrow в образе нет') from error
    except Exception as error:
        raise ValueError(f'parquet-контейнер порта не прочитан: {type(error).__name__}: {error}') from error
    log('отчёт', f'parquet-контейнер: {len(frame)} строк, колонки {list(frame.columns)}')
    return _from_rows(_frame_rows(frame), '' if ext == 'parquet' else ext, depth)


def _from_rows(rows: list[dict], ext: str, depth: int) -> tuple[str, str] | None:
    """Строки DataFrame/parquet-обёртки: одна строка с bytes/путём — это файл."""
    rows = [row for row in rows if any(_present(v) for v in row.values())]
    if len(rows) == 1:
        return _from_object(rows[0], ext, depth + 1)
    if not rows:
        raise ValueError('таблица в порту пуста')
    columns = [k for k in (*_BINARY_KEYS, *_PATH_KEYS) if k in rows[0]]
    raise ValueError(f'таблица в порту: {len(rows)} строк; ожидалась одна строка с файлом отчёта'
                     + (f' (колонки {columns})' if columns else ''))


def _from_bytes(data: bytes, ext: str = '', depth: int = 0) -> tuple[str, str] | None:
    ext = ext.lstrip('.').lower()
    if data[:4] == b'PAR1':
        return _from_parquet(data, ext, depth)
    if data[:1] == b'\x80' or ext in ('pkl', 'pickle'):             # pickle протокола 2+
        log('отчёт', f'формат: pickle ({len(data)} байт) — читаем только встроенные типы')
        try:
            obj = _BuiltinsOnly(io.BytesIO(data)).load()
        except (pickle.UnpicklingError, EOFError, ValueError) as error:
            raise ValueError(f'отчёт в pickle не прочитан: {error}') from error
        log('отчёт', f'в pickle: {describe(obj)}')
        return _from_object(obj, '', depth + 1)
    if data[:2] == b'PK' or ext == 'docx':
        log('отчёт', f'формат: .docx ({len(data)} байт)')
        return '\n'.join(_drop_template(_docx_lines(data))), 'docx'
    if data[:5] == b'%PDF-' or ext == 'pdf':
        raise ValueError('PDF не поддерживается: подайте отчёт в .docx, HTML или текстом')
    if data[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':          # OLE: бинарный .doc (MHTML с .doc читается ниже)
        raise ValueError('старый формат Word (.doc) не поддерживается: пересохраните отчёт в .docx')
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
        raise ValueError(f'данные порта agent_report ({len(data)} байт, начало {data[:8]!r}) '
                         'не похожи на текст, .docx, HTML или pickle')
    return _from_text(text, ext, depth)


def _base64_document(text: str, ext: str) -> bytes | None:
    """bytes документа, если строка — base64 (так bytes иногда сериализуют в JSON)."""
    compact = ''.join(text.split())
    if len(compact) < 64 or not _BASE64.fullmatch(text):
        return None
    try:
        data = base64.b64decode(compact + '=' * (-len(compact) % 4), validate=True)
    except (binascii.Error, ValueError):
        return None
    return data if data[:2] == b'PK' or data[:1] == b'\x80' or data[:4] == b'PAR1' else None


def _looks_like_path(text: str) -> bool:
    return (text.startswith(('/', './', '~/', 'hdfs://', 'viewfs://', 'file://', 's3://'))
            or re.match(r'^[A-Za-z]:\\', text) is not None
            or (' ' not in text and _ext(text) in ('docx', 'html', 'htm', 'mht', 'mhtml', 'pkl', 'pickle', 'parquet')))


def _from_path(text: str, ext: str, depth: int) -> tuple[str, str] | None:
    """Строка-путь: файл или каталог порта; None — если это не путь."""
    candidate = text.strip().removeprefix('file://')
    if not candidate or '\n' in candidate or len(candidate) >= 4096:
        return None
    try:
        path = Path(candidate).expanduser()
        if path.is_dir():
            log('отчёт', f'порт передал каталог: {path}')
            path = _directory_file(path)
        if path.is_file():
            log('отчёт', f'читаем файл: {path} ({path.stat().st_size} байт)')
            suffix = _ext(path.suffix)
            return _from_bytes(path.read_bytes(), suffix if suffix and suffix != 'parquet' else ext, depth + 1)
    except OSError as error:
        if _looks_like_path(candidate):
            raise ValueError(f'файл отчёта {candidate} не прочитан: {error}') from error
        return None
    if _looks_like_path(candidate):
        if candidate.startswith(('hdfs://', 'viewfs://', 's3://')):
            raise ValueError(f'в порт пришёл только URI {candidate}, а не локальный файл: '
                             'проверьте, что порт монтируется как файл')
        raise ValueError(f'путь к отчёту не найден в контейнере: {candidate}')
    return None


def _from_text(text: str, ext: str = '', depth: int = 0) -> tuple[str, str] | None:
    stripped = text.strip()
    if not stripped:
        return None
    if (data := _base64_document(stripped, ext)) is not None:
        log('отчёт', f'строка — base64 документа ({len(data)} байт)')
        return _from_bytes(data, ext, depth + 1)
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
            return _from_object(payload, ext, depth + 1)
    log('отчёт', f'формат: обычный текст ({len(stripped)} симв.)')
    return stripped, 'text'


def _from_object(source: Any, ext: str = '', depth: int = 0) -> tuple[str, str] | None:
    if depth > _MAX_DEPTH:
        raise ValueError('слишком глубокая вложенность данных в порту agent_report')
    if isinstance(source, dict):
        for key in _EXT_KEYS:
            if isinstance(source.get(key), str) and _ext(source[key]):
                ext = _ext(source[key])
                break
        for key in (*_BINARY_KEYS, *_PATH_KEYS):                    # коннектор: {"bin", "ext"} и родственники
            if key in source and _present(source[key]) and not isinstance(source[key], (int, float, bool)):
                log('отчёт', f'словарь: документ под ключом {key!r} ({type(source[key]).__name__}, ext={ext or "?"})')
                return _from_object(source[key], ext, depth + 1)
        text = _doc_browser_text(source)
        if text:
            log('отчёт', 'формат: выход g-aiva-doc-browser')
            return text, 'doc_browser'
        raise ValueError(f'словарь без распознаваемых полей отчёта: ключи {sorted(map(str, source))[:12]}')
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _from_bytes(bytes(source), ext, depth)
    if isinstance(source, Path):
        source = str(source)
    if isinstance(source, str):
        found = _from_path(source, ext, depth)
        return found if found is not None else _from_text(source, ext, depth)
    if isinstance(source, (list, tuple)):
        items = [item for item in source if _present(item)]
        if len(items) == 1:
            return _from_object(items[0], ext, depth + 1)
        raise ValueError(f'в порту список из {len(items)} элементов; ожидался один отчёт')
    if hasattr(source, 'to_dict') and (hasattr(source, 'columns') or hasattr(source, 'index')):
        log('отчёт', f'табличная обёртка: {describe(source)}')
        return _from_rows(_frame_rows(source), ext, depth)
    raise ValueError(f'неподдерживаемый формат отчёта: {type(source).__name__}')


def load(source: Any, max_chars: int = 20_000) -> AgentReport | None:
    """AgentReport из порта agent_report; None, если порт пуст.
    ValueError — если данные поданы, но прочитать их нельзя (или текста в них нет)."""
    if source is None or (isinstance(source, (str, float)) and not _present(source)):
        log('отчёт', f'порт agent_report пуст ({describe(source)}) — анализ без отчёта о разработке')
        return None
    log('отчёт', f'порт agent_report подан: {describe(source)}')
    parsed = _from_object(source)
    if parsed is None or not parsed[0].strip():
        raise ValueError('отчёт прочитан, но текста в нём нет')
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
