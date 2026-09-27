"""Связи между записями: похожие вопросы, общие коды и термины.

Сильнейшие доказательства аномалии часто лежат в ДРУГИХ трейсах: на один и
тот же вопрос агент в одном трейсе нашёл ответ, а в другом — «не найдено»;
один термин расшифрован по-разному; один и тот же сбой повторяется. Модель
видит только свой пакет, поэтому связи находятся здесь, детерминированно,
по всему входу:

- для каждой записи — до MAX_RELATED других записей с общими редкими
  терминами (коды, аббревиатуры, номера, значимые слова) из запроса и ответа;
- порядок обработки, при котором связанные записи идут подряд и попадают
  в один пакет.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

MAX_RELATED = 3
SNIPPET_CHARS = 220

# слова вопроса, которые ничего не говорят о теме
_STOP = frozenset("""
что как где когда зачем почему какой какая какие каких каким какую который которые это эта этот эти
такое такой такая такие есть нет или для при про над под без через между после перед если чтобы
можно нужно надо будет было были быть мне меня мой моя мои вам вас ваш ваша наш наша они она оно
его ее её их там тут здесь все всё всех весь вся уже еще ещё только тоже также очень более менее
напиши подскажи скажи расскажи покажи объясни помоги найди посчитай дай пожалуйста привет здравствуйте
добрый день спасибо вопрос ответ код коду кодам кода коды кодом кодов кодами коде по на из от до не ни да же ли бы то
the and for with what how why which this that from are was
""".split())
_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё_\-\.]*[0-9A-Za-zА-Яа-яЁё]|[0-9A-Za-zА-Яа-яЁё]")


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ''


def _classify(text: str) -> tuple[set[str], set[str]]:
    """(все значимые термины, из них коды). Коды — DFA_OPER_FEE_SC_WD, 47109.99,
    П3399, аббревиатуры (ГБК, CCA); прочие — слова от 4 букв вне стоп-списка.
    Регистр не важен: «крюл» в запросе совпадает с «КРЮЛ» в чужом ответе."""
    found, codes = set(), set()
    for token in _TOKEN.findall(text):
        folded = token.casefold().strip('.-_')
        if not folded or folded in _STOP:
            continue
        coded = any(c.isdigit() for c in token) or '_' in token or (token.isupper() and len(token) >= 2)
        if coded or len(folded) >= 4:
            found.add(folded)
        if coded:
            codes.add(folded)
    return found, codes


def terms(text: str) -> set[str]:
    return _classify(text)[0]


def _snippet(text: str) -> str:
    flat = ' '.join(text.split())
    return flat if len(flat) <= SNIPPET_CHARS else flat[:SNIPPET_CHARS - 1].rstrip() + '…'


def find_related(records: list[dict]) -> list[list[int]]:
    """Для каждой записи — индексы связанных записей (сильнейшие первыми).

    Ключ записи — термины её ЗАПРОСА (о чём спрашивали); совпадение ищется в
    запросе и ответе других записей. Вес термина — IDF по всему входу, так что
    частые слова не связывают всё со всем; записи той же трассы связаны всегда.
    """
    classified = [_classify(_text(r.get('user_query'))) for r in records]
    queries = [found for found, _ in classified]
    # код, заданный в вопросе строчными («что такое крюл»), — тоже код, если где-то он написан как код
    all_codes = set().union(*(_classify(_text(r.get('user_query')) + ' ' + _text(r.get('agent_response')))[1]
                              for r in records)) if records else set()
    docs = [q | terms(_text(r.get('agent_response'))) for q, r in zip(queries, records)]
    n = len(records)
    df = Counter(t for doc in docs for t in doc)
    idf = {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}
    # термин встречается почти везде — это тема всего агента, а не связь
    common = {t for t, c in df.items() if n >= 5 and c > 0.5 * n}
    related = []
    for i, (query, record) in enumerate(zip(queries, records)):
        keys = query - common
        scored = []
        for j, doc in enumerate(docs):
            if j == i:
                continue
            same_trace = record.get('trace_id') is not None and records[j].get('trace_id') == record.get('trace_id')
            shared = keys & doc
            # связь — это общий код/аббревиатура или хотя бы два общих значимых слова:
            # одно слово («кодом», «ошибку») связывает случайные записи
            linked = same_trace or bool(shared & all_codes) or len(shared) >= 2
            score = sum(idf[t] for t in shared) + (100.0 if same_trace else 0.0)
            if linked and score > 0:
                scored.append((-score, j))
        related.append([j for _, j in sorted(scored)[:MAX_RELATED]])
    return related


def related_view(records: list[dict], indices: list[int]) -> list[dict]:
    """Короткие карточки связанных записей для промпта."""
    view = []
    for j in indices:
        record = records[j]
        item = {'trace_id': record.get('trace_id')}
        for key in ('user_query', 'agent_response'):
            if _text(record.get(key)).strip():
                item[key] = _snippet(record[key])
        view.append(item)
    return view


def processing_order(records: list[dict], related: list[list[int]]) -> list[int]:
    """Порядок обработки: запись, за ней её связанные (в ширину) — чтобы
    связанные записи попадали в один пакет; каждая запись ровно один раз."""
    placed: set[int] = set()
    order: list[int] = []
    for start in range(len(records)):
        if start in placed:
            continue
        queue = [start]
        while queue:
            i = queue.pop(0)
            if i in placed:
                continue
            placed.add(i)
            order.append(i)
            queue.extend(j for j in related[i] if j not in placed)
    return order
