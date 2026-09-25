"""Разбор ответа LLM: список результатов, сопоставление по id, валидация вердикта."""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

VERDICTS = ('anomaly', 'normal', 'uncertain')
SEVERITIES = ('low', 'medium', 'high', 'critical')
_VERDICT_ALIASES = {
    'anomaly': 'anomaly', 'anomalous': 'anomaly', 'аномалия': 'anomaly', 'confirmed': 'anomaly',
    'normal': 'normal', 'норма': 'normal', 'false_positive': 'normal', 'not_anomaly': 'normal', 'ok': 'normal',
    'uncertain': 'uncertain', 'unknown': 'uncertain', 'неопределено': 'uncertain', 'не определено': 'uncertain',
}
_SEVERITY_ALIASES = {
    'low': 'low', 'низкая': 'low', 'minor': 'low',
    'medium': 'medium', 'средняя': 'medium', 'moderate': 'medium',
    'high': 'high', 'высокая': 'high', 'major': 'high',
    'critical': 'critical', 'критическая': 'critical', 'blocker': 'critical',
}
# обёртки списка результатов, которые встречаются в ответах моделей
_WRAPPERS = ('results', 'anomalies', 'TRACES_DATA', 'records')


@dataclass(frozen=True)
class Analysis:
    """Анализ одной записи моделью."""
    verdict: str                        # anomaly | normal | uncertain
    rca: Any                            # строка или объект (формат задаёт промпт/add_info)
    confidence: int | None = None       # уверенность модели в вердикте, 0–100
    severity: str | None = None
    span_id: str | None = None
    business_description: str | None = None
    tech_details: str | None = None


def extract_items(text: str) -> list[dict]:
    """Список результатов из ответа: после </think>, в markdown-блоке или внутри текста."""
    text = text.rsplit('</think>', 1)[-1]
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char not in '{[':
            continue
        try:
            payload, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        while isinstance(payload, dict) and (key := next((k for k in _WRAPPERS if k in payload), None)):
            payload = payload[key]
        if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
            return payload
    raise ValueError(f'в ответе модели нет списка результатов: {text[:200]!r}')


def _present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None and value != {} and value != []


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _confidence(value: Any) -> int | None:
    if isinstance(value, str):
        try:
            value = float(value.strip().rstrip('%'))
        except ValueError:
            return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    value = value * 100 if 0 < value <= 1 and isinstance(value, float) else value
    return int(round(min(max(value, 0), 100)))


def parse_analysis(item: dict) -> Analysis | None:
    """Analysis из элемента ответа; None, если элемент не годится (запись уйдёт на повтор).

    Ответ в старом формате (запись целиком, без verdict, с rca_results) — это
    подтверждение аномалии: так работал прежний протокол «оставь аномальные».
    """
    rca = item.get('rca') if _present(item.get('rca')) else item.get('rca_results')
    raw_verdict = item.get('verdict')
    if raw_verdict is None:
        if not _present(rca):
            return None
        return Analysis(verdict='anomaly', rca=rca,
                        business_description=_text(item.get('business_description')),
                        tech_details=_text(item.get('tech_details')))
    verdict = _VERDICT_ALIASES.get(str(raw_verdict).strip().lower())
    if verdict is None or (verdict != 'normal' and not _present(rca)):
        return None
    span_id = item.get('span_id')
    return Analysis(
        verdict=verdict,
        rca=rca if _present(rca) else None,
        confidence=_confidence(item.get('confidence')),
        severity=_SEVERITY_ALIASES.get(str(item.get('severity', '')).strip().lower()),
        span_id=str(span_id).strip() if _present(span_id) and str(span_id).strip().lower() != 'null' else None,
        business_description=_text(item.get('business_description')),
        tech_details=_text(item.get('tech_details')),
    )


def match(items: list[dict], batch: list[int], records: list[dict]) -> dict[int, Analysis]:
    """Анализы по индексам записей пакета.

    Сопоставление — по `id` протокола; для ответов без id — по trace_id,
    если он в пакете единственный, или напрямую для пакета из одной записи.
    Чужие id, дубликаты и негодные элементы пропускаются.
    """
    by_id = {str(i): i for i in batch}
    by_trace: dict[Any, list[int]] = {}
    for i in batch:
        by_trace.setdefault(records[i].get('trace_id'), []).append(i)
    found: dict[int, Analysis] = {}
    for item in items:
        index = None
        if item.get('id') is not None:
            index = by_id.get(str(item['id']).strip())
        elif len(by_trace.get(item.get('trace_id'), ())) == 1:
            index = by_trace[item['trace_id']][0]
        elif len(batch) == 1 and len(items) == 1:
            index = batch[0]
        if index is None:
            logger.warning('RCA: модель вернула неизвестную запись id=%r trace_id=%r', item.get('id'), item.get('trace_id'))
            continue
        if index in found:
            continue
        analysis = parse_analysis(item)
        if analysis is None:
            logger.warning('RCA: негодный анализ записи id=%s (нет вердикта или причины)', index)
            continue
        found[index] = analysis
    return found
