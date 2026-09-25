"""Сигнал детектора (`detector_rca`, схема laim.detector_rca/1) -> доказательства для RCA.

Детектор laim объясняет каждую флагнутую запись: какая ветвь сработала
(поведенческая EPI — тайминги, длины, счётчики, структура шагов; смысловая
SEM — эмбеддинг текста шагов), какие шаги (спаны) и признаки отклонились
от нормы и насколько. Здесь это превращается в детерминированные
доказательства: сила сигнала, гипотеза причины, локализация и ограничения —
для промпта LLM, для итогового rca_results и для режима без LLM.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from laim_rca import glossary

SCHEMA_PREFIX = 'laim.detector_rca/'

# порог "ветвь аномальна": детектор сам центрирует логиты ветвей на z = 2
ELEVATED_Z = 2.0

CATEGORIES = {
    'technical_error': 'Техническая ошибка выполнения',
    'semantic':        'Смысловое отклонение (возможна галлюцинация)',
    'latency':         'Аномальные задержки',
    'llm_usage':       'Нетипичное использование LLM',
    'tools':           'Нетипичная работа с инструментами',
    'error_marker':    'Признаки ошибки в ответе',
    'output_form':     'Нетипичная форма ответа',
    'input_form':      'Нетипичный входной запрос или промпт',
    'structure':       'Нетипичная структура выполнения',
    'unknown':         'Причина по сигналу детектора не определена',
}
RECOMMENDATIONS = {
    'technical_error': 'Проверить логи и доступность сервиса или инструмента на этом шаге; при повторении завести инцидент.',
    'semantic':        'Проверить ответ агента на соответствие фактам и запросу: галлюцинации, нерелевантность, смешение продуктов.',
    'latency':         'Проверить производительность шага: таймауты, нагрузку, сетевые задержки, повторные попытки.',
    'llm_usage':       'Проверить параметры вызова LLM и объём передаваемого контекста.',
    'tools':           'Проверить вызовы инструментов: аргументы, ответы и статусы.',
    'error_marker':    'Проверить текст ответа на сообщения об ошибках и их первопричину.',
    'output_form':     'Проверить содержание и формат ответа на этом шаге.',
    'input_form':      'Проверить входной запрос или промпт шага: возможна инъекция или некорректный контекст.',
    'structure':       'Проверить сценарий выполнения: повторы шагов, зацикливание, лишние или пропущенные шаги.',
    'unknown':         'Требуется ручной анализ трассы.',
}
_FAMILY_CATEGORY = {
    'timing': 'latency', 'llm': 'llm_usage', 'tools': 'tools', 'errors': 'error_marker',
    'output_text': 'output_form', 'input_text': 'input_form', 'structure': 'structure', 'other': 'unknown',
}
_STRENGTH_RU = {'strong': 'сильный', 'moderate': 'умеренный', 'weak': 'слабый'}
_SIGNAL_RU = {'behavior': 'поведенческий (тайминги, длины, счётчики, структура шагов)',
              'semantic': 'смысловой (содержание текста шагов)',
              'mixed':    'смешанный (поведение и содержание текста)'}


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


@dataclass
class SpanEvidence:
    span_id: str | None
    roles: list[str]                    # behavior / semantic
    vs_typical: float | None            # ошибка шага относительно типичной у нормальных трасс
    details: dict                       # name, kind, status, duration_s, excerpts...
    drivers: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        name, kind = self.details.get('name'), self.details.get('kind')
        if name and kind and name != kind:
            return f'«{name}» ({kind})'
        return f'«{name or kind or self.span_id or "?"}»'

    def is_error(self) -> bool:
        status = str(self.details.get('status', '')).upper()
        http = _number(self.details.get('http_status_code'))
        return 'ERROR' in status or (http is not None and http >= 500)


@dataclass
class Evidence:
    agent_id: str | None = None
    p_anomaly: float | None = None
    margin: float | None = None         # (ошибка - порог) / порог
    z_behavior: float | None = None
    z_semantic: float | None = None
    behavior_share: float | None = None
    dominant: str | None = None         # behavior | semantic | mixed
    strength: str | None = None         # strong | moderate | weak
    families: list[tuple[str, float]] = field(default_factory=list)
    spans: list[SpanEvidence] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    category: str = 'unknown'
    hypothesis: str = ''
    caveats: list[str] = field(default_factory=list)
    span_ids: set[str] = field(default_factory=set)     # все шаги, известные детектору

    # --- представления -------------------------------------------------

    def prompt_view(self, max_spans: int = 3, max_features: int = 4, excerpt_chars: int = 200) -> dict:
        """Компактное объяснение детектора для промпта LLM."""
        view: dict[str, Any] = {}
        if self.agent_id is not None:
            view['agent_id'] = self.agent_id
        if self.p_anomaly is not None:
            view['p_anomaly'] = round(self.p_anomaly, 3)
        if self.strength:
            view['strength'] = _STRENGTH_RU[self.strength]
        if self.dominant:
            view['signal'] = _SIGNAL_RU[self.dominant]
        if self.hypothesis:
            view['hypothesis'] = self.hypothesis
        spans = []
        for span in self.spans[:max_spans]:
            item: dict[str, Any] = {'span_id': span.span_id}
            for key in ('name', 'kind', 'status', 'status_message', 'http_status_code', 'llm_model'):
                if key in span.details:
                    item[key] = span.details[key]
            if (duration := _number(span.details.get('duration_s'))) is not None:
                item['duration'] = glossary.format_value(duration * 1e9, 'ns')
            item['role'] = ', '.join('поведение' if r == 'behavior' else 'смысл' for r in span.roles)
            if span.vs_typical is not None:
                item['deviation'] = f'×{span.vs_typical:.1f} от типичной ошибки нормальных трасс'
            if span.drivers:
                item['why'] = span.drivers
            for key, short in (('input_excerpt', 'input'), ('output_excerpt', 'output')):
                if key in span.details:
                    item[short] = _cut(span.details[key], excerpt_chars)
            spans.append(item)
        if spans:
            view['suspicious_spans'] = spans
        if self.features:
            view['deviating_features'] = self.features[:max_features]
        if self.caveats:
            view['caveats'] = self.caveats
        return view

    def output_view(self) -> dict:
        """Сжатый сигнал детектора для итогового rca_results."""
        view: dict[str, Any] = {
            'p_anomaly': round(self.p_anomaly, 3) if self.p_anomaly is not None else None,
            'strength': self.strength,
            'signal': self.dominant,
            'category': CATEGORIES[self.category],
            'hypothesis': self.hypothesis,
        }
        if self.spans:
            view['top_spans'] = [
                {'span_id': s.span_id, 'span': s.title, 'role': s.roles, **({'why': s.drivers} if s.drivers else {})}
                for s in self.spans[:3]]
        if self.features:
            view['top_features'] = self.features[:4]
        if self.caveats:
            view['caveats'] = self.caveats
        return view

    def summary(self) -> str:
        """Одна строка для tech_details, когда LLM не участвовал."""
        parts = [self.hypothesis] if self.hypothesis else []
        if self.p_anomaly is not None:
            parts.append(f'Вероятность аномалии по детектору: {self.p_anomaly:.2f}.')
        if self.features:
            parts.append('Отклонения: ' + '; '.join(self.features[:3]) + '.')
        return ' '.join(parts)

    def location(self) -> dict | None:
        span = self.spans[0] if self.spans else None
        if span is None and self.agent_id is None:
            return None
        location: dict[str, Any] = {'agent_id': self.agent_id}
        if span is not None:
            location |= {'span_id': span.span_id, 'span_name': span.details.get('name'),
                         'span_kind': span.details.get('kind')}
        return location | {'source': 'detector'}

    def span(self, span_id: str) -> SpanEvidence | None:
        return next((s for s in self.spans if s.span_id == span_id), None)


def _cut(text: Any, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def _feature_phrase(view: dict, info: dict) -> str:
    """'длительность шага: 12.3 с при ожидаемых ~0.4 с (выше нормы)'."""
    name = view.get('feature', '?')
    base = info.get('base') or name
    _family, description, unit = glossary.describe(base)
    aggregation = glossary.aggregation_phrase(info.get('aggregation'), info.get('window'))
    label = f'{description} ({aggregation})' if aggregation else description
    direction = view.get('direction')
    trend = {'higher': 'выше нормы', 'lower': 'ниже нормы'}.get(direction, 'отклонение от нормы')
    observed, expected = _number(view.get('observed_raw')), _number(view.get('expected_raw'))
    if observed is None or expected is None:
        return f'{label}: {trend}'
    bound = ('≥ ' if direction == 'higher' else '≤ ') if view.get('clipped') else ''
    return (f'{label}: {bound}{glossary.format_value(observed, unit)} при ожидаемых '
            f'~{glossary.format_value(expected, unit)} ({trend})')


def _dominant(z_behavior: float | None, z_semantic: float | None, behavior_share: float | None) -> str | None:
    behavior = z_behavior is not None and z_behavior >= ELEVATED_Z
    semantic = z_semantic is not None and z_semantic >= ELEVATED_Z
    if behavior and semantic:
        return 'mixed'
    if behavior or semantic:
        return 'behavior' if behavior else 'semantic'
    if behavior_share is not None:      # ни одна ветвь не выделяется: решает раскладка флагующей ошибки
        return 'behavior' if behavior_share >= 0.65 else 'semantic' if behavior_share <= 0.35 else 'mixed'
    return None


def _strength(p_anomaly: float | None) -> str | None:
    if p_anomaly is None:
        return None
    return 'strong' if p_anomaly >= 0.8 else 'moderate' if p_anomaly >= 0.5 else 'weak'


def parse(record: dict) -> Evidence | None:
    """Evidence из record['detector_rca'] (dict или JSON-строка); None, если сигнала нет."""
    raw = record.get('detector_rca')
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, dict) or not str(raw.get('schema', '')).startswith(SCHEMA_PREFIX):
        return None

    scores = _dict(raw.get('scores'))
    z = _dict(scores.get('branch_z'))
    share = _dict(scores.get('flag_share'))
    error, threshold = _number(scores.get('reconstruction_error')), _number(scores.get('threshold'))
    evidence = Evidence(
        agent_id=raw.get('agent_id'),
        p_anomaly=_number(scores.get('p_anomaly')),
        margin=(error - threshold) / threshold if error is not None and threshold else None,
        z_behavior=_number(z.get('behavior')),
        z_semantic=_number(z.get('semantic')),
        behavior_share=_number(share.get('behavior')),
    )
    evidence.dominant = _dominant(evidence.z_behavior, evidence.z_semantic, evidence.behavior_share)
    evidence.strength = _strength(evidence.p_anomaly)

    info = _dict(raw.get('feature_info'))
    catalog = _dict(raw.get('spans'))
    evidence.span_ids = {str(k) for k in catalog}
    behavior = _dict(raw.get('behavior'))

    # признаки в порядке вклада в ошибку; семейства — по суммарной доле ошибки
    features = [f for f in _list(behavior.get('features')) if isinstance(f, dict)]
    families: dict[str, float] = {}
    for feature in features:
        family = glossary.describe(_dict(info.get(feature.get('feature'))).get('base') or feature.get('feature', ''))[0]
        families[family] = families.get(family, 0.0) + (_number(feature.get('error_share')) or 0.0)
    evidence.families = sorted(families.items(), key=lambda kv: -kv[1])
    evidence.features = [_feature_phrase(f, _dict(info.get(f.get('feature')))) for f in features]

    # шаги: поведенческие всегда, смысловые — когда смысловая ветвь значима
    spans: dict[str, SpanEvidence] = {}
    semantic_matters = evidence.dominant in ('semantic', 'mixed')
    for role, entries, limit in (('behavior', _list(behavior.get('spans')), 3),
                                 ('semantic', _list(_dict(raw.get('semantic')).get('spans')), 2 if semantic_matters else 0)):
        for entry in [e for e in entries if isinstance(e, dict)][:limit]:
            span_id = entry.get('span_id')
            key = str(span_id) if span_id is not None else f'{role}#{entry.get("index")}'
            if key not in spans:
                spans[key] = SpanEvidence(span_id=span_id, roles=[], vs_typical=_number(entry.get('vs_typical')),
                                          details=_dict(catalog.get(span_id)))
            span = spans[key]
            span.roles.append(role)
            if role == 'behavior':
                span.drivers = [_feature_phrase(d, _dict(info.get(d.get('feature'))))
                                for d in _list(entry.get('drivers')) if isinstance(d, dict)][:2]
    # шаг с ошибкой статуса — самая сильная улика, он идёт первым
    evidence.spans = sorted(spans.values(), key=lambda s: not s.is_error())

    evidence.category, evidence.hypothesis = _hypothesis(evidence)
    evidence.caveats = _caveats(evidence, raw, features, info)
    return evidence


def _hypothesis(evidence: Evidence) -> tuple[str, str]:
    """Детерминированная гипотеза причины по сигналу детектора."""
    failed = next((s for s in evidence.spans if s.is_error()), None)
    if failed is not None:
        message = failed.details.get('status_message')
        return 'technical_error', (f'Шаг {failed.title} завершился ошибкой'
                                   + (f': {_cut(message, 200)}' if message else '') + '.')
    if evidence.dominant == 'semantic':
        span = next((s for s in evidence.spans if 'semantic' in s.roles), None)
        where = f' шага {span.title}' if span else ''
        return 'semantic', (f'Содержание текста{where} нетипично для нормальной работы агента: '
                            f'проверить на галлюцинацию, нерелевантный или некорректный ответ.')
    family = evidence.families[0][0] if evidence.families else None
    category = _FAMILY_CATEGORY.get(family or 'other', 'unknown')
    if category == 'unknown' and not evidence.features:
        return 'unknown', ''
    text = CATEGORIES[category]
    if evidence.features:
        text += f': {evidence.features[0]}'
    span = next((s for s in evidence.spans if 'behavior' in s.roles), None)
    if span is not None:
        duration = _number(span.details.get('duration_s'))
        text += (f'. Сильнее всего отклоняется шаг {span.title}'
                 + (f', длительность {glossary.format_value(duration * 1e9, "ns")}' if duration is not None else ''))
    if evidence.dominant == 'mixed':
        text += '. Отклоняется и содержание текста — проверить ответ на галлюцинацию'
    return category, text + '.'


def _caveats(evidence: Evidence, raw: dict, features: list[dict], info: dict) -> list[str]:
    caveats = []
    sequence = _dict(raw.get('sequence'))
    if sequence.get('truncated'):
        caveats.append(f'детектор оценил только первые {sequence.get("scored")} из {sequence.get("spans")} шагов')
    if evidence.strength == 'weak':
        caveats.append(f'калиброванная вероятность аномалии низкая ({evidence.p_anomaly:.2f}): '
                       f'запись прошла порог, но вероятно это ложное срабатывание')
    if features and all(_dict(info.get(f.get('feature'))).get('scope') == 'sequence' for f in features[:3]):
        caveats.append('отклонение касается всей последовательности шагов агента, а не отдельного шага')
    drivers = [d for s in _list(_dict(raw.get('behavior')).get('spans')) if isinstance(s, dict)
               for d in _list(s.get('drivers')) if isinstance(d, dict)]
    if any(view.get('clipped') for view in (*features, *drivers)):
        caveats.append('часть значений упёрлась в границу нормализации: реальное отклонение больше показанного')
    if evidence.spans and not any(s.details for s in evidence.spans):
        caveats.append('детали шагов недоступны')
    return caveats
