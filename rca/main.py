"""Нода LAIM RCA (SberDS): фильтрация аномалий детектора laim и поиск первопричин.

Вход `anom_data` — `test_anomalies` детектора: {"anomalies": [...]}, где у
флагнутых записей есть `detector_rca` — объяснение детектора (какая ветвь
сработала, какие шаги и признаки отклонились и насколько). Нода:

1. превращает detector_rca в доказательства (laim_rca.evidence): гипотезу
   причины, подозрительные шаги с фрагментами, отклонения в натуральных
   единицах, силу сигнала и его ограничения;
2. отдаёт записи LLM пакетами; модель по каждой записи возвращает только
   анализ — вердикт (anomaly | normal | uncertain), причину, шаг, severity —
   и не пересказывает записи;
3. собирает выход: подтверждённые аномалии (и, при keep_uncertain,
   неуверенные/непроверенные) с заполненным rca_results; решения по ВСЕМ
   записям, включая отфильтрованные, уходят в аудит.

Выходы: `res` — JSON {"anomalies": [...]} для сборщика отчёта (поля записей
детектора не меняются, кроме пустых business_description/tech_details и
rca_results; сырой detector_rca из выхода убирается), `rca_audit` — JSON
с решениями и статистикой прогона.
"""
from __future__ import annotations

import json
import logging
import math
import time
import zipfile
from collections import deque
from typing import Any
from xml.etree import ElementTree

import httpx
import requests
from gigachat.exceptions import GigaChatException
from langchain_gigachat import GigaChat

from laim_rca import agent_report
from laim_rca import evidence as detector_evidence
from laim_rca.agent_report import AgentReport
from laim_rca.answers import Analysis, extract_items, match
from laim_rca.prompt import record_view, system_prompt, user_message
from laim_rca.related import find_related, processing_order, related_view
from laim_rca.report import assemble, audit
from llm.config import ModelsConfig
from llm.sds_chat_model import DEFAULT_TIMEOUT_SECONDS, SdsChatModel

logger = logging.getLogger(__name__)

MODES = ('llm', 'llm_fallback', 'detector_only')

# 8 записей GigaChat размечал за 35–50 с, когда модель пересказывала пакет;
# без пересказа ответ короче, лимит по байтам учитывает detector_evidence.
BATCH_ITEMS = 8
BATCH_BYTES = 32_000
GROW_AFTER_SUCCESSES = 3
SINGLE_RECORD_ATTEMPTS = 2      # сбоев модели на одиночной записи до отказа от неё
MAX_RECORD_MISSES = 3           # раз, когда модель пропустила запись в успешном ответе
MAX_TRANSPORT_FAILURES = 3
RATE_LIMIT_RETRIES = 5          # подряд 429/503 до перехода к обычной обработке сбоя
BACKOFF_SECONDS = 5.0
BACKOFF_MAX_SECONDS = 60.0
TRANSPORT_ERRORS = (requests.ConnectionError, requests.Timeout, httpx.TransportError)
# Отказ модели на пакете: пакет дробится, одиночная запись после повторов
# остаётся непроверенной (unverified). Ошибки программы не перехватываются.
MODEL_ERRORS = TRANSPORT_ERRORS + (
    requests.HTTPError, GigaChatException, RuntimeError, ValueError,
)

_sleep = time.sleep     # подменяется в тестах


class LlmUnavailable(RuntimeError):
    """LLM недоступна: ни один запрос не прошёл или шлюз перестал отвечать."""

    def __init__(self, message: str, cause: BaseException | None = None):
        super().__init__(message)
        self.__cause__ = cause      # исходная ошибка транспорта/модели видна в traceback


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _finite(value: Any) -> Any:
    """NaN/Inf из входного JSON -> None: иначе запись нельзя сериализовать обратно."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite(v) for v in value]
    return value


def _parse_input(anom_data: Any) -> list[dict]:
    """Записи детектора из JSON-строки, bytes, dict {"anomalies": [...]} или списка."""
    if isinstance(anom_data, (bytes, bytearray)):
        anom_data = bytes(anom_data).decode("utf-8-sig")
    if isinstance(anom_data, str):
        text = anom_data.lstrip("﻿").strip().removeprefix("root:")
        try:
            anom_data = json.loads(text)
            if isinstance(anom_data, str):
                anom_data = json.loads(anom_data)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"anom_data содержит невалидный JSON: строка {error.lineno}, столбец {error.colno}"
            ) from error
    records = anom_data.get("anomalies") if isinstance(anom_data, dict) else anom_data
    if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
        raise ValueError("anom_data.anomalies должен быть списком объектов")
    return [_finite(r) for r in records]


def _flag(value: Any, default: bool) -> bool:
    """Булев параметр ноды: платформа может прислать bool, число или строку."""
    if value is None or value == '':
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'y', 'да', 'on')
    return bool(value)


def _mode(value: Any) -> str:
    mode = str(value or 'llm').strip().lower()
    if mode not in MODES:
        raise ValueError(f"mode должен быть одним из {MODES}, получено {value!r}")
    return mode


def _build_model(model_id: str, llm_temp: float, max_tokens: int) -> SdsChatModel | GigaChat:
    config = ModelsConfig(model=model_id)
    if model_id.lower().startswith("giga"):
        return GigaChat(**(config.contour_llm_configs | {
            "temperature": llm_temp,
            "max_tokens": max_tokens,
            # без явного значения клиент gigachat ждёт ответ всего 30 с
            "timeout": float(config.llm_params["timeout"] or DEFAULT_TIMEOUT_SECONDS),
        }))
    return SdsChatModel(
        base_url=config.contour_configs.get("base_url"),
        model_id=model_id,
        temperature=llm_temp,
        max_tokens=max_tokens,
        top_p=config.llm_params["top_p"],
        timeout=config.llm_params["timeout"],
        verify_ssl_certs=config.verify_ssl_certs,
    )


def _load_agent_report(source: Any, max_chars: Any) -> tuple[AgentReport | None, dict | None]:
    """Опциональный контекст: нет данных — нет контекста; нечитаемые данные не
    роняют ноду (отчёт — обогащение), причина уходит в аудит и лог."""
    try:
        report = agent_report.load(source, max_chars=max(1000, int(float(max_chars or 20_000))))
    except (ValueError, OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as error:
        logger.warning("RCA: отчёт о разработке не прочитан, анализ без него: %s", error)
        return None, {'error': f'{type(error).__name__}: {error}'}
    return report, (report.audit() if report is not None else None)


def _trace_grouped(records: list[dict]) -> list[int]:
    """Порядок обработки: записи одной трассы (разные агенты) — подряд, чтобы
    попадать в один пакет и анализироваться вместе."""
    groups: dict[Any, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record.get("trace_id", f"#{index}"), []).append(index)
    return [index for group in groups.values() for index in group]


def _take_batch(pending: deque[int], sizes: list[int], limit: int) -> list[int]:
    """Следующий пакет: не больше limit записей и BATCH_BYTES (одна запись — всегда)."""
    batch, size = [], 2
    while pending and len(batch) < limit:
        item_size = sizes[pending[0]]
        if batch and size + item_size > BATCH_BYTES:
            break
        batch.append(pending.popleft())
        size += item_size
    return batch


def _throttle_delay(error: Exception, attempt: int) -> float | None:
    """Пауза перед повтором при 429/503 (с учётом Retry-After); None — не троттлинг."""
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None) if response is not None else getattr(error, "status_code", None)
    headers = getattr(response, "headers", None) if response is not None else getattr(error, "headers", None)
    if status not in (429, 503):
        return None
    try:
        retry_after = float((headers or {}).get("retry-after") or 0)
    except (TypeError, ValueError):
        retry_after = 0.0
    backoff = min(BACKOFF_SECONDS * 2 ** attempt, BACKOFF_MAX_SECONDS)
    return min(max(retry_after, backoff), 2 * BACKOFF_MAX_SECONDS)


def _ask(model: SdsChatModel | GigaChat, system: str, views: list[dict]) -> list[dict]:
    response = model.invoke([("system", system), ("human", user_message(views))])
    if response.response_metadata.get("finish_reason") in {"length", "max_tokens"}:
        raise ValueError("ответ модели обрезан по лимиту токенов")
    return extract_items(response.content)


def _analyze(records: list[dict], views: list[dict], model: SdsChatModel | GigaChat, model_id: str,
             system: str, stats: dict, order: list[int] | None = None) -> tuple[dict[int, Analysis], LlmUnavailable | None]:
    """Адаптивные пакеты: сбой — пакет делится пополам, успехи — размер растёт
    обратно; пропущенные моделью записи возвращаются в очередь."""
    sizes = [len(_dump(view).encode("utf-8")) + 1 for view in views]
    pending = deque(order if order is not None else _trace_grouped(records))
    analyses: dict[int, Analysis] = {}
    attempts: dict[int, int] = {}
    misses: dict[int, int] = {}
    limit, successes, transport_failures, throttled = BATCH_ITEMS, 0, 0, 0
    last_error: Exception | None = None
    any_success = False

    while pending:
        batch = _take_batch(pending, sizes, limit)
        started = time.monotonic()
        stats["requests"] += 1
        try:
            found = match(_ask(model, system, [views[i] for i in batch]), batch, records)
            if not found:
                # ответ разобран, но годного анализа нет ни по одной записи — это сбой, а не успех
                raise ValueError("в ответе модели нет годного анализа ни одной записи пакета")
        except MODEL_ERRORS as error:
            stats["failed_requests"] += 1
            last_error, successes = error, 0
            logger.warning("RCA: пакет из %d записей не прошёл: %s", len(batch), error)
            delay = _throttle_delay(error, throttled)
            if delay is not None and throttled < RATE_LIMIT_RETRIES:
                throttled += 1
                stats["throttled"] += 1
                logger.warning("RCA: шлюз просит подождать (%s), пауза %.0f с", type(error).__name__, delay)
                _sleep(delay)
                pending.extendleft(reversed(batch))
                continue
            throttled = 0
            if len(batch) > 1:
                limit = len(batch) // 2
                pending.extendleft(reversed(batch))
                continue
            index = batch[0]
            attempts[index] = attempts.get(index, 0) + 1
            transport_failures = transport_failures + 1 if isinstance(error, TRANSPORT_ERRORS) else 0
            if transport_failures >= MAX_TRANSPORT_FAILURES:
                return analyses, LlmUnavailable(
                    f"Модель {model_id} не отвечает: {transport_failures} одиночных запросов "
                    f"подряд завершились ошибкой соединения: {error}", error)
            if attempts[index] < SINGLE_RECORD_ATTEMPTS:
                pending.appendleft(index)
            else:
                logger.warning("RCA: запись trace_id=%r не проанализирована моделью", records[index].get("trace_id"))
            continue

        any_success, transport_failures, throttled = True, 0, 0
        analyses.update(found)
        missing = [i for i in batch if i not in found]
        for index in reversed(missing):
            misses[index] = misses.get(index, 0) + 1
            if misses[index] < MAX_RECORD_MISSES:
                pending.appendleft(index)
            else:
                logger.warning("RCA: модель %d раз пропустила запись trace_id=%r", misses[index],
                               records[index].get("trace_id"))
        logger.info("RCA: пакет %d записей за %.1f с, разобрано %d", len(batch), time.monotonic() - started, len(found))
        successes += 1
        if successes >= GROW_AFTER_SUCCESSES and limit < BATCH_ITEMS:
            limit, successes = min(limit * 2, BATCH_ITEMS), 0

    if records and not any_success:
        return analyses, LlmUnavailable(f"RCA: ни один запрос к модели {model_id} не прошёл: {last_error}", last_error)
    return analyses, None


def main(
    anom_data: Any,
    model_id: str = 'minimax-m2.5',
    add_info: str = ' ',
    llm_temp: float = 0.001,
    max_tokens: float = 8192,
    mode: str = 'llm',
    use_detector_evidence: bool = True,
    keep_uncertain: bool = True,
    agent_report: Any = None,
    report_max_chars: int = 20_000,
    evidence_detail: str = 'brief',
) -> dict[str, str]:
    """mode: llm — анализ LLM, сбой LLM роняет ноду; llm_fallback — при
    недоступности LLM RCA по сигналу детектора; detector_only — без LLM.
    use_detector_evidence — передавать ли LLM объяснение детектора.
    keep_uncertain — оставлять ли в выходе записи с вердиктом uncertain и
    непроверенные моделью (с RCA по сигналу детектора).
    agent_report — ОПЦИОНАЛЬНЫЙ порт: отчёт о разработке агента (.docx,
    текст или выход g-aiva-doc-browser); если подан — становится контекстом
    анализа (не более report_max_chars символов).
    evidence_detail — brief: LLM видит из сигнала детектора только вероятность и
    подозрительные шаги с фрагментами; full — ещё признаки, отклонения и гипотезу."""
    records = _parse_input(anom_data)
    mode, model_id = _mode(mode), str(model_id).strip()
    with_evidence = _flag(use_detector_evidence, True)
    keep = _flag(keep_uncertain, True)
    evidences = [detector_evidence.parse(record) for record in records]
    context, context_audit = _load_agent_report(agent_report, report_max_chars)

    stats: dict[str, Any] = {"requests": 0, "failed_requests": 0, "throttled": 0,
                             "elapsed_s": 0.0, "fallback": None}
    analyses: dict[int, Analysis] = {}
    if records and mode != 'detector_only':
        started = time.monotonic()
        failure: LlmUnavailable | None = None
        try:
            model = _build_model(model_id, float(llm_temp), int(float(max_tokens)))
        except (ValueError, RuntimeError, GigaChatException) as error:
            if mode == 'llm':
                raise
            failure = LlmUnavailable(f"RCA: модель {model_id} недоступна: {error}", error)
        if failure is None:
            # связи между записями — по всему входу: доказательства часто в соседних трейсах
            links = find_related(records)
            detail = 'full' if str(evidence_detail).strip().lower() == 'full' else 'brief'
            views = [record_view(i, r, e, with_evidence, related_view(records, links[i]), detail)
                     for i, (r, e) in enumerate(zip(records, evidences))]
            system = system_prompt(add_info, with_evidence, context.text if context else None)
            analyses, failure = _analyze(records, views, model, model_id, system, stats,
                                         processing_order(records, links))
        stats["elapsed_s"] = round(time.monotonic() - started, 1)
        if failure is not None:
            if mode == 'llm':
                raise failure
            stats["fallback"] = str(failure)
            logger.warning("RCA: %s — записи без анализа получают RCA по сигналу детектора", failure)

    llm_used = bool(analyses) or (mode != 'detector_only' and stats["requests"] > stats["failed_requests"])
    output, decisions = assemble(
        records, evidences, analyses,
        analyzed_by=f"llm:{model_id}", keep_uncertain=keep, llm_used=llm_used)
    report = audit(decisions, mode=mode, model_id=model_id, use_detector_evidence=with_evidence,
                   keep_uncertain=keep, llm=stats)
    report['agent_report'] = context_audit
    logger.info("RCA: записей на входе %d, в выходе %d (%s)", len(records), len(output), report["counts"])
    return {'res': _dump({'anomalies': output}), 'rca_audit': _dump(report)}
