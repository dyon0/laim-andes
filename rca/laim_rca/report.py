"""Итог RCA: решение по записи, rca_results, аудит.

Вердикты:
- anomaly    — LLM подтвердила аномалию: запись в выходе;
- normal     — LLM сочла срабатывание ложным: запись только в аудите;
- uncertain  — LLM не уверена: в выходе, если keep_uncertain;
- unverified — LLM запись не проанализировала (сбой, режим detector_only):
               в выходе с RCA по сигналу детектора, если keep_uncertain
               (в detector_only — всегда: без LLM фильтровать нечем).
"""
from __future__ import annotations

import json
import re
from typing import Any

from laim_rca.answers import Analysis
from laim_rca.evidence import CATEGORIES, RECOMMENDATIONS, Evidence

AUDIT_SCHEMA = 'laim.rca_audit/1'
_NARRATIVE = ('business_description',)


def detector_rca(evidence: Evidence | None) -> dict:
    """RCA без LLM — по сигналу детектора."""
    if evidence is None:
        return {'root_cause': 'Анализ не выполнен: у записи нет объяснения детектора (detector_rca), LLM не использовалась.',
                'evidence': [], 'recommendation': RECOMMENDATIONS['unknown']}
    facts = [f'{s.title}: ' + '; '.join(s.drivers) for s in evidence.spans[:3] if s.drivers] + evidence.features[:3]
    return {'root_cause': evidence.hypothesis or CATEGORIES['unknown'],
            'evidence': facts,
            'recommendation': RECOMMENDATIONS[evidence.category]}


def _location(analysis: Analysis | None, evidence: Evidence | None) -> dict | None:
    """Где причина: шаг, указанный моделью (если он известен детектору), иначе самый отклоняющийся шаг."""
    if analysis is not None and analysis.span_id and evidence is not None and (
            analysis.span_id in evidence.span_ids or evidence.span(analysis.span_id) is not None):
        span = evidence.span(analysis.span_id)
        details = span.details if span is not None else {}
        return {'agent_id': evidence.agent_id, 'span_id': analysis.span_id,
                'span_name': details.get('name'), 'span_kind': details.get('kind'), 'source': 'llm'}
    return evidence.location() if evidence is not None else None


_HEX_ID = re.compile(r'\b[0-9a-fA-F]{8,64}\b')


def _mentioned_traces(rca: Any, known: list[str], own: Any) -> list[str]:
    """trace_id других записей, на которые ссылается причина (целиком или по
    префиксу от 8 символов: «6fdbccc1…») — для перекрёстных ссылок в отчёте."""
    text = rca if isinstance(rca, str) else json.dumps(rca, ensure_ascii=False)
    found = []
    for token in _HEX_ID.findall(text):
        match = next((t for t in known if t.lower().startswith(token.lower())), None)
        if match and match != str(own) and match not in found:
            found.append(match)
    return found


def _agreement(verdict: str, evidence: Evidence | None) -> str | None:
    """Согласие LLM с силой сигнала детектора — обратная связь для детектора."""
    if evidence is None or evidence.strength is None or verdict not in ('anomaly', 'normal'):
        return None
    detector_says_anomaly = evidence.strength in ('strong', 'moderate')
    return 'agrees' if (verdict == 'anomaly') == detector_says_anomaly else 'disagrees'


def _reason(rca: Any) -> str:
    """Короткая причина для аудита."""
    if isinstance(rca, dict):
        for key in ('root_cause', 'error_source', 'reason', 'anomaly_category', 'category'):
            if isinstance(rca.get(key), str) and rca[key].strip():
                return rca[key].strip()[:300]
        return json.dumps(rca, ensure_ascii=False)[:300]
    return str(rca or '').strip()[:300]


def assemble(records: list[dict], evidences: list[Evidence | None], analyses: dict[int, Analysis], *,
             analyzed_by: str, keep_uncertain: bool, llm_used: bool) -> tuple[list[dict], list[dict]]:
    """(выходные записи в исходном порядке, решения по всем записям для аудита)."""
    output, decisions = [], []
    known_traces = [str(r.get('trace_id')) for r in records if r.get('trace_id')]
    for index, (record, evidence) in enumerate(zip(records, evidences)):
        analysis = analyses.get(index)
        verdict = analysis.verdict if analysis is not None else 'unverified'
        kept = (verdict == 'anomaly' or not llm_used
                or (keep_uncertain and verdict in ('uncertain', 'unverified')))
        if analysis is None:
            rca = detector_rca(evidence)
        else:
            rca = analysis.rca if analysis.rca is not None else 'Модель сочла срабатывание детектора ложным.'

        results: dict[str, Any] = {
            'verdict': verdict,
            'verdict_confidence': analysis.confidence if analysis is not None else None,
            'severity': analysis.severity if analysis is not None else None,
            'rca': rca,
        }
        if (location := _location(analysis, evidence)) is not None:
            results['location'] = location
        if (mentioned := _mentioned_traces(rca, known_traces, record.get('trace_id'))):
            results['related_traces'] = mentioned
        if evidence is not None:
            results['detector_evidence'] = evidence.output_view()
        results['analyzed_by'] = analyzed_by if analysis is not None else 'detector'

        if kept:
            out = {key: value for key, value in record.items() if key != 'detector_rca'}
            # детектор оставляет описания пустыми — их заполняет анализ, заданные не трогаем
            # tech_details не заполняется: технические детали — часть RCA, отдельное поле их дублировало
            fills = {'business_description': analysis.business_description if analysis else None}
            for key in _NARRATIVE:
                if fills[key] and not str(out.get(key) or '').strip():
                    out[key] = fills[key]
            out['rca_results'] = results
            output.append(out)

        decisions.append({
            'index': index,
            'trace_id': record.get('trace_id'),
            'agent_id': evidence.agent_id if evidence is not None else None,
            'decision': 'kept' if kept else 'dropped',
            'verdict': verdict,
            'verdict_confidence': results['verdict_confidence'],
            'severity': results['severity'],
            'reason': _reason(rca),
            'detector': ({'p_anomaly': evidence.p_anomaly, 'strength': evidence.strength,
                          'category': evidence.category} if evidence is not None else None),
            'agreement': _agreement(verdict, evidence),
        })
    return output, decisions


def audit(decisions: list[dict], *, mode: str, model_id: str, use_detector_evidence: bool,
          keep_uncertain: bool, llm: dict) -> dict:
    counts = {'input': len(decisions),
              'with_detector_evidence': sum(d['detector'] is not None for d in decisions),
              'output': sum(d['decision'] == 'kept' for d in decisions)}
    for verdict in ('anomaly', 'normal', 'uncertain', 'unverified'):
        counts[verdict] = sum(d['verdict'] == verdict for d in decisions)
    agreement = [d['agreement'] for d in decisions if d['agreement'] is not None]
    return {
        'schema': AUDIT_SCHEMA,
        'mode': mode,
        'model_id': model_id,
        'use_detector_evidence': use_detector_evidence,
        'keep_uncertain': keep_uncertain,
        'counts': counts,
        'detector_agreement': {'agrees': agreement.count('agrees'), 'disagrees': agreement.count('disagrees')},
        'llm': llm,
        'decisions': decisions,
    }
