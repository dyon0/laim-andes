"""Общие фикстуры: фейковая LLM вместо SdsChatModel/GigaChat, записи детектора."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as rca  # noqa: E402

FIXTURES = Path(__file__).parent / 'fixtures'


class Reply:
    def __init__(self, content: str, finish_reason: str = 'stop'):
        self.content = content
        self.response_metadata = {'finish_reason': finish_reason}


def sent_views(messages) -> list[dict]:
    """Записи пакета из запроса: всё после TRACES_DATA — JSON {"anomalies": [...]}."""
    return json.loads(messages[-1][1].split('TRACES_DATA:\n', 1)[1])['anomalies']


def results(items: list[dict]) -> Reply:
    return Reply(json.dumps({'results': items}, ensure_ascii=False))


def verdicts(messages, verdict: str = 'anomaly', **extra) -> Reply:
    """Модель-эхо протокола: по каждой записи вердикт и RCA = 'причина <trace_id>'."""
    return results([{'id': v['id'], 'verdict': verdict, 'confidence': 90, 'severity': 'high',
                     'rca': f"причина {v['trace_id']}", **extra} for v in sent_views(messages)])


class FakeLlm:
    """Подменяет SdsChatModel и GigaChat: пишет вызовы, ответ строит behavior(messages)."""

    kwargs: dict = {}
    calls: list = []
    sleeps: list = []
    behavior = staticmethod(verdicts)

    def __init__(self, **kwargs):
        FakeLlm.kwargs = kwargs

    def invoke(self, messages):
        FakeLlm.calls.append(messages)
        return FakeLlm.behavior(messages)


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    monkeypatch.setenv('AI_GATEWAY_URL', 'http://sds-ai-gateway:8097')
    monkeypatch.delenv('TIMEOUT', raising=False)
    monkeypatch.setattr(rca, 'SdsChatModel', FakeLlm)
    monkeypatch.setattr(rca, 'GigaChat', FakeLlm)
    monkeypatch.setattr(rca, '_sleep', lambda seconds: FakeLlm.sleeps.append(seconds))
    FakeLlm.kwargs, FakeLlm.calls, FakeLlm.sleeps = {}, [], []
    FakeLlm.behavior = staticmethod(verdicts)
    return FakeLlm


def anomaly(i: int, **fields) -> dict:
    return {'trace_id': f't{i}', 'anomaly_type': 'hallucination', 'confidence': 90,
            'user_query': f'вопрос {i}', 'agent_response': f'ответ {i}', 'rca_results': ''} | fields


def detector_rca(*, agent: str = 'agent-1', p: float = 0.93, z_behavior: float = 6.0, z_semantic: float = 0.5,
                 share: float = 0.8, spans: dict | None = None, behavior_spans: list | None = None,
                 features: list | None = None, semantic_spans: list | None = None,
                 info: dict | None = None, truncated: bool = False) -> dict:
    """Синтетический detector_rca (laim.detector_rca/1): по умолчанию — задержка вызова инструмента."""
    spans = spans if spans is not None else {
        's-tool': {'name': 'get_rate', 'kind': 'tool', 'status': 'STATUS_CODE_OK', 'duration_s': 12.3,
                   'input_excerpt': '{"product": "кредит"}', 'output_excerpt': '{"rate": 22.5}'},
        's-llm': {'name': 'answer', 'kind': 'llm', 'status': 'STATUS_CODE_OK', 'duration_s': 2.1,
                  'output_excerpt': 'Ставка 22,5% годовых'}}
    info = info if info is not None else {
        'tool_duration': {'base': 'tool_duration', 'aggregation': None, 'window': None, 'log1p': True, 'scope': 'step'},
        'duration_max': {'base': 'duration', 'aggregation': 'max', 'window': None, 'log1p': True, 'scope': 'sequence'},
        'char_count_rolling_mean_w5': {'base': 'char_count', 'aggregation': 'rolling_mean', 'window': 5,
                                       'log1p': True, 'scope': 'step'}}
    driver = {'feature': 'tool_duration', 'direction': 'higher', 'observed': 23.2, 'expected': 19.8,
              'observed_raw': 1.23e10, 'expected_raw': 4.0e8, 'error': 3.1}
    return {
        'schema': 'laim.detector_rca/1',
        'agent_id': agent,
        'scores': {'p_anomaly': p, 'reconstruction_error': 0.6, 'threshold': 0.4,
                   'branch_z': {'behavior': z_behavior, 'semantic': z_semantic},
                   'flag_share': {'behavior': share, 'semantic': round(1 - share, 6)},
                   'logit': {'behavior': 2.0, 'semantic': -0.3, 'combined': 0.7, 'bias': 0.0}},
        'sequence': {'spans': 40 if truncated else 12, 'scored': 30 if truncated else 12, 'truncated': truncated},
        'spans': spans,
        'feature_info': info,
        'behavior': {
            'spans': behavior_spans if behavior_spans is not None else [
                {'rank': 1, 'index': 3, 'span_id': 's-tool', 'error': 2.5, 'error_share': 0.4,
                 'vs_typical': 14.2, 'drivers': [driver]},
                {'rank': 2, 'index': 5, 'span_id': 's-llm', 'error': 0.9, 'error_share': 0.15,
                 'vs_typical': 4.1, 'drivers': []}],
            'features': features if features is not None else [
                {'rank': 1, **driver, 'error_share': 0.45, 'vs_typical': 20.0,
                 'peak_span_index': 3, 'peak_span_id': 's-tool'},
                {'rank': 2, 'feature': 'duration_max', 'direction': 'higher', 'observed': 23.4, 'expected': 21.0,
                 'observed_raw': 1.45e10, 'expected_raw': 1.3e9, 'error': 1.2, 'error_share': 0.2,
                 'vs_typical': 6.0, 'peak_span_index': 3, 'peak_span_id': 's-tool'}]},
        'semantic': {'spans': semantic_spans if semantic_spans is not None else [
            {'rank': 1, 'index': 5, 'span_id': 's-llm', 'error': 0.01, 'error_share': 0.3, 'vs_typical': 1.2}]},
    }


def load_fixture(name: str = 'laim_test_anomalies.json') -> dict:
    return json.loads((FIXTURES / name).read_text(encoding='utf-8'))
