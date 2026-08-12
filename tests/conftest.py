import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SAMPLE = REPO / 'data' / 'traces_1k_sample.parquet'
GOLDEN = Path(__file__).parent / 'golden' / 'golden.json'


@pytest.fixture(scope='session')
def sample_spans() -> pl.DataFrame:
    return pl.read_parquet(SAMPLE)


@pytest.fixture(scope='session')
def fixture_spans(sample_spans: pl.DataFrame) -> pl.DataFrame:
    """Deterministic 20-trace subset of the real sample (sorted trace_ids)."""
    ids = sample_spans.select('trace_id').unique().sort('trace_id').head(20)
    return sample_spans.join(ids, on='trace_id', how='semi').sort('trace_id', 'start_time_ns')


@pytest.fixture(scope='session')
def golden() -> dict:
    import json
    if not GOLDEN.exists():
        pytest.skip('golden.json not generated — run `make golden`')
    return json.loads(GOLDEN.read_text())


def _valid_span_row() -> dict:
    """One spec-conformant non-LLM span (sentinels where mandated)."""
    return {
        'trace_id': 'dHJhY2Ux', 'span_id': 'c3BhbjE=', 'parent_span_id': 'root',
        'origin_span_id': 'outside', 'agent_id': 'CI1', 'session_id': 'c2Vzc2lvbjE=',
        'start_time_ns': 1_000_000_000, 'end_time_ns': 2_000_000_000,
        'status_code': 'STATUS_CODE_OK', 'status_message': '',
        'aef_kind': 'chain', 'span_name': 'chain.step',
        'input_text': '{"x": 1}', 'output_text': 'ok',
        'llm_model': '', 'llm_prompt_tokens': -1, 'llm_completion_tokens': -1,
        'llm_total_tokens': -1, 'llm_precached_prompt_tokens': -1,
        'llm_temperature': -1.0, 'llm_top_p': -1.0, 'llm_max_tokens': -1,
        'llm_repetition_penalty': -1.0, 'llm_profanity_check': False, 'llm_stream': False,
        'http_method': 'NONE', 'http_path': '', 'http_status_code': -1,
        'request_headers': '', 'response_headers': '',
        'kafka_topic': '', 'kafka_cluster': '', 'kafka_consumer_group': '',
        'kafka_bootstrap_servers': '',
        'meta_langgraph_step': -1, 'meta_langgraph_node': '',
        'meta_langgraph_triggers': '', 'meta_langgraph_path': '',
        'meta_checkpoint_ns': '', 'meta_tags': '', 'meta_extra': '',
        'service_name': 'svc', 'service_version': '1.0.0',
        'session_id_derived': False, 'session_id_generated': False,
        'nexus_distrib_ver': '1.2.3',
    }


@pytest.fixture(scope='session')
def valid_span_row() -> dict:
    return _valid_span_row()


@pytest.fixture(scope='session')
def malformed_spans() -> dict[str, pl.DataFrame]:
    """Named single-span frames, each violating exactly one contract rule."""
    from ars.specification.spec import SpansData

    def frame(**overrides) -> pl.DataFrame:
        row = {**_valid_span_row(), **overrides}
        return pl.DataFrame([row]).cast(dict(SpansData.schema), strict=False)

    return {
        'bad_trace_id_not_base64': frame(trace_id='not base64!'),
        'time_reversed': frame(end_time_ns=500_000_000),
        'negative_start': frame(start_time_ns=-5),
        'bad_agent_id': frame(agent_id='WRONG'),
        'error_without_message': frame(status_code='STATUS_CODE_ERROR', status_message=''),
        'llm_temp_on_non_llm': frame(llm_temperature=0.7),
        'both_session_flags': frame(session_id_derived=True, session_id_generated=True),
        'null_mandatory_session': frame(session_id=None),
        'bad_nexus_version': frame(nexus_distrib_ver='not-a-version'),
        'http_status_out_of_range': frame(
            aef_kind='input_request', http_method='GET', http_path='/x',
            request_headers='{"a": "b"}', response_headers='{"a": "b"}',
            http_status_code=999),
    }
