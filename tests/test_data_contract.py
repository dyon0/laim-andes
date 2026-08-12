"""Data-contract tests against the spec (ars/specification) and the validation
engine (ars/data/validation.py — currently NOT wired into the pipeline, F-34;
these tests pin the engine itself so wiring it in Phase 5 is safe).
"""
import polars as pl
import pytest

from ars.specification.common_core import Sentinel
from ars.specification.spec import SpansData, recast


def _validity(df: pl.DataFrame) -> dict[str, bool]:
    out = {}
    for name, expr in SpansData.validities():
        out[name] = bool(df.select(expr.alias('v'))['v'].fill_null(False).all())
    return out


def test_valid_span_passes_all_validities(valid_span_row):
    df = pl.DataFrame([valid_span_row]).cast(dict(SpansData.schema), strict=False)
    v = _validity(df)
    failed = [k for k, ok in v.items() if not ok]
    assert not failed, f'valid span failed: {failed}'


VIOLATED_FIELD = {
    'bad_trace_id_not_base64': 'trace_id',
    'time_reversed': 'end_time_ns',
    'negative_start': 'start_time_ns',
    'bad_agent_id': 'agent_id',
    'error_without_message': 'status_message',
    'llm_temp_on_non_llm': 'llm_temperature',
    'both_session_flags': 'session_id_derived',
    'null_mandatory_session': 'session_id',
    'bad_nexus_version': 'nexus_distrib_ver',
    'http_status_out_of_range': 'http_status_code',
}


@pytest.mark.parametrize('case', sorted(VIOLATED_FIELD))
def test_malformed_span_fails_exactly_its_field(case, malformed_spans):
    v = _validity(malformed_spans[case])
    assert v[VIOLATED_FIELD[case]] is False, f'{case}: expected {VIOLATED_FIELD[case]} invalid'


def test_mandatory_violation_rejects_whole_trace(sample_spans, malformed_spans):
    """Spec rule: one bad span in a mandatory field rejects the entire trace."""
    from ars.data.validation import Quality
    good_trace = recast(sample_spans.filter(
        pl.col('trace_id') == sample_spans['trace_id'][0]).lazy()).collect()
    bad_span = malformed_spans['bad_trace_id_not_base64'].select(good_trace.columns)
    # poison one extra span next to the good trace
    poisoned = pl.concat([good_trace, bad_span], how='vertical')
    # note: the poisoned span keeps its invalid trace_id → its own trace is rejected
    q = Quality(poisoned.lazy())
    tagged = q.tagged().collect()
    by_trace = tagged.group_by('trace_id').agg(pl.col('rejected').any()).sort('trace_id')
    rejects = dict(by_trace.iter_rows())
    assert rejects['not base64!'] is True
    assert rejects[good_trace['trace_id'][0]] is False


def test_sentinels_match_spec_constants():
    assert Sentinel.empties['INT64'] == -1
    assert Sentinel.empties['FLOAT'] == -1.0
    assert Sentinel.empties['BOOLEAN'] is False
    assert Sentinel.empties['BYTE_ARRAY (UTF8)'] == ''


def test_schema_has_46_fields_and_matches_sample(sample_spans):
    assert len(SpansData.column_names) == 46
    assert set(sample_spans.columns) == set(SpansData.column_names)


def test_recast_repairs_dtype_drift(sample_spans):
    """The shipped sample drifts from the spec (Float64 step, Categorical enums);
    recast() must produce exactly the spec schema."""
    out = recast(sample_spans.lazy()).collect()
    assert dict(out.schema) == dict(SpansData.schema)


def test_recast_fills_missing_column_with_sentinel(valid_span_row):
    row = dict(valid_span_row)
    row.pop('llm_total_tokens')
    df = pl.DataFrame([row])
    out = recast(df.lazy()).collect()
    assert out['llm_total_tokens'].to_list() == [-1]


@pytest.mark.characterization_bug  # F-34: the pipeline itself never validates input
def test_pipeline_load_has_no_validation_gate():
    import inspect
    from ars.stages import s1__data
    src = inspect.getsource(s1__data.load_spans)
    assert 'validation' not in src and 'Quality' not in src
