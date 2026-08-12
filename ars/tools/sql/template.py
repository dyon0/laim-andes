from    typing      import ClassVar
from    dataclasses import dataclass


@dataclass(frozen = True)
class Template:
    eligible        : ClassVar[str] = (
        'eligible AS (\n'
        '    SELECT {schema.trace} AS trace, MIN({schema.begin}) AS trace_begin, COUNT(*) AS span_count\n'
        '    FROM {table}\n'
        '    GROUP BY {schema.trace}\n'
        '    HAVING {having}\n'
        ')')

    present         : ClassVar[str] = (
        'present AS (\n'
        '    SELECT {schema.trace} AS trace, MIN({schema.begin}) AS trace_begin\n'
        '    FROM {table}\n'
        '    WHERE {span_ok}\n'
        '    GROUP BY {schema.trace}\n'
        ')')

    capped_all      : ClassVar[str] = (
        'capped AS (\n'
        '    SELECT trace, trace_begin FROM eligible\n'
        ')')

    capped_limit    : ClassVar[str] = (
        'capped AS (\n'
        '    SELECT trace, trace_begin FROM (\n'
        '        SELECT trace, trace_begin,\n'
        '               SUM(span_count) OVER (ORDER BY trace_begin, trace\n'
        '                                     ROWS UNBOUNDED PRECEDING) AS cum_spans\n'
        '        FROM eligible\n'
        '    ) s\n'
        '    WHERE cum_spans <= {limit}\n'
        ')')

    ordered         : ClassVar[str] = (
        'ordered AS (\n'
        '    SELECT trace, trace_begin,\n'
        '           ROW_NUMBER() OVER (ORDER BY trace_begin, trace) AS pos\n'
        '    FROM {source}\n'
        ')')

    marked          : ClassVar[str] = (
        'marked AS (\n'
        '    SELECT trace,\n'
        '           (ROW_NUMBER() OVER (ORDER BY {order}) <= COUNT(*) OVER () - {cut}) AS is_holdout\n'
        '    FROM ordered\n'
        ')')

    output_aligned  : ClassVar[str] = (
        'SELECT r.*, m.is_holdout AS holdout\n'
        'FROM {table} r\n'
        'JOIN marked m ON r.{schema.trace} = m.trace{holdout}\n'
        'ORDER BY r.{schema.trace}, r.{schema.begin}')

    output_open     : ClassVar[str] = (
        'SELECT r.*, m.is_holdout AS holdout\n'
        'FROM {table} r\n'
        'JOIN marked m ON r.{schema.trace} = m.trace\n'
        'WHERE {span_ok}{holdout}\n'
        'ORDER BY r.{schema.trace}, r.{schema.begin}{limit}')

    limit           : ClassVar[str] = '\nLIMIT {param}'

    @staticmethod
    def query(ctes: tuple[str, ...], tail: str) -> str:
        return 'WITH ' + ',\n'.join(ctes) + '\n' + tail