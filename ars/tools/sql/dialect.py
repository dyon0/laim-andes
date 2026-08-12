from    typing      import ClassVar
from    dataclasses import dataclass


@dataclass(frozen = True)
class Dialect:
    param       : str
    floor       : str
    modulo      : str
    random      : str
    bool_and    : str
    bool_or     : str


@dataclass(frozen = True)
class Dialects:
    FreeSQL     : ClassVar[Dialect] = Dialect(
        param       = ':{}',
        floor       = 'floor({})',
        modulo      = 'mod({}, {})',
        random      = 'random()',
        bool_and    = 'MIN(CASE WHEN {} THEN 1 ELSE 0 END) = 1',
        bool_or     = 'MAX(CASE WHEN {} THEN 1 ELSE 0 END) = 1')
    SQLite      : ClassVar[Dialect] = Dialect(
        param       = ':{}',
        floor       = 'CAST({} AS INTEGER)',
        modulo      = '(({}) % {})',
        random      = 'random()',
        bool_and    = 'MIN(CASE WHEN {} THEN 1 ELSE 0 END) = 1',
        bool_or     = 'MAX(CASE WHEN {} THEN 1 ELSE 0 END) = 1')
    DuckDB      : ClassVar[Dialect] = Dialect(
        param       = '${}',
        floor       = 'floor({})',
        modulo      = '(({}) % {})',
        random      = 'random()',
        bool_and    = 'bool_and({})',
        bool_or     = 'bool_or({})')
    Postgres    : ClassVar[Dialect] = Dialect(
        param       = '%({})s',
        floor       = 'floor({})',
        modulo      = 'mod({}, {})',
        random      = 'random()',
        bool_and    = 'bool_and({})',
        bool_or     = 'bool_or({})')

    @staticmethod
    def of(name: str) -> Dialect:
        return {
            'FreeSQL'   : Dialects.FreeSQL,
            'SQLite'    : Dialects.SQLite,
            'DuckDB'    : Dialects.DuckDB,
            'Postgres'  : Dialects.Postgres}.get(name, Dialects.FreeSQL)