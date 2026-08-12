from    typing                          import ClassVar, Iterator
from    dataclasses                     import dataclass

import  polars                          as pl

from    ars.specification.common_core   import AttrGroup, Patterns, PhysType, Sentinel


@dataclass(frozen = True)
class Predicate:
    is_llm      : ClassVar[pl.Expr] = pl.col('aef_kind') == 'llm'
    is_http     : ClassVar[pl.Expr] = pl.col('aef_kind').is_in(('input_request', 'output_request'))
    is_kafka    : ClassVar[pl.Expr] = pl.col('aef_kind').is_in(('kafka_produce', 'kafka_consume'))
    is_consume  : ClassVar[pl.Expr] = pl.col('aef_kind') == 'kafka_consume'
    is_error    : ClassVar[pl.Expr] = pl.col('status_code') == 'STATUS_CODE_ERROR'


@dataclass(frozen = True)
class Json:
    @staticmethod
    def like(col: str) -> pl.Expr:
        return pl.col(col).str.contains(Patterns.json_like)

    @staticmethod
    def strict(col: str) -> pl.Expr:
        return pl.col(col).str.json_decode(pl.String).is_not_null()


@dataclass(frozen = True, eq = False)
class SpanAttr:
    name            : str
    group           : AttrGroup
    type_parquet    : PhysType
    type_polars     : type[pl.DataType] | pl.Enum
    sentinel        : None | str
    validation      : pl.Expr
    mandatory       : bool

    @property
    def empty(self) -> None | bool | int | float | str:
        return Sentinel.empty(self.sentinel, self.type_parquet)

    @property
    def trash(self) -> None | int | float | str:
        return Sentinel.garbage(self.type_parquet)

    @property
    def validity(self) -> pl.Expr:
        return pl.col(self.name).is_not_null() & self.validation


@dataclass(frozen = True, eq = False)
class SpansDataSpec:
    attrs   : tuple[SpanAttr, ...]

    @property
    def schema(self) -> pl.Schema:
        return pl.Schema(
            map(lambda a: (a.name, a.type_polars), self.attrs),
            check_dtypes = True)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(map(lambda a: a.name, self.attrs))

    @property
    def mandatory_names(self) -> frozenset[str]:
        return frozenset(map(
            lambda a: a.name,
            filter(lambda a: a.mandatory, self.attrs)))

    def validities(self) -> Iterator[tuple[str, pl.Expr]]:
        return map(lambda a: (a.name, a.validity), self.attrs)