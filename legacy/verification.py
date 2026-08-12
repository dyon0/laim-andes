from    typing                  import Any, Iterator
from    dataclasses             import dataclass, asdict
from    functools               import reduce
from    itertools               import pairwise, starmap

import  polars                  as pl

from    ars.data.stages_meta    import S1Meta


@dataclass(frozen = True)
class Compare:
    @staticmethod
    def shown(value: Any, as_repr: bool) -> Any:
        return str(value) if as_repr else value

    @staticmethod
    def normalize(value: Any, tolerance: int, as_repr: bool) -> Any:
        recurse = lambda v: Compare.normalize(v, tolerance, as_repr)

        match value:
            case float():   return Compare.shown(round(value, tolerance), as_repr)
            case dict():    return Compare.shown(dict(starmap(lambda k, v: (k, recurse(v)), value.items())), as_repr)
            case tuple():   return Compare.shown(tuple(map(recurse, value)), as_repr)
            case list():    return Compare.shown(list(map(recurse, value)), as_repr)
            case _:         return Compare.shown(value, as_repr)

    @staticmethod
    def rounded(df: pl.DataFrame, tolerance: int) -> pl.DataFrame:
        return df.with_columns(pl.selectors.numeric().cast(pl.Float64).round(tolerance))

    @staticmethod
    def prepared(meta: S1Meta, tolerance: int, as_repr: bool) -> dict[str, Any]:
        return dict(starmap(lambda k, v: (k, Compare.normalize(v, tolerance, as_repr)), asdict(meta).items()))


def compare_dataframes(actual: pl.DataFrame, expected: pl.DataFrame, float_tolerance: int = 5, as_repr: bool = False) -> bool:
    left    = Compare.rounded(actual,   float_tolerance)
    right   = Compare.rounded(expected, float_tolerance)

    return str(left) == str(right) if as_repr else left.equals(right)


def compare_s1meta(meta1: S1Meta, meta2: S1Meta, float_tolerance: int = 5, as_repr: bool = False) -> bool:
    return Compare.prepared(meta1, float_tolerance, as_repr) == Compare.prepared(meta2, float_tolerance, as_repr)


def verify_s1_consistency(artifacts: Iterator[tuple[S1Meta, pl.DataFrame]], float_tolerance: int = 5, as_repr: bool = False) -> bool:
    step = lambda acc, two: (acc
        and compare_dataframes(two[0][1], two[1][1], float_tolerance, as_repr)
        and compare_s1meta(two[0][0], two[1][0], float_tolerance, as_repr))

    return reduce(step, pairwise(artifacts), True) #todo: перепроверить корректность логики: что, если lrn(artifacts) нечетное?