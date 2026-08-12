from    typing      import Any, Callable
from    dataclasses import dataclass
from    itertools   import dropwhile
from    inspect     import currentframe, getouterframes

from    pathlib     import PurePath
from    json        import dump as json_dump,   load as json_load
from    pickle      import dump as pickle_dump, load as pickle_load, HIGHEST_PROTOCOL


@dataclass(frozen = True)
class FrameNotFound: ...


@dataclass(frozen = True)
class Found:
    var: str
    val: Any

    def __iter__(self): return iter((self.var, self.val))


def get_func_var_kv(func: Callable, var_name: str, strict: bool = True) -> type[FrameNotFound] | Found:
    seen = lambda info: (Found(var_name, info.frame.f_locals[var_name])
                        if info.frame.f_code is func.__code__
                        else FrameNotFound)
    
    if (point := currentframe()) is None:
        raise RuntimeError('попытка поиска в глобальном фрейме')

    found = next(dropwhile(lambda x: x is FrameNotFound, map(seen, getouterframes(point.f_back or point))), FrameNotFound)
    if strict and found is FrameNotFound:
        raise RuntimeError(f'фрейм функции {func.__name__} ({func}) не найден')

    return found


@dataclass(frozen = True)
class FileIO:
    @staticmethod
    def pickle_write(path: PurePath, payload: dict[str, Any]) -> None:
        with open(path, 'wb') as handle:                    pickle_dump(payload, handle, protocol = HIGHEST_PROTOCOL)

    @staticmethod
    def pickle_read(path: PurePath) -> dict[str, Any]:
        with open(path, 'rb') as handle:                    return pickle_load(handle)

    @staticmethod
    def json_write(path: PurePath, payload: dict[str, Any]) -> None:
        with open(path, 'w', encoding = 'utf-8') as handle: json_dump(payload, handle, indent = 4, ensure_ascii = False)

    @staticmethod
    def json_read(path: PurePath) -> dict[str, Any]:
        with open(path, 'r', encoding = 'utf-8') as handle: return json_load(handle)