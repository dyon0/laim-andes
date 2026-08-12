from    typing                              import Any, Callable, ClassVar, Literal
from    dataclasses                         import dataclass
from    functools                           import reduce, wraps
from    itertools                           import takewhile, repeat

from    os                                  import environ
from    time                                import time, sleep
from    concurrent.futures                  import ThreadPoolExecutor

from    psutil                              import virtual_memory, Process

import  jax                                 as jx

from    ars.tools.abstraction.composition   import composable


type Metric = float
type Sample = tuple[None | Metric, None | Metric]


@dataclass(frozen = True)
class Hardware:
    gpu: ClassVar[bool] = jx.default_backend() == 'gpu'


def cpu_memory(cmd: Literal['free', 'used'], prec: int = 5) -> Metric:
    match cmd:
        case 'free':    return round(virtual_memory().available  / (1024 * 1024), prec)
        case 'used':    return round(Process().memory_info().rss / (1024 * 1024), prec)
        case _:         raise ValueError('неизвестная команда оценки ram')


def nvs_memory(cmd: Literal['free', 'used'], prec: int = 5) -> None | Metric:
    stats = (jx.devices()[0].memory_stats() or None) if Hardware.gpu else None

    match (cmd, stats):
        case (_, None):     return None
        case ('used', _):   return round(stats['bytes_in_use'] / (1024 * 1024), prec)
        case ('free', _):   return round((stats['bytes_limit'] - stats['bytes_in_use']) / (1024 * 1024), prec)
        case _:             raise ValueError('неизвестная команда оценки vram')


def available_memory_mb(prec: int = 5) -> Sample:
    return cpu_memory('free', prec), nvs_memory('free', prec)


def measure_during_call(
        func            : Callable,
        args            : tuple,
        kwargs          : dict,
        interval_sec    : float = 0.05,
        prec            : int = 5,
) -> tuple[Any, Metric, None | Metric, None | Metric]:
    read    = lambda  : (cpu_memory('used', prec), nvs_memory('used', prec) if Hardware.gpu else None)
    tick    = lambda _: (sleep(interval_sec), read())[1]

    if environ.get('ARS_BENCH_PEAK', 'on').strip().lower() in ('off', 'false', '0', 'no'):
        start   = time()
        result  = func(*args, **kwargs)

        return result, time() - start, None, None

    with ThreadPoolExecutor(max_workers = 1) as pool:
        start   = time()
        pending = pool.submit(func, *args, **kwargs)
        samples = tuple(takewhile(lambda _: not pending.done(), map(tick, repeat(None))))
        result  = pending.result()
        elapsed = time() - start

    taken   = samples or (read(),)
    folder  = lambda a, b: b if a is None else a if b is None else max(a, b)

    return result, elapsed, reduce(folder, map(lambda s: s[0], taken), None), reduce(folder, map(lambda s: s[1], taken), None)


@composable
def benchmark(name: None | str = None):
    def decorate(func: Callable) -> Callable:
        display = name if isinstance(name, str) else func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            result, elapsed, peak_cpu, peak_gpu = measure_during_call(func, args, kwargs)
            setattr(wrapper, 'elapsed',     elapsed)
            setattr(wrapper, 'peak_cpu',    peak_cpu)
            setattr(wrapper, 'peak_gpu',    peak_gpu)

            scheme = getattr(wrapper, '__cs_call', None)
            if name is not None and scheme is not None:
                scheme.print_performance_metrics(display, elapsed, peak_cpu, peak_gpu)

            return result

        setattr(wrapper, '__benchmarked',   True)
        setattr(wrapper, '__display_name',  display)
        setattr(wrapper, 'elapsed',         None)
        setattr(wrapper, 'peak_cpu',        None)
        setattr(wrapper, 'peak_gpu',        None)

        return wrapper

    return decorate(name) if callable(name) else decorate


def inject_color_scheme(module_globals: dict, scheme: Any) -> None:
    benchmarked = filter(lambda obj: callable(obj) and getattr(obj, '__benchmarked', False), module_globals.values())

    _ = tuple(map(lambda obj: setattr(obj, '__cs_call', scheme), benchmarked))