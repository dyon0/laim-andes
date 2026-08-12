from typing                 import Mapping, Iterable, Callable, Any
from types                  import SimpleNamespace
from functools              import partial
from itertools              import starmap
from dataclasses            import dataclass

from pathlib                import PurePath
from sys                    import modules as sys_modules

from worker_node_compose    import composable


@dataclass(frozen = True)
class Delivery:
    payload     : 'bytes | str | DataFrame'
    checksum    : str
    extract_dir : PurePath
    entry_file  : str
    entry_func  : str
    params      : Mapping[str, Any]


@dataclass(frozen = True)
class RunContext:
    project_dir : PurePath
    entry_file  : PurePath
    entry_func  : str
    params      : Mapping[str, Any]


@composable
def unpack(delivery: Delivery) -> RunContext:
    def decode(payload: 'bytes | str | DataFrame') -> bytes:
        from base64 import b64decode

        match payload:
            case bytes():   return payload
            case str():     return b64decode(payload)
            case _:         return b64decode(payload['archive_b64'].iloc[0])

    def verify(raw: bytes, checksum: str) -> bytes:
        from hashlib import sha256

        if sha256(raw).hexdigest() != checksum:
            raise ValueError('sha256 полученной нагрузки не совпал с присланным')

        return raw

    def extract(raw: bytes, into: PurePath) -> PurePath:
        from io         import BytesIO
        from tempfile   import mkdtemp
        from tarfile    import open as tar_open
        from pathlib    import Path

        Path(into).mkdir(parents = True, exist_ok = True)
        target = mkdtemp(prefix = 'project_', dir = into.as_posix())

        with tar_open(fileobj = BytesIO(raw), mode = 'r:gz') as tar:
            tar.extractall(path = target, filter = 'data')

        return PurePath(target)

    raw = verify(decode(delivery.payload), delivery.checksum)

    return RunContext(
        project_dir = extract(raw, delivery.extract_dir),
        entry_file  = PurePath(delivery.entry_file),
        entry_func  = delivery.entry_func,
        params      = delivery.params)


def via_runpy_path(context: RunContext) -> Any:
    from runpy import run_path

    namespace = run_path((context.project_dir / context.entry_file).as_posix())

    return namespace[context.entry_func](**context.params)


def via_runpy_module(context: RunContext) -> Any:
    from runpy import run_module

    name    = '.'.join(context.entry_file.with_suffix('').parts)
    cached  = sys_modules.pop(name, None)

    try: return run_module(name)[context.entry_func](**context.params)
    finally:
        sys_modules.pop(name, None)

        if cached is not None: sys_modules[name] = cached


def via_subprocess(context: RunContext) -> Any:
    from subprocess import run as sub_run
    from pickle     import dump as pkl_dump, loads as pkl_loads
    from pathlib    import Path
    from sys        import executable

    params_path = context.project_dir / '.params.pkl'
    result_path = context.project_dir / '.result.pkl'

    with open(params_path.as_posix(), 'wb') as handle:
        pkl_dump(dict(context.params), handle)

    boot = (
        'import runpy, pickle, sys; '

        'script, params_file, target, func = sys.argv[1:5]; '
        'sys.argv = [script]; '
        'ns = runpy.run_path(script); '

        'pickle.dump(ns[func](**pickle.load(open(params_file, "rb"))), open(target, "wb"))'
    )

    done = sub_run((executable, '-u', '-c', boot,
                        (context.project_dir / context.entry_file).as_posix(),
                        params_path.as_posix(),
                        result_path.as_posix(),
                        context.entry_func),
                    cwd = context.project_dir.as_posix())

    if done.returncode: raise RuntimeError(f'нагрузка отправила код возврата {done.returncode}')

    return pkl_loads(Path(result_path).read_bytes())


def execute(runner: Callable[[RunContext], Any], context: RunContext) -> Any:
    from shutil import rmtree
    from sys    import path as sys_path

    project_dir = context.project_dir
    project_dir_as_posix = project_dir.as_posix()

    def purge() -> None:
        def drain(it: Iterable) -> None:
            from collections import deque

            deque(it, maxlen = 0)
        
        stale = tuple(starmap(
            lambda k, v: k,
            filter(
                lambda kv: str(getattr(kv[1], '__file__', '') or '').startswith(project_dir_as_posix),
                sys_modules.items())))

        drain(map(lambda name: sys_modules.pop(name, None), stale))

    sys_path.insert(0, project_dir_as_posix)

    try: return runner(context)
    finally:
        sys_path.remove(project_dir_as_posix)
        purge()
        rmtree(project_dir_as_posix, ignore_errors = True)


Run = SimpleNamespace(
    runpy_path      = (unpack >> partial(execute, via_runpy_path)   ).compile(),
    runpy_module    = (unpack >> partial(execute, via_runpy_module) ).compile(),
    subprocess      = (unpack >> partial(execute, via_subprocess)   ).compile())