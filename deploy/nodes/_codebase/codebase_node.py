from typing                 import Iterator, Iterable
from types                  import SimpleNamespace
from dataclasses            import dataclass

from pathlib                import PurePath, Path
from base64                 import b64encode

from codebase_node_compose  import composable, identity


@dataclass(frozen = True)
class Codebase:
    root                : PurePath
    exclude_dirs        : frozenset[str]
    exclude_suffixes    : frozenset[str]


@composable
def pack(codebase: Codebase) -> bytes:
    from io         import BytesIO
    from tarfile    import open as tar_open

    def drain(it: Iterable) -> None:
        from collections import deque

        deque(it, maxlen = 0)
    
    def collect() -> Iterator[PurePath]:
        def expand(path: PurePath) -> Iterator[PurePath]:
            file = Path(path)

            return map(PurePath, filter(Path.is_file, file.rglob('*')) if file.is_dir() else iter((file,)))

        def allowed(p: PurePath) -> bool:
            return (not any(map(codebase.exclude_dirs.__contains__, p.parts))
                    and p.suffix not in codebase.exclude_suffixes)
        
        return filter(allowed, expand(codebase.root))

    buf = BytesIO()

    with tar_open(fileobj = buf, mode = 'w:gz') as tar:
        drain(map(
            lambda p: tar.add(PurePath.as_posix(p),
                                arcname     = PurePath.as_posix(p.relative_to(codebase.root)),
                                recursive   = False),
            collect()))

    return buf.getvalue()


@composable
def to_digest(raw: bytes) -> str:
    from hashlib import sha256

    return sha256(raw).hexdigest()


@composable
def to_b64(raw: bytes) -> str:
    return b64encode(raw).decode('ascii')


@composable
def to_dframe(raw: bytes) -> 'DataFrame':
    from pandas import DataFrame

    return DataFrame({'archive_b64': (b64encode(raw).decode('ascii'),)})


Pack = SimpleNamespace(
    binary = (pack >> (identity     | to_digest)).compile(),
    string = (pack >> (to_b64       | to_digest)).compile(),
    dframe = (pack >> (to_dframe    | to_digest)).compile())