from functools              import partial
from itertools              import chain, starmap

from codebase_node_config   import Package
from codebase_node          import Codebase


def main() -> dict[str, 'bytes | str | DataFrame']:
    workers     = Package.workers.items()
    codebase    = partial(Codebase,
                            exclude_dirs        = Package.exclude_dirs,
                            exclude_suffixes    = Package.exclude_suffixes)
    
    pack        = Package.pack_method << codebase

    def entries(port: str, dir: str) -> tuple[tuple[str, 'bytes | str | DataFrame'], tuple[str, str]]:
        content, digest = pack(Package.base / dir)

        return (f'codebase_{port}', content), (f'checksum_{port}', digest)

    result = dict(chain.from_iterable(starmap(entries, workers)))
    print(result)

    return result

if __name__ == '__main__': main()