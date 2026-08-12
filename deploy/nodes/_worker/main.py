from typing             import Any
from types              import MappingProxyType
from functools          import partial

from worker_node_config import Worker
from worker_node        import Delivery


def main(
    received_codebase       : 'bytes | str | DataFrame',
    received_checksum       : str,

    **params                : Any
) -> Any:
    delivery    = partial(Delivery,
                            extract_dir = Worker.extract_dir,
                            entry_file  = Worker.entry_file,
                            entry_func  = Worker.entry_func,
                            params      = MappingProxyType(params)
    )

    run         = Worker.run_method << delivery

    return run(received_codebase, received_checksum)