from dataclasses            import dataclass
from pathlib                import PurePath
from worker_node_compose    import Composition
from worker_node            import Run


@dataclass(frozen = True)
class Worker:
    run_method  : Composition   = Run.subprocess

    extract_dir : PurePath      = PurePath('/tmp/codebase') / 'laim-ars-end2end'

    entry_file  : str           = 'run_all.py'
    entry_func  : str           = 'main'