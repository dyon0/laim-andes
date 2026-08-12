from    typing      import Tuple, cast
from    functools   import partial, reduce
from    itertools   import chain, starmap

from    pathlib     import Path
from    subprocess  import run
from    sys         import executable, exit, stderr


def install_via_subprocess(
    wheels_dir: str,
    requirements_file: str,
    python: str = executable
) -> None:
    #целевой вариант
    
    cmd = (
        python, '-m', 'pip', 'install',
        '--no-index', '--find-links', wheels_dir,
        '-r', requirements_file
    )
    
    run_install = partial(run, cmd, capture_output=True, text=True)
    result = run_install()
    
    match result.returncode:
        case 0:
            print('все зависимости успешно установлены')
        case code:
            error_lines = chain(
                (f'ошибка установки (код {code}):',),
                result.stderr.splitlines()
            )
            print('\n'.join(error_lines), file = stderr)
            exit(code)


def install_composition(
    wheels_dir: str,
    requirements_file: str,
    python: str = executable
) -> None:
    cmd = (
        python, '-m', 'pip', 'install',
        '--no-index', '--find-links', wheels_dir,
        '-r', requirements_file
    )
    
    actions = (
        lambda _: run(cmd, capture_output = True, text = True),
        lambda r: (r.returncode, r.stdout, r.stderr)
    )
    
    result = cast(Tuple[int, str, str], reduce(lambda acc, f: f(acc), actions, None))

    match result[0]:
        case 0:
            print('установка завершена')
        case code:
            err_msg = result[2]
            formatted = starmap(
                '{}\n{}'.format,
                ((f'Ошибка (код {code}):', err_msg),)
            )
            print(next(formatted), file=stderr)
            exit(code)


def install_idempotent(
    wheels_dir: str,
    requirements_file: str,
    marker: str = '/.deps_installed',
    python: str = executable
) -> None:
    def maybe_install(path_exists: bool) -> None:
        match path_exists:
            case True:
                print('зависимости уже установлены (маркер найден)')
            case False:
                cmd = (
                    python, '-m', 'pip', 'install',
                    '--no-index', '--find-links', wheels_dir,
                    '-r', requirements_file
                )
                proc = run(cmd, capture_output=True, text=True)

                match proc.returncode:
                    case 0:
                        with open(marker, 'w') as f: f.write('installed')
                        print('установка завершена, маркер создан')
                    case code:
                        err_output = (f'ошибка pip (код {code}):', proc.stderr)
                        print('\n'.join(err_output), file=stderr)
                        exit(code)

    maybe_install(Path(marker).exists())