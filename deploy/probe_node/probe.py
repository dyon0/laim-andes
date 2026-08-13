"""SberDS environment probe — a standalone diagnostic node.

Purpose: answer, from INSIDE the real `py312-gpu` image, the questions that
decide how the LAIM detector node must be packaged and how it can use the
8×H100 host:

  1. Does the base image already ship torch / jax, and are they importable?
     If torch fails, WHY (which shared object is missing, from which build).
  2. What package index can this image reach, and does it carry the packages
     we would need (intel-* runtime, CUDA-variant torch)?
  3. What CUDA driver / runtime and how many GPUs are actually visible?
  4. What does an input port really deliver (single file? directory? how many
     parts? how big?) — including the embedder blob's true format.
  5. What are the real CPU / memory limits of the container?

Design rules (this file is deliberately boring and defensive):
  * stdlib ONLY at module level — importing torch/jax/polars here would
    reproduce the very crash we are diagnosing;
  * every probe is wrapped so a failure is RECORDED, never raised — the node
    must always finish and always return a report;
  * everything is printed to stdout (visible in the SberDS log) AND returned
    on the output ports as JSON/text/HTML.

Deploy: copy `probe.py` + `descriptor.json` to the root of a new git repo and
register it as a node on the same base image as the detector node.
"""

import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from pathlib import Path

# ----------------------------------------------------------------- helpers

_REPORT: dict = {}
_LINES: list[str] = []


def out(line: str = '') -> None:
    """Print to the platform log and keep a copy for the text/HTML ports."""
    print(line, flush=True)
    _LINES.append(line)


def section(title: str) -> None:
    out('')
    out('=' * 78)
    out(f'[PROBE] {title}')
    out('=' * 78)


def kv(key: str, value) -> None:
    out(f'  {key:.<38} {value}')


def probe(name: str):
    """Decorator: run a probe, record its result, never let it raise."""
    def wrap(fn):
        def run(*a, **k):
            section(name)
            try:
                result = fn(*a, **k)
                _REPORT[name] = result
                return result
            except Exception as exc:              # noqa: BLE001 - probe must not die
                out(f'  !! probe failed: {type(exc).__name__}: {exc}')
                for ln in traceback.format_exc().splitlines()[-6:]:
                    out(f'     {ln}')
                _REPORT[name] = {'probe_error': f'{type(exc).__name__}: {exc}'}
                return None
        return run
    return wrap


def sh(cmd: list[str], timeout: int = 60) -> dict:
    """Run a command, capture everything, never raise."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {'cmd': ' '.join(cmd), 'rc': p.returncode,
                'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()[:4000]}
    except FileNotFoundError:
        return {'cmd': ' '.join(cmd), 'rc': None, 'error': 'command not found'}
    except subprocess.TimeoutExpired:
        return {'cmd': ' '.join(cmd), 'rc': None, 'error': f'timeout after {timeout}s'}
    except Exception as exc:                       # noqa: BLE001
        return {'cmd': ' '.join(cmd), 'rc': None, 'error': f'{type(exc).__name__}: {exc}'}


def human(n) -> str:
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if abs(n) < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


def read_first(*paths: str) -> str | None:
    for p in paths:
        try:
            return Path(p).read_text().strip()
        except Exception:                          # noqa: BLE001
            continue
    return None


# ------------------------------------------------------------------ probes

@probe('01_python_runtime')
def python_runtime() -> dict:
    import site
    info = {
        'python_version': sys.version.replace('\n', ' '),
        'executable': sys.executable,
        'platform': platform.platform(),
        'site_packages': [p for p in sys.path if 'site-packages' in p or 'dist-packages' in p],
    }
    try:
        info['site_getsitepackages'] = site.getsitepackages()
    except Exception:                              # noqa: BLE001
        pass
    kv('python', info['python_version'][:60])
    kv('executable', info['executable'])
    kv('platform', info['platform'])
    out('  site-packages on sys.path:')
    for p in info['site_packages']:
        exists = 'exists' if Path(p).is_dir() else 'MISSING'
        kv(f'    {p}', exists)
    return info


@probe('02_cpu_limits')
def cpu_limits() -> dict:
    info = {
        'os_cpu_count': os.cpu_count(),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
    }
    try:
        info['sched_getaffinity'] = len(os.sched_getaffinity(0))
    except Exception:                              # noqa: BLE001
        info['sched_getaffinity'] = 'unavailable'

    # cgroup v2 then v1
    v2 = read_first('/sys/fs/cgroup/cpu.max')
    if v2:
        info['cgroup_v2_cpu_max'] = v2
        parts = v2.split()
        if len(parts) == 2 and parts[0] != 'max':
            info['effective_cores_from_cgroup'] = round(int(parts[0]) / int(parts[1]), 2)
    quota = read_first('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    period = read_first('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota and period:
        info['cgroup_v1_quota_us'] = quota
        info['cgroup_v1_period_us'] = period
        if quota != '-1':
            info['effective_cores_from_cgroup'] = round(int(quota) / int(period), 2)

    for k, v in info.items():
        kv(k, v)
    out('')
    out('  >> ANSWER Q5: "effective_cores_from_cgroup" is the real CPU budget.')
    out('     If it is ~1, the polars feature engineering will be the bottleneck')
    out('     regardless of how many GPUs are attached.')
    return info


@probe('03_memory_limits')
def memory_limits() -> dict:
    info = {}
    v2 = read_first('/sys/fs/cgroup/memory.max')
    if v2:
        info['cgroup_v2_memory_max'] = v2
        info['cgroup_v2_memory_max_human'] = human(v2) if v2 != 'max' else 'unlimited'
    v1 = read_first('/sys/fs/cgroup/memory/memory.limit_in_bytes')
    if v1:
        info['cgroup_v1_limit'] = v1
        info['cgroup_v1_limit_human'] = human(v1)
    try:
        meminfo = dict(
            (ln.split(':')[0], ln.split(':')[1].strip())
            for ln in Path('/proc/meminfo').read_text().splitlines()[:5])
        info['proc_meminfo'] = meminfo
    except Exception:                              # noqa: BLE001
        pass
    for k, v in info.items():
        kv(k, v)
    return info


@probe('04_disk_space')
def disk_space(model_store_dir: str) -> dict:
    import shutil as sh_
    info = {}
    for path in ('/tmp', '/opt/module', str(model_store_dir), os.getcwd()):
        try:
            usage = sh_.disk_usage(path)
            info[path] = {'total': human(usage.total), 'free': human(usage.free)}
            kv(f'free on {path}', f'{human(usage.free)} of {human(usage.total)}')
        except Exception as exc:                   # noqa: BLE001
            info[path] = f'unavailable: {exc}'
            kv(f'free on {path}', f'unavailable ({exc})')

    # can we WRITE to the model store? (train mode must persist the bundle)
    target = Path(model_store_dir)
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe_file = target / '.laim_probe_write_test'
        probe_file.write_text('ok')
        probe_file.unlink()
        info['model_store_writable'] = True
        kv('model_store_dir writable', f'YES ({model_store_dir})')
    except Exception as exc:                       # noqa: BLE001
        info['model_store_writable'] = f'NO: {type(exc).__name__}: {exc}'
        kv('model_store_dir writable', f'NO — {exc}')
    out('')
    out('  >> The model bundle (train mode) is written to model_store_dir and its')
    out('     PATH is passed to the inference node. It must be writable here and')
    out('     readable from the inference node.')
    return info


@probe('05_gpu_nvidia_smi')
def gpu_nvidia_smi() -> dict:
    info = {}
    smi = sh(['nvidia-smi'], timeout=60)
    if smi.get('rc') != 0:
        for alt in ('/usr/bin/nvidia-smi', '/usr/local/nvidia/bin/nvidia-smi'):
            if Path(alt).exists():
                smi = sh([alt], timeout=60)
                break
    info['nvidia_smi'] = smi
    if smi.get('stdout'):
        for ln in smi['stdout'].splitlines()[:15]:
            out(f'  {ln}')
    else:
        out(f"  nvidia-smi unavailable: {smi.get('error') or smi.get('stderr')}")

    q = sh(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total,memory.used',
            '--format=csv,noheader'], timeout=60)
    info['gpu_table'] = q
    if q.get('stdout'):
        out('')
        out('  GPUs (index, name, driver, mem.total, mem.used):')
        for ln in q['stdout'].splitlines():
            out(f'    {ln}')
        info['gpu_count'] = len(q['stdout'].splitlines())
    info['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
    kv('CUDA_VISIBLE_DEVICES', info['CUDA_VISIBLE_DEVICES'])
    out('')
    out('  >> ANSWER Q3: driver_version + the "CUDA Version:" in the header decide')
    out('     which torch/jax CUDA variant we may pin (cu121 / cu124 / cu126 / cu128).')
    return info


@probe('06_import_checks')
def import_checks() -> dict:
    """The heart of the probe: can each dependency actually be imported?"""
    targets = ('torch', 'jax', 'jaxlib', 'flax', 'optax', 'polars', 'numpy',
               'pandas', 'sklearn', 'sentence_transformers', 'transformers',
               'pyarrow', 'altair', 'tqdm', 'psutil')
    results = {}
    for name in targets:
        entry = {}
        try:
            from importlib.metadata import version
            entry['dist_version'] = version(name if name != 'sklearn' else 'scikit-learn')
        except Exception:                          # noqa: BLE001
            entry['dist_version'] = None
        try:
            mod = __import__(name)
            entry['import'] = 'OK'
            entry['version'] = getattr(mod, '__version__', None)
            entry['file'] = getattr(mod, '__file__', None)
        except Exception as exc:                   # noqa: BLE001
            entry['import'] = 'FAILED'
            entry['error'] = f'{type(exc).__name__}: {exc}'
        results[name] = entry
        status = entry['import']
        detail = entry.get('version') or entry.get('dist_version') or ''
        mark = '  ' if status == 'OK' else '!!'
        kv(f'{mark} {name}', f"{status:7s} {detail}  {entry.get('error', '')}")
    out('')
    out('  >> ANSWER Q1: any package showing "FAILED" is broken IN THE IMAGE.')
    out('     A package with dist_version but FAILED import = installed but unusable.')
    return results


@probe('07_torch_deep_dive')
def torch_deep_dive(deep_ldd: bool) -> dict:
    """Why torch fails (libsycl.so.9) — locate the build and its real deps."""
    info = {}

    # where does the torch DISTRIBUTION live, even if importing it fails?
    candidates = []
    for p in sys.path:
        t = Path(p) / 'torch'
        if t.is_dir():
            candidates.append(str(t))
    info['torch_dirs_on_path'] = candidates
    for c in candidates:
        kv('torch package dir', c)

    try:
        from importlib.metadata import distribution
        dist = distribution('torch')
        info['torch_dist_version'] = dist.version
        info['torch_dist_location'] = str(dist.locate_file(''))
        kv('torch dist version', dist.version)
        kv('torch dist location', info['torch_dist_location'])
        reqs = dist.requires or []
        info['torch_requires'] = reqs
        intel = [r for r in reqs if 'intel' in r.lower() or 'sycl' in r.lower()]
        nvidia = [r for r in reqs if 'nvidia' in r.lower()]
        info['torch_requires_intel'] = intel
        info['torch_requires_nvidia'] = nvidia
        out('  declared dependencies containing "intel"/"sycl":')
        for r in (intel or ['    <none>']):
            out(f'    {r}')
        out('  declared dependencies containing "nvidia" (first 8):')
        for r in (nvidia[:8] or ['    <none>']):
            out(f'    {r}')
        out('')
        out('  >> If "intel"/"sycl" deps are listed, this is the XPU-enabled PyPI')
        out('     build; it needs the Intel oneAPI runtime that the image lacks.')
    except Exception as exc:                       # noqa: BLE001
        info['torch_dist_error'] = str(exc)
        kv('torch dist metadata', f'unavailable: {exc}')

    # hunt the actual missing shared object
    libs = {}
    for c in candidates:
        libdir = Path(c) / 'lib'
        if libdir.is_dir():
            names = sorted(p.name for p in libdir.glob('*.so*'))
            libs[str(libdir)] = {'count': len(names),
                                 'sycl': [n for n in names if 'sycl' in n.lower()],
                                 'sample': names[:12]}
            kv(f'shared objects in {libdir}', len(names))
            kv('  ...of which SYCL-related', libs[str(libdir)]['sycl'] or 'none')
    info['torch_libs'] = libs

    # is libsycl anywhere on the system at all?
    found = []
    search_dirs = ['/usr/lib64', '/usr/lib', '/usr/local/lib', '/usr/local/lib64',
                   '/opt/intel', '/usr/local/cuda/lib64']
    for d in search_dirs:
        try:
            found += [str(p) for p in Path(d).rglob('libsycl*') if p.is_file()][:5]
        except Exception:                          # noqa: BLE001
            continue
    info['libsycl_found_on_system'] = found
    kv('libsycl* found on system', found or 'NOT FOUND ANYWHERE')
    info['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '<unset>')
    kv('LD_LIBRARY_PATH', info['LD_LIBRARY_PATH'])

    # what does torch's native extension actually link against?
    if deep_ldd:
        for c in candidates:
            for so in sorted(Path(c).glob('_C*.so'))[:1]:
                res = sh(['ldd', str(so)], timeout=90)
                info['ldd_torch_C'] = res
                out('')
                out(f'  ldd {so}:')
                text = res.get('stdout', '') or res.get('error', '')
                missing = [ln for ln in text.splitlines() if 'not found' in ln]
                for ln in (missing or text.splitlines()[:15]):
                    out(f'    {ln.strip()}')
                info['ldd_missing'] = [m.strip() for m in missing]
                if missing:
                    out('')
                    out('  >> These "not found" entries are the exact root cause.')
    return info


@probe('08_jax_devices')
def jax_devices() -> dict:
    """Does JAX see all 8 GPUs? (decides the multi-GPU design.)"""
    info = {}
    try:
        import jax
        info['jax_version'] = jax.__version__
        kv('jax version', jax.__version__)
        try:
            info['default_backend'] = jax.default_backend()
            kv('default backend', info['default_backend'])
        except Exception as exc:                   # noqa: BLE001
            info['default_backend'] = f'error: {exc}'
        try:
            devs = jax.devices()
            info['device_count'] = len(devs)
            info['devices'] = [str(d) for d in devs]
            kv('jax.device_count()', len(devs))
            for d in devs:
                out(f'    {d}')
        except Exception as exc:                   # noqa: BLE001
            info['devices_error'] = f'{type(exc).__name__}: {exc}'
            kv('jax.devices()', f'FAILED: {exc}')
        try:
            info['local_device_count'] = jax.local_device_count()
        except Exception:                          # noqa: BLE001
            pass
        out('')
        out('  >> ANSWER (multi-GPU): device_count is how many H100s JAX can drive.')
        out('     The current detector code uses devices[0] ONLY; the rest idle.')
    except Exception as exc:                       # noqa: BLE001
        info['jax_import_error'] = f'{type(exc).__name__}: {exc}'
        kv('jax import', f'FAILED: {exc}')
    return info


@probe('09_torch_cuda')
def torch_cuda() -> dict:
    info = {}
    try:
        import torch
        info['version'] = torch.__version__
        info['file'] = torch.__file__
        kv('torch version', torch.__version__)
        kv('torch file', torch.__file__)
        for label, fn in (
            ('torch.version.cuda', lambda: torch.version.cuda),
            ('cuda.is_available', lambda: torch.cuda.is_available()),
            ('cuda.device_count', lambda: torch.cuda.device_count()),
            ('cudnn.version', lambda: torch.backends.cudnn.version()),
            ('xpu.is_available', lambda: getattr(torch, 'xpu', None) and torch.xpu.is_available()),
        ):
            try:
                val = fn()
            except Exception as exc:               # noqa: BLE001
                val = f'error: {exc}'
            info[label] = str(val)
            kv(label, val)
        try:
            for i in range(torch.cuda.device_count()):
                out(f'    cuda:{i} = {torch.cuda.get_device_name(i)}')
        except Exception:                          # noqa: BLE001
            pass
    except Exception as exc:                       # noqa: BLE001
        info['import_error'] = f'{type(exc).__name__}: {exc}'
        kv('torch import', f'FAILED: {exc}')
        out('  (expected if probe 07 shows the missing SYCL runtime)')
    return info


@probe('10_pip_environment')
def pip_environment(probe_packages: str, check_network: bool) -> dict:
    """What index can this image reach, and does it carry what we need?"""
    info = {}
    info['pip_version'] = sh([sys.executable, '-m', 'pip', '--version'], timeout=60)
    kv('pip', info['pip_version'].get('stdout') or info['pip_version'].get('error'))

    info['pip_config'] = sh([sys.executable, '-m', 'pip', 'config', 'list'], timeout=60)
    out('  pip config:')
    for ln in (info['pip_config'].get('stdout') or '<empty>').splitlines():
        out(f'    {ln}')

    conf_files = {}
    for c in ('/etc/pip.conf', '/etc/xdg/pip/pip.conf',
              str(Path.home() / '.pip/pip.conf'), str(Path.home() / '.config/pip/pip.conf')):
        try:
            if Path(c).exists():
                conf_files[c] = Path(c).read_text()[:1500]
        except Exception:                          # noqa: BLE001
            continue
    info['pip_conf_files'] = conf_files
    for path, body in conf_files.items():
        out(f'  {path}:')
        for ln in body.splitlines():
            out(f'    {ln}')
    if not conf_files:
        out('  (no pip.conf found in the usual locations)')

    info['pip_env_vars'] = {k: v for k, v in os.environ.items()
                            if k.upper().startswith(('PIP_', 'UV_')) or 'INDEX' in k.upper()}
    for k, v in info['pip_env_vars'].items():
        kv(k, v)

    # freeze: full inventory of what the image ships.
    # `pip freeze` can be unavailable (e.g. a venv without the pip module), so
    # fall back to importlib.metadata — the inventory is too important to lose.
    freeze = sh([sys.executable, '-m', 'pip', 'freeze'], timeout=180)
    lines = (freeze.get('stdout') or '').splitlines()
    info['pip_freeze_source'] = 'pip freeze'
    if not lines:
        try:
            from importlib.metadata import distributions
            lines = sorted(
                f"{d.metadata['Name']}=={d.version}"
                for d in distributions() if d.metadata and d.metadata['Name'])
            info['pip_freeze_source'] = 'importlib.metadata (pip unavailable)'
        except Exception as exc:                   # noqa: BLE001
            info['inventory_error'] = str(exc)
    info['pip_freeze'] = '\n'.join(lines)
    info['pip_freeze_count'] = len(lines)
    kv('packages installed in image', f"{len(lines)}  (via {info['pip_freeze_source']})")
    interesting = [ln for ln in lines
                   if any(t in ln.lower() for t in
                          ('torch', 'jax', 'intel', 'sycl', 'nvidia', 'polars',
                           'sentence', 'transformers', 'flax', 'optax'))]
    out('  relevant preinstalled packages:')
    for ln in (interesting or ['    <none>']):
        out(f'    {ln}')

    # can pip RESOLVE the packages we may need to add?
    resolutions = {}
    for pkg in [p.strip() for p in str(probe_packages).split(',') if p.strip()]:
        res = sh([sys.executable, '-m', 'pip', 'index', 'versions', pkg], timeout=120)
        ok = res.get('rc') == 0
        resolutions[pkg] = res
        kv(f'index has "{pkg}"', 'YES — ' + (res.get('stdout', '').splitlines() or [''])[0]
           if ok else 'NO / unreachable — ' + (res.get('stderr', '') or res.get('error', ''))[:120])
    info['package_resolution'] = resolutions
    out('')
    out('  >> ANSWER Q2: the pip config/index above + which packages resolve tell us')
    out('     whether we can pin a CUDA-variant torch or must add the intel runtime.')

    if check_network:
        hosts = [('pypi.org', 443), ('files.pythonhosted.org', 443),
                 ('download.pytorch.org', 443)]
        # also test whatever index the image is configured with
        blob = json.dumps(info.get('pip_conf_files', {})) + json.dumps(info.get('pip_env_vars', {}))
        for token in blob.split():
            if token.startswith('http'):
                try:
                    from urllib.parse import urlparse
                    h = urlparse(token.strip('",')).hostname
                    if h:
                        hosts.append((h, 443))
                except Exception:                  # noqa: BLE001
                    pass
        reach = {}
        out('  network reachability (TCP connect, 5s timeout):')
        for host, port in dict.fromkeys(hosts):
            try:
                with socket.create_connection((host, port), timeout=5):
                    reach[host] = 'reachable'
            except Exception as exc:               # noqa: BLE001
                reach[host] = f'unreachable: {type(exc).__name__}'
            out(f'    {host:40s} {reach[host]}')
        info['network'] = reach
    return info


@probe('11_input_port_layout')
def input_port_layout(ports: dict) -> dict:
    """What does a file port ACTUALLY deliver: file or directory, how many parts."""
    info = {}
    for name, raw in ports.items():
        if not raw:
            out(f'  {name}: <not connected>')
            info[name] = {'connected': False}
            continue
        p = Path(str(raw))
        entry = {'connected': True, 'raw_value': str(raw), 'exists': p.exists(),
                 'is_dir': p.is_dir(), 'is_file': p.is_file()}
        out('')
        kv(f'{name} raw value', raw)
        kv('  exists / is_dir / is_file', f'{p.exists()} / {p.is_dir()} / {p.is_file()}')
        if p.is_dir():
            children = sorted(p.iterdir())
            entry['n_children'] = len(children)
            entry['children_sample'] = [c.name for c in children[:5]]
            total = 0
            for c in children:
                try:
                    total += c.stat().st_size if c.is_file() else 0
                except Exception:                  # noqa: BLE001
                    pass
            entry['total_size'] = human(total)
            kv('  children', len(children))
            kv('  total size', human(total))
            for c in children[:5]:
                try:
                    kv(f'    {c.name}', human(c.stat().st_size) if c.is_file() else '<dir>')
                except Exception:                  # noqa: BLE001
                    kv(f'    {c.name}', '<stat failed>')
            if len(children) > 5:
                out(f'    ... and {len(children) - 5} more')
        elif p.is_file():
            size = p.stat().st_size
            entry['size'] = human(size)
            kv('  size', human(size))
            # magic bytes — is this extension-less blob a zip/tar/gzip?
            try:
                head = p.open('rb').read(8)
                entry['magic_hex'] = head.hex()
                sigs = {b'PK\x03\x04': 'ZIP archive', b'\x1f\x8b': 'GZIP',
                        b'ustar': 'TAR', b'\x28\xb5\x2f\xfd': 'ZSTD',
                        b'BZh': 'BZIP2', b'\x80\x02': 'pickle',
                        b'PAR1': 'PARQUET'}
                fmt = next((v for k, v in sigs.items() if head.startswith(k)), None)
                if not fmt:
                    tar_check = p.open('rb')
                    tar_check.seek(257)
                    fmt = 'TAR' if tar_check.read(5) == b'ustar' else 'unknown'
                    tar_check.close()
                entry['detected_format'] = fmt
                kv('  magic bytes', head.hex())
                kv('  detected format', fmt)
                out('')
                out('  >> ANSWER Q4: this is the true format of the port payload.')
                out('     The detector node must unpack by MAGIC BYTES, not by suffix')
                out('     (the platform delivers extension-less blobs).')
            except Exception as exc:               # noqa: BLE001
                entry['magic_error'] = str(exc)
        info[name] = entry
    return info


@probe('12_environment_variables')
def environment_variables() -> dict:
    keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES',
            'NVIDIA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'PATH', 'PYTHONPATH',
            'JAX_PLATFORMS', 'XLA_FLAGS', 'XLA_PYTHON_CLIENT_PREALLOCATE',
            'XLA_PYTHON_CLIENT_MEM_FRACTION', 'HF_HOME', 'HF_HUB_OFFLINE',
            'TRANSFORMERS_CACHE', 'TMPDIR', 'HOME', 'CUDA_HOME')
    info = {k: os.environ.get(k, '<unset>') for k in keys}
    for k, v in info.items():
        kv(k, v if len(str(v)) < 120 else str(v)[:117] + '...')
    info['_all_env_keys'] = sorted(os.environ)
    return info


# -------------------------------------------------------------------- main

def main(**params) -> dict:
    """SberDS entry point. Every parameter is optional."""
    out('#' * 78)
    out('#  LAIM SberDS ENVIRONMENT PROBE')
    out('#  Answers: image contents, package index, CUDA/GPUs, port layout, limits')
    out('#' * 78)
    out(f'  received parameters: {sorted(params)}')

    model_store_dir = str(params.get('model_store_dir') or '/mnt/data/laim/models')
    probe_packages = str(params.get('probe_pip_packages')
                         or 'torch,intel-sycl-rt,intel-cmplr-lib-rt,jax,polars,sentence-transformers')
    check_network = str(params.get('check_network', 'true')).strip().lower() in ('true', '1', 'yes', 'on')
    deep_ldd = str(params.get('deep_ldd', 'true')).strip().lower() in ('true', '1', 'yes', 'on')

    python_runtime()
    cpu_limits()
    memory_limits()
    disk_space(model_store_dir)
    gpu_nvidia_smi()
    import_checks()
    torch_deep_dive(deep_ldd)
    jax_devices()
    torch_cuda()
    pip_environment(probe_packages, check_network)
    input_port_layout({
        'path_probe_data': params.get('path_probe_data'),
        'path_probe_model': params.get('path_probe_model'),
    })
    environment_variables()

    # ------------------------------------------------------------ verdict
    section('SUMMARY / VERDICT')
    imports = _REPORT.get('06_import_checks') or {}
    broken = [n for n, e in imports.items() if isinstance(e, dict) and e.get('import') == 'FAILED']
    jaxinfo = _REPORT.get('08_jax_devices') or {}
    cpuinfo = _REPORT.get('02_cpu_limits') or {}
    gpuinfo = _REPORT.get('05_gpu_nvidia_smi') or {}

    verdict = {
        'broken_imports': broken,
        'jax_device_count': jaxinfo.get('device_count'),
        'gpu_count_nvidia_smi': gpuinfo.get('gpu_count'),
        'effective_cores': cpuinfo.get('effective_cores_from_cgroup', cpuinfo.get('os_cpu_count')),
        'torch_requires_intel': (_REPORT.get('07_torch_deep_dive') or {}).get('torch_requires_intel'),
        'ldd_missing': (_REPORT.get('07_torch_deep_dive') or {}).get('ldd_missing'),
    }
    kv('broken imports', broken or 'none — all dependencies usable')
    kv('GPUs (nvidia-smi)', verdict['gpu_count_nvidia_smi'])
    kv('GPUs visible to JAX', verdict['jax_device_count'])
    kv('effective CPU cores', verdict['effective_cores'])
    kv('torch intel/sycl deps', verdict['torch_requires_intel'] or 'none declared')
    kv('missing shared objects', verdict['ldd_missing'] or 'none')
    _REPORT['00_verdict'] = verdict

    out('')
    out('  Send this whole log back for the packaging + multi-GPU decision.')
    out('#' * 78)

    text = '\n'.join(_LINES)
    try:
        Path('/tmp/laim_probe_report.json').write_text(
            json.dumps(_REPORT, indent=2, ensure_ascii=False, default=str))
    except Exception:                              # noqa: BLE001
        pass

    html = ('<html><head><meta charset="utf-8"><title>LAIM environment probe</title></head>'
            '<body style="font-family:monospace;background:#111;color:#ddd">'
            f'<h2>LAIM SberDS environment probe</h2><pre>{_escape(text)}</pre>'
            '</body></html>')

    return {
        'probe_report': _REPORT,
        'probe_text': text,
        'probe_html': html,
    }


def _escape(s: str) -> str:
    return (s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))


if __name__ == '__main__':
    # local smoke run: python probe.py [port_dir_or_file]
    argv = sys.argv[1:]
    result = main(**({'path_probe_data': argv[0]} if argv else {}))
    print(f'\n[local] report keys: {sorted(result["probe_report"])}')
