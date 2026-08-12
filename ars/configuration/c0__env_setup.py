from    os          import environ
from    dataclasses import dataclass

environ['XLA_PYTHON_CLIENT_PREALLOCATE']     = 'true'
environ['XLA_PYTHON_CLIENT_MEM_FRACTION']    = '0.80'
#environ['XLA_PYTHON_CLIENT_ALLOCATOR']       = 'platform'
environ['XLA_FLAGS']                         = (
    '--xla_gpu_autotune_level=3 '
    '--xla_dump_to=/tmp/xla_hlo_dump '
)

environ['JAX_NUM_COMPILATION_THREADS']       = '20'

#environ['CUDA_LAUNCH_BLOCKING'] = '1' # включать только для дебага — замедляет всё на 10–30%

environ['TF_CPP_MIN_LOG_LEVEL']              = '3'
environ['TF_CPP_VMODULE']                    = ''
environ['GRPC_VERBOSITY']                    = 'ERROR'
environ['GLOG_minloglevel']                  = '3'
environ['ABSL_LOGGING_VERBOSITY']            = '1'
environ['JAX_LOG_COMPILES']                  = '1'


import jax as jx

jx.config.update('jax_compilation_cache_dir',                  '/tmp/jax_cache')
jx.config.update('jax_persistent_cache_min_compile_time_secs',  5)
jx.config.update('jax_persistent_cache_min_entry_size_bytes',  -1)
jx.config.update('jax_persistent_cache_enable_xla_caches',     'xla_gpu_per_fusion_autotune_cache_dir')

device = environ.get('ARS_DEVICE', 'cpu').strip().lower()

environ['JAX_PLATFORMS'] = 'cuda' if device == 'gpu' else 'cpu'

if device != 'gpu':
    environ['CUDA_VISIBLE_DEVICES'] = ''

jx.config.update('jax_platforms', environ['JAX_PLATFORMS'])


environ['HF_HUB_OFFLINE'] = '1'


@dataclass(frozen = True)
class Runtime:
    @staticmethod
    def apply(track_peak: bool, disable_progress: bool, progress_every: float) -> None:
        environ['ARS_BENCH_PEAK'] = 'on' if track_peak else 'off'

        if disable_progress:        environ['ARS_PROGRESS']         = 'off'
        if progress_every > 0.0:    environ['ARS_PROGRESS_EVERY']   = repr(progress_every)