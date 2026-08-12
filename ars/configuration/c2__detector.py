from    dataclasses             import dataclass, field

from    pathlib                 import PurePath

from    ars.models.metrics      import MetricName
from    ars.data.stages_meta    import S1Meta
from    ars.tools.tui.tui       import ColorSchemeDataScience


@dataclass(frozen = True)
class S2Config:
    '''конфигурация этапа обучения детектора'''

    s1_meta         : S1Meta
    output_dir      : PurePath
    output_prefix   : str
    run_id          : None | str

    threshold_metric    : MetricName    = 'youden'
    select_metric       : MetricName    = 'youden'

    # optional experiment-grid overrides; None keeps the values compiled into
    # ars/configuration/experiments/e2__detector.py (legacy behavior)
    experiments : None | tuple[str, ...]    = None
    epochs      : None | int                = None
    patience    : None | int                = None

    n_thresholds                : int   = 10000
    inference_normal_count      : int   = 100000
    inference_anomalous_count   : int   = 100000

    encode_chunk                : int   = 1024
    seq_pad_chunk               : int   = 8192

    # F-03: floor for train-latent std in latent normalization (0.0 = legacy
    # behavior of std+eps, which overflowed on degenerate dimensions)
    latent_std_floor    : float = 1e-3

    seed    : int   = 12345
    eps     : float = 1e-8
    device  : str   = 'cpu'

    output_color_scheme : ColorSchemeDataScience    = field(
        default_factory = ColorSchemeDataScience)

    def _prefix_for(self) -> str:
        return f'{self.output_prefix}_{self.run_id}' if self.run_id else self.output_prefix