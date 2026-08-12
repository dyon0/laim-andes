from    dataclasses             import dataclass, field

from    pathlib                 import PurePath

from    ars.data.stages_meta    import S1Meta, S2Meta
from    ars.models.metrics      import MetricName
from    ars.tools.tui.tui       import ColorSchemeDataScience


@dataclass(frozen = True)
class S3Config:
    s1_meta         : S1Meta
    s2_meta         : S2Meta
    output_dir      : PurePath
    output_prefix   : str
    run_id          : None | str

    select_metric   : MetricName    = 'f1'

    seed    : int   = 12345
    eps     : float = 1e-8
    device  : str   = 'cpu'

    output_color_scheme : ColorSchemeDataScience    = field(
        default_factory = ColorSchemeDataScience)

    def _prefix_for(self) -> str:
        return f'{self.output_prefix}_{self.run_id}' if self.run_id else self.output_prefix