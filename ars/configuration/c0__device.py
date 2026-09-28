from    typing      import Literal
from    dataclasses import dataclass

from    os          import environ


@dataclass(frozen = True)
class Device:
    name : str   = 'cpu'

    @staticmethod
    def of(value: str = '', fallback: 'None | Device' = None) -> 'Device':
        picked = (str(value).strip() or environ.get('ARS_DEVICE', '').strip()).lower()

        return Device(picked) if picked else (fallback or Device())

    @property
    def torch(self) -> str:
        return 'cuda' if self.name == 'gpu' else 'cpu'

    @property
    def jax(self) -> str:
        return 'cuda' if self.name == 'gpu' else 'cpu'

    @property
    def engine(self) -> Literal['gpu', 'streaming']:
        return 'gpu' if self.name == 'gpu' else 'streaming'

    def force(self) -> None:
        environ['JAX_PLATFORMS'] = self.jax

        # gpu: undo only the BLANKING done for cpu (c0__env_setup / force('cpu'));
        # an operator's own selection (e.g. CUDA_VISIBLE_DEVICES=2) is kept —
        # it used to be popped, exposing every GPU of the host
        if self.name == 'gpu':
            if environ.get('CUDA_VISIBLE_DEVICES') == '': environ.pop('CUDA_VISIBLE_DEVICES')
        else: environ['CUDA_VISIBLE_DEVICES'] = ''

        import jax as jx

        jx.config.update('jax_platforms', self.jax)
