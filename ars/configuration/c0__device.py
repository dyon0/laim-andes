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

        if self.name == 'gpu': environ.pop('CUDA_VISIBLE_DEVICES', None)
        else: environ['CUDA_VISIBLE_DEVICES'] = ''

        import jax as jx

        jx.config.update('jax_platforms', self.jax)
