import omegaconf
from tensordict import TensorDict

from agent.lib.metta_layer import LayerBase

class ObsShaper(LayerBase):
    def __init__(self, obs_shape, num_objects, **cfg):
        super().__init__(**cfg)
        self.obs_shape = obs_shape
        self.output_size = num_objects

    def _forward(self, td: TensorDict):
        x = td['x']

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError('Invalid input tensor shape', x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError('Invalid input tensor shape', x.shape)

        x = x.reshape(B*TT, *space_shape)

        td[self.name] = x.float()
        return td
