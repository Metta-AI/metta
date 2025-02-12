from agent.lib.metta_layer import LayerBase
import omegaconf
from tensordict import TensorDict
from torch import nn

class ObsShaper(LayerBase):
    def __init__(self, metta_agent, **cfg):
        super().__init__(metta_agent, **cfg)
        
        self.cfg = omegaconf.OmegaConf.create(cfg)
        self.metta_agent = metta_agent
        self.obs_shape = self.metta_agent.obs_shape
        self.output_size = self.metta_agent.num_objects

    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]
        
        if self.input_source is not None:
            self.metta_agent.components[self.input_source].forward(td)

        state = td['state']
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

        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B

        x = x.reshape(B*TT, *space_shape)

        td[self.name] = x.float()

        return td
