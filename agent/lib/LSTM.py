from tensordict import TensorDict
import omegaconf

from agent.lib.metta_layer import LayerBase

class MettaLSTM(LayerBase):
    def __init__(self, metta_agent, **cfg):
        '''Taken from models.py.
        Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.'''

        super().__init__(metta_agent, **cfg)
        self.cfg = omegaconf.OmegaConf.create(cfg)
        object.__setattr__(self, 'metta_agent', metta_agent)
        self.obs_shape = self.metta_agent.obs_shape
        self.hidden_size = self.metta_agent.hidden_size


    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]

        if self.input_source is not None:
            self.metta_agent.components[self.input_source].forward(td)

        x = td['x']
        hidden = td[self.input_source]
        state = td["state"]

        # for some reason, td seems to convert state from a tuple to a tensor
        if state is not None:
            state = tuple(state)

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
        assert hidden.shape == (B*TT, self.input_size)

        hidden = hidden.reshape(B, TT, self.input_size)
        hidden = hidden.transpose(0, 1)

        hidden, state = self.layer(hidden, state)

        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(B*TT, self.hidden_size)

        if state is not None:
            state = tuple(s.detach() for s in state)

        td[self.name] = hidden
        td["state"] = state

        return td