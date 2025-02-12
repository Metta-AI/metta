import torch.nn as nn
from agent.lib.metta_layer import LayerBase
from tensordict import TensorDict
import omegaconf
import torch

class MettaLSTM(LayerBase):
    def __init__(self, metta_agent, **cfg):
        '''Taken from models.py.
        Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain.
        Requires that your policy define encode_observations and decode_actions.
        See the Default policy for an example.'''
        super().__init__(metta_agent, **cfg)
        self.cfg = omegaconf.OmegaConf.create(cfg)
        self.metta_agent = metta_agent
        self.obs_shape = self.metta_agent.obs_shape
        self.hidden_size = self.metta_agent.hidden_size

    def forward(self, td: TensorDict):
        if self.name in td:
            return td[self.name]

        if self.input_source is not None:
            self.metta_agent.components[self.input_source].forward(td)

        x = td['x']
        hidden = td["_encoded_obs_"]
        state = td["state"]

        # --- do we need? ---
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

        #--- It's unclear why our state is not given as a tuple.
        # Split the state tensor into two along the first dimension
        if state is not None:
            state_1, state_2 = torch.split(state, 1, dim=0)
            
            # Remove the first dimension and add a new dimension at the end
            state_1 = state_1.squeeze(0)
            state_2 = state_2.squeeze(0)
            
            # Adjust the size of each state tensor to [1, 48, 129]
            # state_1 = torch.cat((state_1, torch.zeros(1, 48, 1)), dim=-1)
            # state_2 = torch.cat((state_2, torch.zeros(1, 48, 1)), dim=-1)
            state_1 = state_1[:, :, :128]  # Ensure the size is [1, 48, 128]
            state_2 = state_2[:, :, :128]  # Ensure the size is [1, 48, 128]
            # Combine into a tuple
            state = (state_1, state_2)
        #---

        hidden, state = self.layer(hidden, state)
        hidden = hidden.transpose(0, 1)

        hidden = hidden.reshape(B*TT, self.hidden_size)

        td[self.name] = hidden
        td["state"] = state

        return td