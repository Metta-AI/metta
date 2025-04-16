import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase

# file name test, delete this comment after testing


class LSTM(LayerBase):
    def __init__(self, obs_shape, hidden_size, **cfg):
        """Taken from models.py.
        Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain."""

        super().__init__(**cfg)
        self.obs_shape = obs_shape
        self.hidden_size = hidden_size
        self.num_layers = self._nn_params["num_layers"]

    def _make_net(self):
        net = nn.LSTM(self._input_size, self.hidden_size, **self._nn_params)

        for name, param in net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)  # torch's default is uniform

        return net

    def _forward(self, td: TensorDict):
        x = td["x"]
        hidden = td[self._input_source]
        state = td["state"]

        if state is not None:
            split_size = self.num_layers
            state = (state[:split_size], state[split_size:])

        x_shape, space_shape = x.shape, self.obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if x_shape[-space_n:] != space_shape:
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B
        assert hidden.shape == (B * TT, self._input_size)

        hidden = hidden.reshape(B, TT, self._input_size)
        hidden = hidden.transpose(0, 1)

        hidden, state = self._net(hidden, state)

        hidden = hidden.transpose(0, 1)
        hidden = hidden.reshape(B * TT, self.hidden_size)

        if state is not None:
            state = tuple(s.detach() for s in state)
            state = torch.cat(state, dim=0)

        td[self._name] = hidden
        td["state"] = state

        return td
