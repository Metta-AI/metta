import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class LSTM(LayerBase):
    def __init__(self, obs_shape, hidden_size, **cfg):
        """Taken from models.py.
        Wraps your policy with an LSTM without letting you shoot yourself in the
        foot with bad transpose and shape operations. This saves much pain."""

        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)  # make sure no Omegaconf types are used in forward passes
        self.hidden_size = hidden_size
        # self._out_tensor_shape = [hidden_size] # delete this
        self.num_layers = self._nn_params["num_layers"]

    def _make_net(self):
        self._out_tensor_shape = [self.hidden_size]
        net = nn.LSTM(self._in_tensor_shapes[0][0], self.hidden_size, **self._nn_params)

        for name, param in net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)  # torch's default is uniform

        return net

    @torch.compile(disable=True)  # Dynamo doesn't support compiling LSTMs
    def _forward(self, td: TensorDict):
        x = td["x"]
        hidden = td[self._sources[0]["name"]]
        state = td["state"]

        if state is not None:
            split_size = self.num_layers
            state = (state[:split_size], state[split_size:])

        x_shape, space_shape = x.shape, self._obs_shape
        x_n, space_n = len(x_shape), len(space_shape)
        if tuple(x_shape[-space_n:]) != tuple(space_shape):
            raise ValueError("Invalid input tensor shape", x.shape)

        if x_n == space_n + 1:
            B, TT = x_shape[0], 1
        elif x_n == space_n + 2:
            B, TT = x_shape[:2]
        else:
            raise ValueError("Invalid input tensor shape", x.shape)

        if state is not None:
            assert state[0].shape[1] == state[1].shape[1] == B
        assert hidden.shape == (B * TT, self._in_tensor_shapes[0][0])

        hidden = hidden.reshape(B, TT, self._in_tensor_shapes[0][0])
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
