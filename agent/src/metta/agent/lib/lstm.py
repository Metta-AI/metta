from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict

from metta.agent.lib.metta_layer import LayerBase


class LSTM(LayerBase):
    """
    LSTM layer that handles tensor reshaping and state management automatically.

    This class wraps a PyTorch LSTM with proper tensor shape handling, making it easier
    to integrate LSTMs into neural network policies without dealing with complex tensor
    manipulations. It handles reshaping inputs/outputs, manages hidden states, and ensures
    consistent tensor dimensions throughout the forward pass.

    The layer processes tensors of shape [B, TT, ...] or [B, ...], where:
    - B is the batch size
    - TT is an optional time dimension

    It reshapes inputs appropriately for the LSTM, processes them through the network,
    and reshapes outputs back to the expected format, while also managing the LSTM's
    hidden state.

    Note that the __init__ of any layer class and the MettaAgent are only called when the agent
    is instantiated and never again. I.e., not when it is reloaded from a saved policy.
    """

    def __init__(self, obs_shape, hidden_size, **cfg):
        super().__init__(**cfg)
        self._obs_shape = list(obs_shape)  # make sure no Omegaconf types are used in forward passes
        self.hidden_size = hidden_size
        self.num_layers = self._nn_params["num_layers"]

        self.max_envs = 524288  # av hack. need a better way. don't want to get it from the env, it should just work.
        # preallocate state buffer. [0] = hidden states, [1] = cell states
        # self.register_buffer(
        #     "_memory", torch.zeros(2, self.num_layers, self.max_envs, self.hidden_size), persistent=False
        # )
        self.lstm_h: Dict[int, torch.Tensor] = {}
        self.lstm_c: Dict[int, torch.Tensor] = {}
        self._memory = True

    def get_memory(self):
        return self._memory

    def set_memory(self, memory):
        self._memory = memory

    def reset_memory(self):
        self.lstm_h.clear()
        self.lstm_c.clear()

    def setup(self, source_components):
        """Setup the layer and create the network."""
        super().setup(source_components)
        self._net = self._make_net()

    def _make_net(self):
        self._out_tensor_shape = [self.hidden_size]
        net = nn.LSTM(self._in_tensor_shapes[0][0], self.hidden_size, **self._nn_params)

        for name, param in net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # torch's default is uniform

        return net

    @torch.compile(disable=True)  # Dynamo doesn't support compiling LSTMs
    def _forward(self, td: TensorDict):
        hidden = td[self._sources[0]["name"]]  # → (2, num_layers, batch, hidden_size)

        TT = td.bptt
        B = td.batch

        hidden = rearrange(hidden, "(b t) h -> t b h", b=B, t=TT)

        if hasattr(td, "training_env_id"):
            training_env_id = td.training_env_id.start
        else:
            training_env_id = 0

        h_0 = (
            self.lstm_h[training_env_id]
            if training_env_id in self.lstm_h
            else torch.zeros(self.num_layers, B, self.hidden_size, device=hidden.device)
        )
        c_0 = (
            self.lstm_c[training_env_id]
            if training_env_id in self.lstm_c
            else torch.zeros(self.num_layers, B, self.hidden_size, device=hidden.device)
        )
        state = (h_0, c_0)

        hidden, (h_n, c_n) = self._net(hidden, state)

        self.lstm_h[training_env_id] = h_n.detach()
        self.lstm_c[training_env_id] = c_n.detach()

        hidden = rearrange(hidden, "t b h -> (b t) h")

        td[self._name] = hidden

        return td
