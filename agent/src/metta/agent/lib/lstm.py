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
        self._memory = None
        # self.memory = TensorDict(
        #     {
        #         "lstm_h": torch.zeros(self.num_layers, 1, self.hidden_size, dtype=torch.float32),
        #         "lstm_c": torch.zeros(self.num_layers, 1, self.hidden_size, dtype=torch.float32),
        #     },
        #     batch_size=[],
        # )

    def get_memory(self):
        return self._memory

    def set_memory(self, memory):
        self._memory = memory

    def reset_memory(self):
        self._memory = None

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
        # x = td["x"]
        hidden = td[self._sources[0]["name"]]
        # state = None
        state = self._memory

        # av delete this
        # lstm_h = td["lstm_h"]
        # lstm_c = td["lstm_c"]
        # state = None
        # if lstm_h is not None and lstm_c is not None:
        #     # LSTM expects (num_layers, batch, features), so we permute
        #     state = (lstm_h.permute(1, 0, 2), lstm_c.permute(1, 0, 2))

        # x_shape, space_shape = x.shape, self._obs_shape
        # x_n, space_n = len(x_shape), len(space_shape)
        # if tuple(x_shape[-space_n:]) != tuple(space_shape):
        #     raise ValueError("Invalid input tensor shape", x.shape)

        # if x_n == space_n + 1:
        #     #     # rollout mode, feed the cell state from the previous step
        #     B, TT = x_shape[0], 1
        # #     lstm_h = td["lstm_h"]
        # #     lstm_c = td["lstm_c"]
        # #     if lstm_h is not None and lstm_c is not None:
        # #         # LSTM expects (num_layers, batch, features), so we permute
        # #         state = (lstm_h.permute(1, 0, 2).contiguous(), lstm_c.permute(1, 0, 2).contiguous())
        # #     else:
        # #         state = None
        # elif x_n == space_n + 2:
        #     # training mode. We feed a bptt number of observations. LSTM will handle cell state.
        #     B, TT = x_shape[:2]
        # else:
        #     raise ValueError("Invalid input tensor shape", x.shape)

        # av delete this
        # if state is not None:
        #     assert state[0].shape[1] == state[1].shape[1] == B, "LSTM state batch size mismatch"
        # assert hidden.shape == (B * TT, self._in_tensor_shapes[0][0]), (
        #     f"Hidden state shape {hidden.shape} does not match expected {(B * TT, self._in_tensor_shapes[0][0])}"
        # )
        B = td.batch_size.numel()
        TT = 1
        if td["env_obs"].dim() != 3:
            TT = td["env_obs"].shape[1]

        hidden = rearrange(hidden, "(b t) h -> t b h", b=B, t=TT)

        hidden, state = self._net(hidden, state)

        hidden = rearrange(hidden, "t b h -> (b t) h")

        # if state is not None:
        #     # Unpack the state tuple and permute back to (batch, num_layers, features)
        #     lstm_h, lstm_c = state
        #     td["lstm_h"] = lstm_h.detach().permute(1, 0, 2).contiguous()
        #     td["lstm_c"] = lstm_c.detach().permute(1, 0, 2).contiguous()
        self._memory = (state[0].detach(), state[1].detach())

        td[self._name] = hidden

        return td
