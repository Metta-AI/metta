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

    def __init__(self, **cfg):
        super().__init__(**cfg)
        self.hidden_size = self._nn_params["hidden_size"]
        self.num_layers = self._nn_params["num_layers"]

        self.lstm_h: Dict[int, torch.Tensor] = {}
        self.lstm_c: Dict[int, torch.Tensor] = {}

    def has_memory(self):
        return True

    def get_memory(self):
        return self.lstm_h, self.lstm_c

    def set_memory(self, memory):
        """Cannot be called at the MettaAgent level - use policy.component[this_layer_name].set_memory()"""
        self.lstm_h, self.lstm_c = memory[0], memory[1]

    def reset_memory(self):
        self.lstm_h.clear()
        self.lstm_c.clear()

    def setup(self, source_components):
        """Setup the layer and create the network."""
        super().setup(source_components)
        self._net = self._make_net()

    def _make_net(self):
        self._out_tensor_shape = [self.hidden_size]
        net = nn.LSTM(self._in_tensor_shapes[0][0], **self._nn_params)

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

        if training_env_id in self.lstm_h and training_env_id in self.lstm_c:
            h_0 = self.lstm_h[training_env_id]
            c_0 = self.lstm_c[training_env_id]
            # reset the hidden state if the episode is done or truncated
            dones = td.get("dones", None)
            truncateds = td.get("truncateds", None)
            if dones is not None and truncateds is not None:
                reset_mask = dones.bool() | truncateds.bool()
                h_0[:, reset_mask, :] = 0
                c_0[:, reset_mask, :] = 0
        else:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=hidden.device)
            c_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=hidden.device)

        hidden, (h_n, c_n) = self._net(hidden, (h_0, c_0))

        self.lstm_h[training_env_id] = h_n.detach()
        self.lstm_c[training_env_id] = c_n.detach()

        hidden = rearrange(hidden, "t b h -> (b t) h")

        td[self._name] = hidden

        return td
