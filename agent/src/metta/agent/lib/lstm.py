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
        # Get hidden_size from _nn_params
        hidden_size = self._nn_params.get("hidden_size", self.hidden_size)
        self._out_tensor_shape = [hidden_size]

        # Guard against setup order issues for static analyzers and runtime safety
        assert (
            getattr(self, "_in_tensor_shapes", None) is not None
            and isinstance(self._in_tensor_shapes, list)
            and len(self._in_tensor_shapes) > 0
            and isinstance(self._in_tensor_shapes[0], list)
            and len(self._in_tensor_shapes[0]) > 0
        ), "LSTM requires a valid input tensor shape from its source component"
        net = nn.LSTM(self._in_tensor_shapes[0][0], **self._nn_params)

        for name, param in net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # torch's default is uniform

        return net

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def _forward(self, td: TensorDict):
        assert (
            getattr(self, "_sources", None) is not None and isinstance(self._sources, list) and len(self._sources) > 0
        ), "LSTM requires at least one source component"
        hidden = td[self._sources[0]["name"]]  # → (2, num_layers, batch, hidden_size)

        TT = td["bptt"][0]
        B = td["batch"][0]

        hidden = rearrange(hidden, "(b t) h -> t b h", b=B, t=TT)

        training_env_id_start = td.get("training_env_id_start", torch.tensor([0], dtype=torch.long))[0].item()

        if training_env_id_start in self.lstm_h and training_env_id_start in self.lstm_c:
            h_0 = self.lstm_h[training_env_id_start]
            c_0 = self.lstm_c[training_env_id_start]
            # reset the hidden state if the episode is done or truncated
            dones = td.get("dones", None)
            truncateds = td.get("truncateds", None)
            if dones is not None and truncateds is not None:
                reset_mask = (dones.bool() | truncateds.bool()).view(1, -1, 1)
                h_0 = h_0.masked_fill(reset_mask, 0)
                c_0 = c_0.masked_fill(reset_mask, 0)
        else:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=hidden.device)
            c_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=hidden.device)

        hidden, (h_n, c_n) = self._net(hidden, (h_0, c_0))

        self.lstm_h[training_env_id_start] = h_n.detach()
        self.lstm_c[training_env_id_start] = c_n.detach()

        hidden = rearrange(hidden, "t b h -> (b t) h")

        td[self._name] = hidden

        return td
