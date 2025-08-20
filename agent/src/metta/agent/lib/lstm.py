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
        self.reset_in_training = cfg.get("reset_in_training", False)

        self.lstm_h: Dict[int, torch.Tensor] = {}
        self.lstm_c: Dict[int, torch.Tensor] = {}

    def __setstate__(self, state):
        """Ensure LSTM hidden states are properly initialized after loading from checkpoint."""
        self.__dict__.update(state)
        # Reset hidden states when loading from checkpoint to avoid batch size mismatch
        if not hasattr(self, "lstm_h"):
            self.lstm_h = {}
        if not hasattr(self, "lstm_c"):
            self.lstm_c = {}
        # Clear any existing states to handle batch size mismatches
        self.lstm_h.clear()
        self.lstm_c.clear()

    def on_rollout_start(self):
        self.reset_memory()

    def on_train_phase_start(self):
        self.reset_memory()

    def on_mb_start(self):
        # if self.reset_in_training:
        #     pass
        # else:
        #     self.reset_memory()
        self.reset_memory()

    def on_eval_start(self):
        self.reset_memory()

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

        self.lstm = nn.LSTM(self._in_tensor_shapes[0][0], **self._nn_params)

        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # torch's default is uniform

        return None

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def _forward(self, td: TensorDict):
        assert (
            getattr(self, "_sources", None) is not None and isinstance(self._sources, list) and len(self._sources) > 0
        ), "LSTM requires at least one source component"
        latent = td[self._sources[0]["name"]]  # → (batch * TT, hidden_size)

        TT = td["bptt"][0]
        B = td["batch"][0]

        training_env_id_start = td.get("training_env_id_start", None)
        if training_env_id_start is None:
            training_env_id_start = 0
        else:
            training_env_id_start = training_env_id_start[0].item()

        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        if dones is not None and truncateds is not None:
            reset_mask = (dones.bool() | truncateds.bool()).view(1, -1, 1)
        else:
            reset_mask = torch.ones(1, B, 1, device=latent.device)

        if training_env_id_start in self.lstm_h and training_env_id_start in self.lstm_c:
            h_0 = self.lstm_h[training_env_id_start]
            c_0 = self.lstm_c[training_env_id_start]
            if TT == 1:
                h_0 = h_0.masked_fill(reset_mask, 0)
                c_0 = c_0.masked_fill(reset_mask, 0)
        else:
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=latent.device)
            c_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=latent.device)

        latent = rearrange(latent, "(b t) h -> t b h", b=B, t=TT)
        if self.reset_in_training and TT != 1:
            hidden, (h_n, c_n) = self._forward_train_step(latent, h_0, c_0, reset_mask, B, TT)
        else:
            hidden, (h_n, c_n) = self.lstm(latent, (h_0, c_0))

        hidden = rearrange(hidden, "t b h -> (b t) h")

        self.lstm_h[training_env_id_start] = h_n.detach()
        self.lstm_c[training_env_id_start] = c_n.detach()

        td[self._name] = hidden

        return td

    def _forward_train_step(self, latent, h_t, c_t, reset_mask, B, TT):
        # latent is (B * TT, input_size)
        # h_0 is (num_layers, B, input_size)
        # c_0 is (num_layers, B, input_size)
        # reset_mask is (1, B * TT, 1)

        reset_mask = reset_mask.view(1, B, TT, 1)
        hidden = None

        for t in range(TT):
            latent_t = latent[t, :, :].unsqueeze(0)
            # reset_mask_t = reset_mask[0, :, t, :]
            # h_t = h_t.masked_fill(reset_mask_t, 0)
            # c_t = c_t.masked_fill(reset_mask_t, 0)
            hidden_t, (h_t, c_t) = self.lstm(latent_t, (h_t, c_t))  # one time step
            # stack hidden
            if hidden is None:
                hidden = hidden_t
            else:
                hidden = torch.cat([hidden, hidden_t], dim=0)

        return hidden, (h_t, c_t)
