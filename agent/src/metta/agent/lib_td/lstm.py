from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict

from metta.common.config.config import Config


class LSTMConfig(Config):
    latent_size: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    in_key: str = "latent"
    out_key: str = "hidden"


class LSTM(nn.Module):
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

    def __init__(self, config: Optional[LSTMConfig] = None):
        super().__init__()
        self.config = config or LSTMConfig()
        self.latent_size = self.config.latent_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key
        self.net = nn.LSTM(self.latent_size, self.hidden_size, self.num_layers)

        for name, param in self.net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # torch's default is uniform

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

    @torch._dynamo.disable  # Exclude LSTM forward from Dynamo to avoid graph breaks
    def _forward(self, td: TensorDict):
        latent = td[self.in_key]  # → (2, num_layers, batch, hidden_size)

        TT = td["bptt"][0]
        B = td["batch"][0]

        latent = rearrange(latent, "(b t) h -> t b h", b=B, t=TT)

        training_env_id_start = td.get("training_env_id_start", None)
        if training_env_id_start is None:
            training_env_id_start = 0
        else:
            training_env_id_start = training_env_id_start[0].item()

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
            h_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=latent.device)
            c_0 = torch.zeros(self.num_layers, B, self.hidden_size, device=latent.device)

        hidden, (h_n, c_n) = self.net(latent, (h_0, c_0))

        self.lstm_h[training_env_id_start] = h_n.detach()
        self.lstm_c[training_env_id_start] = c_n.detach()

        hidden = rearrange(hidden, "t b h -> (b t) h")

        td[self.out_key] = hidden

        return td

    def get_memory(self):
        return self.lstm_h, self.lstm_c

    def set_memory(self, memory):
        """Cannot be called at the MettaAgent level - use policy.component[this_layer_name].set_memory()"""
        self.lstm_h, self.lstm_c = memory[0], memory[1]

    def reset_memory(self):
        self.lstm_h.clear()
        self.lstm_c.clear()
