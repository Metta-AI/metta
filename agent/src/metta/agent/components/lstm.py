from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class LSTMConfig(ComponentConfig):
    in_key: str
    out_key: str
    name: str = "lstm"
    latent_size: int = 128
    hidden_size: int = 128
    num_layers: int = 2

    def make_component(self, env=None):
        return LSTM(config=self)


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

    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        self.latent_size = self.config.latent_size
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_layers
        self.in_key = self.config.in_key
        self.out_key = self.config.out_key
        self.net = nn.LSTM(self.latent_size, self.hidden_size, self.num_layers, batch_first=True)

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
    def forward(self, td: TensorDict):
        latent = td[self.in_key]

        if "bptt" not in td.keys():
            raise KeyError("TensorDict is missing required 'bptt' metadata")

        TT = int(td["bptt"][0].item())
        if TT <= 0:
            raise ValueError("bptt entries must be positive")

        total_batch = latent.shape[0]
        B, remainder = divmod(total_batch, TT)
        if remainder != 0:
            raise ValueError("latent batch size must be divisible by bptt")

        if "batch" in td.keys():
            B = int(td["batch"][0].item())

        latent = rearrange(latent, "(b t) h -> b t h", b=B, t=TT)

        # Ensure cuDNN keeps weights in a fused fast-path layout after transfers/checkpoints.
        self.net.flatten_parameters()

        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is not None:
            flat_env_ids = training_env_ids.reshape(-1)
        else:
            flat_env_ids = torch.arange(B, device=latent.device)

        training_env_id_start = int(flat_env_ids[0].item()) if flat_env_ids.numel() else 0

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

        hidden = rearrange(hidden, "b t h -> (b t) h")

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
