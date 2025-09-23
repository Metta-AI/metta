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
        self.net = nn.LSTM(self.latent_size, self.hidden_size, self.num_layers)

        for name, param in self.net.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 1)  # Joseph originally had this as 0
            elif "weight" in name:
                nn.init.orthogonal_(param, 1)  # torch's default is uniform

        self._state_capacity = 0
        self._h_buffer: torch.Tensor
        self._c_buffer: torch.Tensor
        self.register_buffer("_h_buffer", torch.zeros(0, dtype=torch.float32), persistent=False)
        self.register_buffer("_c_buffer", torch.zeros(0, dtype=torch.float32), persistent=False)

    def __setstate__(self, state):
        """Ensure LSTM hidden states are properly initialized after loading from checkpoint."""
        self.__dict__.update(state)
        # Reset hidden states when loading from checkpoint to avoid batch size mismatch
        if not hasattr(self, "_state_capacity"):
            self._state_capacity = 0
        if not hasattr(self, "_h_buffer"):
            self.register_buffer("_h_buffer", torch.zeros(0, dtype=torch.float32), persistent=False)
        if not hasattr(self, "_c_buffer"):
            self.register_buffer("_c_buffer", torch.zeros(0, dtype=torch.float32), persistent=False)
        self._state_capacity = 0
        self.reset_memory()

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

        latent = rearrange(latent, "(b t) h -> t b h", b=B, t=TT)

        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is not None:
            flat_env_ids = training_env_ids.reshape(-1).to(torch.long)
        else:
            flat_env_ids = torch.arange(B, device=latent.device, dtype=torch.long)

        self._ensure_state_capacity(int(flat_env_ids.max().item()) + 1 if flat_env_ids.numel() else 0, latent)

        h_0 = self._h_buffer[:, flat_env_ids].to(latent.dtype)
        c_0 = self._c_buffer[:, flat_env_ids].to(latent.dtype)

        # reset the hidden state if the episode is done or truncated
        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        if dones is not None and truncateds is not None:
            reset_mask = (dones.bool() | truncateds.bool()).view(1, -1, 1)
            h_0 = h_0.masked_fill(reset_mask, 0)
            c_0 = c_0.masked_fill(reset_mask, 0)

        hidden, (h_n, c_n) = self.net(latent, (h_0, c_0))

        with torch.no_grad():
            self._h_buffer[:, flat_env_ids] = h_n.detach().to(self._h_buffer.dtype)
            self._c_buffer[:, flat_env_ids] = c_n.detach().to(self._c_buffer.dtype)

        hidden = rearrange(hidden, "t b h -> (b t) h")

        td[self.out_key] = hidden

        return td

    def get_memory(self):
        return self._h_buffer, self._c_buffer

    def set_memory(self, memory):
        """Cannot be called at the MettaAgent level - use policy.component[this_layer_name].set_memory()"""
        self._h_buffer = memory[0]
        self._c_buffer = memory[1]
        self._state_capacity = self._h_buffer.shape[1] if self._h_buffer.ndim > 1 else 0

    def reset_memory(self):
        if self._state_capacity == 0:
            return
        self._h_buffer.zero_()
        self._c_buffer.zero_()

    def _ensure_state_capacity(self, capacity: int, reference: torch.Tensor) -> None:
        if capacity <= self._state_capacity:
            return

        device = reference.device
        dtype = reference.dtype
        new_h = torch.zeros(self.num_layers, capacity, self.hidden_size, device=device, dtype=dtype)
        new_c = torch.zeros_like(new_h)

        if self._state_capacity > 0:
            new_h[:, : self._state_capacity] = self._h_buffer.to(device=device, dtype=dtype)
            new_c[:, : self._state_capacity] = self._c_buffer.to(device=device, dtype=dtype)

        self._h_buffer = new_h
        self._c_buffer = new_c
        self._state_capacity = capacity
