import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class LLevel(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class HLevel(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class HRMReasoningConfig(ComponentConfig):
    """HRM reasoning component that processes encoded observations."""

    class_path: str = "metta.agent.components.hrm.HRMReasoning"
    in_key: str = "hrm_obs_encoded"
    out_key: str = "hrm_reasoning"
    name: str = "hrm_reasoning"
    embed_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_multiplier: int = 4  # FFN hidden size = embed_dim * ffn_multiplier
    # ACT parameters
    H_cycles: int = 3  # Number of high-level reasoning cycles
    L_cycles: int = 5  # Number of low-level processing cycles per H-cycle
    Mmin: int = 1  # Minimum number of reasoning segments
    Mmax: int = 10  # Maximum number of reasoning segments

    def make_component(self, env=None) -> "HRMReasoning":
        return HRMReasoning(self)


# Memory
class HRMMemory:
    def __init__(self):
        self.carry = {}

    def has_memory(self):
        return True

    def set_memory(self, memory):
        self.carry = memory

    def get_memory(self):
        return self.carry

    def reset_memory(self):
        self.carry = {}

    def reset_env_memory(self, env_id):
        if env_id in self.carry:
            del self.carry[env_id]


class HRMReasoning(nn.Module):
    """HRM reasoning component using hierarchical reasoning with ACT."""

    def __init__(self, config: HRMReasoningConfig):
        super().__init__()
        self.config = config

        self.memory = HRMMemory()
        self.carry = self.memory.get_memory()

        # Hierarchical reasoning levels
        self.L_level = LLevel(embed_dim=config.embed_dim, num_layers=config.num_layers)
        self.H_level = HLevel(embed_dim=config.embed_dim, num_layers=config.num_layers)

        # Q-head for halting decision (2 outputs: halt, continue)
        self.q_head = nn.Linear(config.embed_dim, 2)

        # Multi-head attention layers (kept for backward compatibility)
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_heads, batch_first=True)
                for _ in range(config.num_layers)
            ]
        )

        # Feed-forward networks
        ffn_hidden = config.embed_dim * config.ffn_multiplier
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.embed_dim, ffn_hidden),
                    nn.ReLU(),
                    nn.Linear(ffn_hidden, config.embed_dim),
                )
                for _ in range(config.num_layers)
            ]
        )

        # Layer normalizations
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(config.embed_dim) for _ in range(config.num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(config.embed_dim) for _ in range(config.num_layers)])

    def _rms_norm(self, x: torch.Tensor, variance_epsilon: float = 1e-5) -> torch.Tensor:
        """RMSNorm (no bias)."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + variance_epsilon)

    def forward(self, td: TensorDict) -> TensorDict:
        x = td[self.config.in_key]

        # Handle different input shapes - get batch size and ensure 2D input
        if x.ndim == 4:
            # Spatial input: flatten to (batch, embed_dim)
            batch_size = x.shape[0]
            x = x.view(batch_size, -1).mean(dim=1)  # Pool spatial dims
        elif x.ndim == 3:
            # Sequence input: pool to (batch, embed_dim)
            x = x.mean(dim=1)
        # else: x.ndim == 2, already (batch, embed_dim)

        batch_size = x.shape[0]
        device = x.device

        # Get reset mask from dones and truncateds like LSTM does
        dones = td.get("dones", None)
        truncateds = td.get("truncateds", None)
        if dones is not None and truncateds is not None:
            reset_mask = (dones.bool() | truncateds.bool()).view(-1, 1)
        else:
            # We're in eval mode
            reset_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)

        # Get training_env_ids to track which environment each batch element is from
        training_env_ids = td.get("training_env_ids", None)
        if training_env_ids is None:
            training_env_ids = torch.arange(batch_size, device=device)
        else:
            training_env_ids = training_env_ids.reshape(batch_size)

        # Allocate memory for new environments if needed
        max_num_envs = training_env_ids.max().item() + 1
        if "z_l" not in self.carry or self.carry["z_l"].shape[0] < max_num_envs:
            num_allocated_envs = max_num_envs - (self.carry["z_l"].shape[0] if "z_l" in self.carry else 0)
            z_l_new = torch.zeros(num_allocated_envs, self.config.embed_dim, device=device)
            z_h_new = torch.zeros(num_allocated_envs, self.config.embed_dim, device=device)
            if "z_l" in self.carry:
                self.carry["z_l"] = torch.cat([self.carry["z_l"], z_l_new], dim=0).to(device)
                self.carry["z_h"] = torch.cat([self.carry["z_h"], z_h_new], dim=0).to(device)
            else:
                self.carry["z_l"] = z_l_new
                self.carry["z_h"] = z_h_new

        # Retrieve hidden states for current environments
        z_l = self.carry["z_l"][training_env_ids]
        z_h = self.carry["z_h"][training_env_ids]

        # Reset hidden states where episodes ended
        z_l = z_l.masked_fill(reset_mask, 0)
        z_h = z_h.masked_fill(reset_mask, 0)

        # Main ACT loop with hierarchical reasoning
        segments = 0
        while True:
            # Run most cycles without gradients for efficiency
            with torch.no_grad():
                for i in range(self.config.H_cycles * self.config.L_cycles - 1):
                    z_l = self.L_level(z_l, z_h + x)
                    if (i + 1) % self.config.L_cycles == 0:
                        z_h = self.H_level(z_h, z_l)

            # Final cycle with gradients for learning
            z_l = self.L_level(z_l, z_h + x)
            z_h = self.H_level(z_h, z_l)

            # Halting decision using Q-head
            q_logits = self.q_head(z_h)  # Shape: (batch, 2)
            q_values = torch.sigmoid(q_logits)  # Q̂_halt, Q̂_continue

            # Check halting condition (halt if halt_prob > continue_prob and reached Mmin)
            if segments >= self.config.Mmax or (
                q_values[:, 0].mean() > q_values[:, 1].mean() and segments >= self.config.Mmin
            ):
                break

            segments += 1

        # Store hidden states back to memory for their respective environments
        self.carry["z_l"][training_env_ids] = z_l.detach()
        self.carry["z_h"][training_env_ids] = z_h.detach()

        td[self.config.out_key] = z_h
        return td
