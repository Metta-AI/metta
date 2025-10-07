import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class HRMObsEncodingConfig(ComponentConfig):
    """HRM observation encoding component that processes grid observations."""

    class_path: str = "metta.agent.components.hrm.HRMObsEncoding"
    in_key: str = "env_obs"
    out_key: str = "hrm_obs_encoded"
    name: str = "hrm_obs_encoding"
    embed_dim: int = 256
    out_width: int = 11
    out_height: int = 11

    def make_component(self, env=None) -> "HRMObsEncoding":
        return HRMObsEncoding(self)


class HRMObsEncoding(nn.Module):
    """HRM observation encoding component."""

    def __init__(self, config: HRMObsEncodingConfig):
        super().__init__()
        self.config = config

        # Create coordinate embeddings like in the original HRM
        self.coord_embed_x = nn.Parameter(torch.randn(config.out_width, config.embed_dim))
        self.coord_embed_y = nn.Parameter(torch.randn(config.out_height, config.embed_dim))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.embed_dim)

    def forward(self, td: TensorDict) -> TensorDict:
        obs = td[self.config.in_key]
        batch_size = obs.shape[0]

        # Create coordinate grid
        x_coords = torch.arange(self.config.out_width, device=obs.device)
        y_coords = torch.arange(self.config.out_height, device=obs.device)

        # Get coordinate embeddings
        x_embed = self.coord_embed_x[x_coords]  # (W, embed_dim)
        y_embed = self.coord_embed_y[y_coords]  # (H, embed_dim)

        # Broadcast and combine coordinates
        x_embed = x_embed.unsqueeze(0).expand(self.config.out_height, -1, -1)  # (H, W, embed_dim)
        y_embed = y_embed.unsqueeze(1).expand(-1, self.config.out_width, -1)  # (H, W, embed_dim)

        # Combine coordinate embeddings
        coord_embed = x_embed + y_embed  # (H, W, embed_dim)
        coord_embed = coord_embed.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, H, W, embed_dim)

        # Apply layer norm
        encoded_obs = self.layer_norm(coord_embed)

        td[self.config.out_key] = encoded_obs
        return td


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
    """HRM reasoning component using transformer layers."""

    def __init__(self, config: HRMReasoningConfig):
        super().__init__()
        self.config = config

        self.memory = HRMMemory()
        self.carry = self.memory.get_memory()

        # Multi-head attention layers
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

        if x.ndim == 2:
            # Already pooled, use directly
            reasoning_output = x
        else:
            if x.ndim == 4:
                # Flatten spatial dimensions for attention
                batch_size, height, width, embed_dim = x.shape
                x = x.view(batch_size, height * width, embed_dim)
            # else: already in (batch_size, sequence_length, embed_dim) format

            # Apply transformer layers
            for i in range(self.config.num_layers):
                # Self-attention with residual
                x_norm = self.layer_norms1[i](x)
                attn_out, _ = self.attention_layers[i](x_norm, x_norm, x_norm)
                x = x + attn_out

                # Feed-forward with residual
                x_norm = self.layer_norms2[i](x)
                ffn_out = self.ffn_layers[i](x_norm)
                x = x + ffn_out

                # Apply RMS norm
                x = self._rms_norm(x)

        # Handle different input shapes:
        # - (B, D): Already pooled by upstream component, skip transformer
        # - (B, N, D): Sequence input, apply transformer
        # - (B, H, W, D): Spatial input, flatten and apply transformer

        # Main ACT loop (for inference; training adds deep supervision)

        if self.carry is not None:
            self.z_l = self.carry["z_l"]
            self.z_h = self.carry["z_h"]
        else:
            self.z_l = torch.zeros(x.shape[0], self.config.embed_dim)
            self.z_h = torch.zeros(x.shape[0], self.config.embed_dim)

        halt = False
        segments = 0
        while True:
            with torch.no_grad():
                for i in range(self.config.H_cycles * self.config.L_cycles - 1):  # Most timesteps no_grad
                    z_l = self.L_level(z_l, z_h + x)
                    if (i + 1) % self.config.L_cycles == 0:
                        z_h = self.H_level(z_h, z_l)

            # Final 1-step with gradients (for approx gradient)
            z_l = self.L_level(z_l, z_h + x)
            z_h = self.H_level(z_h, z_l)

            # Q-head: sigmoid on linear projection of z_h (two values: halt, continue)
            q_logits = self.q_head(z_h)  # Shape: [2]
            q_values = torch.sigmoid(q_logits)  # Q̂_halt, Q̂_continue

            # Halting decision (with Mmin/Mmax)
            if segments >= self.config.Mmax or (
                q_values[0] > q_values[1] and segments >= self.config.Mmin
            ):  # halt=0, continue=1
                break  # Use y_hat as final prediction

            segments += 1

        td[self.config.out_key] = z_h
        return td


class HRMActorConfig(ComponentConfig):
    """HRM actor component that generates action logits."""

    class_path: str = "metta.agent.components.hrm.HRMActor"
    in_key: str = "hrm_reasoning"
    out_key: str = "logits"
    name: str = "hrm_actor"
    embed_dim: int = 256
    num_actions: int | None = None

    def make_component(self, env=None) -> "HRMActor":
        # Calculate total number of logits needed from max_action_args
        if env is not None and hasattr(env, "max_action_args"):
            # For MultiDiscrete action space, sum up all possible actions
            self.num_actions = sum(max_arg + 1 for max_arg in env.max_action_args)
        elif env is not None and hasattr(env, "action_space"):
            if hasattr(env.action_space, "n"):
                self.num_actions = env.action_space.n
            elif hasattr(env.action_space, "nvec"):
                # Fallback for MultiDiscrete without max_action_args
                self.num_actions = sum(env.action_space.nvec)

        if self.num_actions is None:
            raise ValueError(
                "Could not determine num_actions from environment. "
                "Environment must have max_action_args or action_space attribute."
            )

        return HRMActor(self)


class HRMActor(nn.Module):
    """HRM actor component that outputs action logits."""

    def __init__(self, config: HRMActorConfig):
        super().__init__()
        self.config = config

        # Action projection layers
        self.action_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim), nn.ReLU(), nn.Linear(config.embed_dim, config.num_actions)
        )

    def forward(self, td: TensorDict) -> TensorDict:
        reasoning_output = td[self.config.in_key]

        # Generate action logits
        logits = self.action_proj(reasoning_output)

        # Clamp to prevent NaN/inf
        logits = torch.clamp(logits, min=-10.0, max=10.0)

        td[self.config.out_key] = logits
        return td


class HRMCriticConfig(ComponentConfig):
    """HRM critic component that generates value estimates."""

    class_path: str = "metta.agent.components.hrm.HRMCritic"
    in_key: str = "hrm_reasoning"
    out_key: str = "values"
    name: str = "hrm_critic"
    embed_dim: int = 256

    def make_component(self, env=None) -> "HRMCritic":
        return HRMCritic(self)


class HRMCritic(nn.Module):
    """HRM critic component that outputs value estimates."""

    def __init__(self, config: HRMCriticConfig):
        super().__init__()
        self.config = config

        # Value projection layers
        self.value_proj = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim), nn.ReLU(), nn.Linear(config.embed_dim, 1)
        )

    def forward(self, td: TensorDict) -> TensorDict:
        reasoning_output = td[self.config.in_key]

        # Generate value estimates
        values = self.value_proj(reasoning_output)

        td[self.config.out_key] = values
        return td
