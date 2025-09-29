
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

    def make_component(self, env=None) -> "HRMReasoning":
        return HRMReasoning(self)


class HRMReasoning(nn.Module):
    """HRM reasoning component using transformer layers."""

    def __init__(self, config: HRMReasoningConfig):
        super().__init__()
        self.config = config

        # Multi-head attention layers
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_heads, batch_first=True)
                for _ in range(config.num_layers)
            ]
        )

        # Feed-forward networks
        self.ffn_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.embed_dim, config.embed_dim * 4),
                    nn.ReLU(),
                    nn.Linear(config.embed_dim * 4, config.embed_dim),
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

        # Flatten spatial dimensions for attention
        batch_size, height, width, embed_dim = x.shape
        x = x.view(batch_size, height * width, embed_dim)

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

        # Global average pooling
        reasoning_output = x.mean(dim=1)  # (batch_size, embed_dim)

        td[self.config.out_key] = reasoning_output
        return td


class HRMActorConfig(ComponentConfig):
    """HRM actor component that generates action logits."""

    class_path: str = "metta.agent.components.hrm.HRMActor"
    in_key: str = "hrm_reasoning"
    out_key: str = "logits"
    name: str = "hrm_actor"
    embed_dim: int = 256
    num_actions: int = 100

    def make_component(self, env=None) -> "HRMActor":
        # Use actual action space size if environment is provided
        if env is not None and hasattr(env, "num_actions"):
            self.num_actions = env.num_actions
        elif env is not None and hasattr(env, "action_space") and hasattr(env.action_space, "n"):
            self.num_actions = env.action_space.n
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
