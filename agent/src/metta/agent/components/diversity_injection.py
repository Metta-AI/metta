"""Diversity injection for RL policy networks.

When reward gradients vanish (stuck in local minima or flat regions), this module
automatically expands exploration of nearby representational variants by injecting
agent-specific random perturbations into the encoder output.

Key insight: when PPO loss → 0 (stuck), the diversity loss term automatically
dominates, pushing α higher and increasing representational spread across agents.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from metta.agent.components.component_config import ComponentConfig


class DiversityInjectionConfig(ComponentConfig):
    """Configuration for diversity injection layer."""

    in_key: str
    out_key: str
    name: str = "diversity_injection"

    # Number of agent slots to support (should match max agents in training)
    num_agents: int = 256

    # Low-rank approximation rank for memory efficiency
    # W = U @ V.T where U, V are (hidden_dim, rank)
    projection_rank: int = 32

    # Initial value for log_alpha (α = exp(log_alpha))
    # -1.0 means α starts at ~0.37
    log_alpha_init: float = -1.0

    # Maximum value for α to prevent explosion
    alpha_max: float = 5.0

    # Whether to apply LayerNorm after injection for stability
    use_layer_norm: bool = True

    # Key in TensorDict containing agent IDs (training_env_ids by default)
    agent_id_key: str = "training_env_ids"

    def make_component(self, env=None) -> nn.Module:
        return DiversityInjection(config=self)


class DiversityInjection(nn.Module):
    """Applies agent-specific random perturbations to encoder output.

    Architecture:
        obs → [shared encoder] → h → h + α * perturbation → [policy_head] → logits
                                                          → [value_head] → value

    Where perturbation = W_rand[agent_id] @ h using low-rank factorization.
    """

    def __init__(self, config: DiversityInjectionConfig):
        super().__init__()
        self.config = config
        self.in_key = config.in_key
        self.out_key = config.out_key
        self.agent_id_key = config.agent_id_key
        self.alpha_max = config.alpha_max

        # Learned scalar controlling perturbation strength
        self.log_alpha = nn.Parameter(torch.tensor(config.log_alpha_init))

        # Lazy initialization - we don't know hidden_dim until first forward
        self._hidden_dim: int | None = None

        # Register placeholder buffers (will be replaced on first forward)
        self.register_buffer("_projection_u", None)
        self.register_buffer("_projection_v", None)

        self.layer_norm: nn.LayerNorm | None = None

    def _initialize_projections(self, hidden_dim: int, device: torch.device, dtype: torch.dtype) -> None:
        """Initialize random projection matrices on first forward pass."""
        if self._hidden_dim == hidden_dim and self._projection_u is not None:
            # Already initialized, just ensure device matches
            if self._projection_u.device != device:
                self._projection_u = self._projection_u.to(device)
                self._projection_v = self._projection_v.to(device)
                if self.layer_norm is not None:
                    self.layer_norm = self.layer_norm.to(device)
            return

        self._hidden_dim = hidden_dim
        rank = self.config.projection_rank
        num_agents = self.config.num_agents

        # Create low-rank factorization: W = U @ V.T
        # Scale by 1/sqrt(rank) for stable initialization
        scale = 1.0 / (rank**0.5)

        # Generate deterministic random projections per agent using seeded generators
        projection_u = torch.zeros(num_agents, hidden_dim, rank, dtype=dtype, device=device)
        projection_v = torch.zeros(num_agents, rank, hidden_dim, dtype=dtype, device=device)

        for agent_idx in range(num_agents):
            gen = torch.Generator()
            gen.manual_seed(agent_idx * 31337)  # Deterministic per-agent seed
            projection_u[agent_idx] = (
                torch.randn(hidden_dim, rank, generator=gen, dtype=dtype, device="cpu").to(device) * scale
            )
            projection_v[agent_idx] = (
                torch.randn(rank, hidden_dim, generator=gen, dtype=dtype, device="cpu").to(device) * scale
            )

        # Update buffers in-place
        self._projection_u = projection_u
        self._projection_v = projection_v

        # Initialize LayerNorm if enabled
        if self.config.use_layer_norm and self.layer_norm is None:
            self.layer_norm = nn.LayerNorm(hidden_dim).to(device)

    @property
    def alpha(self) -> torch.Tensor:
        """Current perturbation strength coefficient."""
        return self.log_alpha.exp().clamp(max=self.alpha_max)

    def forward(self, td: TensorDict) -> TensorDict:
        h = td[self.in_key]  # (batch, hidden_dim) or (batch, time, hidden_dim)

        # Initialize on first forward
        self._initialize_projections(h.shape[-1], h.device, h.dtype)

        # Get agent IDs - handle both (batch,) and (batch, time) shapes
        if self.agent_id_key in td.keys():
            agent_ids = td[self.agent_id_key]
            # Flatten to 1D if needed, take first element per batch item if (batch, time)
            if agent_ids.dim() > 1:
                agent_ids = agent_ids[:, 0] if agent_ids.shape[1] > 0 else agent_ids.squeeze(-1)
            agent_ids = agent_ids.long() % self.config.num_agents
        else:
            # Default to agent 0 if no agent IDs provided (e.g., during eval)
            agent_ids = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)

        # Compute perturbation using low-rank factorization
        # h @ U @ V.T = (h @ U) @ V.T
        original_shape = h.shape
        if h.dim() == 3:
            # (batch, time, hidden) -> (batch * time, hidden)
            batch, time, hidden = h.shape
            h_flat = h.reshape(batch * time, hidden)
            # Expand agent_ids to match flattened batch
            agent_ids = agent_ids.unsqueeze(1).expand(batch, time).reshape(batch * time)
        else:
            h_flat = h
            batch, time = h.shape[0], 1

        # Gather projection matrices for each sample's agent
        # _projection_u: (num_agents, hidden_dim, rank)
        # _projection_v: (num_agents, rank, hidden_dim)
        u = self._projection_u[agent_ids]  # (batch, hidden_dim, rank)
        v = self._projection_v[agent_ids]  # (batch, rank, hidden_dim)

        # Compute perturbation: h @ U @ V.T
        # (batch, hidden) @ (batch, hidden, rank) -> (batch, rank)
        intermediate = torch.einsum("bh,bhr->br", h_flat, u)
        # (batch, rank) @ (batch, rank, hidden) -> (batch, hidden)
        perturbation = torch.einsum("br,brh->bh", intermediate, v)

        # Apply perturbation with learned coefficient
        alpha = self.alpha
        h_div = h_flat + alpha * perturbation

        # Apply LayerNorm for stability when α is large
        if self.layer_norm is not None:
            h_div = self.layer_norm(h_div)

        # Reshape back if needed
        if len(original_shape) == 3:
            h_div = h_div.reshape(original_shape)

        td[self.out_key] = h_div

        return td

    def get_diversity_loss(self) -> torch.Tensor:
        """Return diversity loss term: -log_alpha.

        This encourages α to grow when other losses are small.
        """
        return -self.log_alpha

    def extra_repr(self) -> str:
        return (
            f"in_key={self.in_key}, out_key={self.out_key}, "
            f"num_agents={self.config.num_agents}, rank={self.config.projection_rank}, "
            f"alpha_max={self.alpha_max}, use_layer_norm={self.config.use_layer_norm}"
        )
