# metta/rl/loss/crl_critic.py
"""Contrastive RL Critic Network.

Implements the critic architecture from "1000 Layer Networks for Self-Supervised RL"
(Wang et al., 2025). The critic consists of:
- State-action encoder φ(s,a)
- Goal encoder ψ(g)
- Output: f(s,a,g) = ||φ(s,a) - ψ(g)||₂

Uses deep residual networks with Layer Normalization and Swish activation.
"""

from typing import Optional

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation with learnable beta parameter."""

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


class ResidualBlock(nn.Module):
    """Residual block with 4 Dense layers, Layer Normalization, and Swish activation.

    Following the architecture from the paper: each residual block consists of
    4 repeated units of (Dense → LayerNorm → Swish), with a residual connection
    applied after the final activation.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            Swish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CRLEncoder(nn.Module):
    """Deep residual encoder for CRL.

    Maps inputs to a fixed-dimensional embedding space using deep ResNets.
    The depth is defined as the total number of Dense layers across all residual blocks.
    With 4 layers per block, depth = 4 * num_blocks.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (width of the network)
        embedding_dim: Output embedding dimension
        depth: Total number of Dense layers (must be multiple of 4)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        depth: int = 4,
    ):
        super().__init__()

        if depth < 4:
            raise ValueError(f"Depth must be at least 4, got {depth}")
        if depth % 4 != 0:
            raise ValueError(f"Depth must be a multiple of 4, got {depth}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.num_blocks = depth // 4

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            Swish(),
        )

        # Deep residual blocks
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(self.num_blocks)]
        )

        # Output projection to embedding space
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to embedding space.

        Args:
            x: Input tensor of shape (..., input_dim)

        Returns:
            Embedding tensor of shape (..., embedding_dim)
        """
        h = self.input_proj(x)
        h = self.residual_layers(h)
        return self.output_proj(h)


class CRLCritic(nn.Module):
    """Contrastive RL Critic.

    The critic consists of two encoders:
    - State-action encoder φ(s,a): encodes the current state and action
    - Goal encoder ψ(g): encodes the goal state

    The critic output is the L2 distance (or negative L2 for maximization):
    f(s,a,g) = ||φ(s,a) - ψ(g)||₂

    Following the paper, we use the L2 norm for the InfoNCE objective.

    Args:
        state_dim: Dimension of state observations
        action_dim: Dimension of actions (for discrete, use embedding)
        goal_dim: Dimension of goal observations
        hidden_dim: Hidden layer dimension for both encoders
        embedding_dim: Output embedding dimension
        depth: Total depth for each encoder (number of Dense layers)
        discrete_actions: Whether actions are discrete
        num_actions: Number of discrete actions (if discrete_actions=True)
        action_embed_dim: Embedding dimension for discrete actions
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        depth: int = 4,
        discrete_actions: bool = True,
        num_actions: Optional[int] = None,
        action_embed_dim: int = 32,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.discrete_actions = discrete_actions

        # Action embedding for discrete actions
        if discrete_actions:
            if num_actions is None:
                raise ValueError("num_actions must be provided for discrete actions")
            self.action_embed = nn.Embedding(num_actions, action_embed_dim)
            sa_input_dim = state_dim + action_embed_dim
        else:
            self.action_embed = None
            sa_input_dim = state_dim + action_dim

        # State-action encoder φ(s,a)
        self.state_action_encoder = CRLEncoder(
            input_dim=sa_input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            depth=depth,
        )

        # Goal encoder ψ(g)
        self.goal_encoder = CRLEncoder(
            input_dim=goal_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            depth=depth,
        )

    def encode_state_action(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Encode state-action pair to embedding.

        Args:
            state: State tensor of shape (..., state_dim)
            action: Action tensor of shape (...) for discrete or (..., action_dim) for continuous

        Returns:
            Embedding tensor of shape (..., embedding_dim)
        """
        if self.discrete_actions:
            # Embed discrete actions
            action_emb = self.action_embed(action.long())
        else:
            action_emb = action

        # Concatenate state and action
        sa = torch.cat([state, action_emb], dim=-1)
        return self.state_action_encoder(sa)

    def encode_goal(self, goal: torch.Tensor) -> torch.Tensor:
        """Encode goal to embedding.

        Args:
            goal: Goal tensor of shape (..., goal_dim)

        Returns:
            Embedding tensor of shape (..., embedding_dim)
        """
        return self.goal_encoder(goal)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute critic output f(s,a,g) = -||φ(s,a) - ψ(g)||₂.

        Note: We return negative L2 distance so that higher values = closer to goal.
        This makes the actor objective a maximization problem.

        Args:
            state: State tensor of shape (..., state_dim)
            action: Action tensor
            goal: Goal tensor of shape (..., goal_dim)

        Returns:
            Critic values of shape (...)
        """
        phi = self.encode_state_action(state, action)
        psi = self.encode_goal(goal)

        # L2 distance
        distance = torch.norm(phi - psi, dim=-1, p=2)

        # Return negative distance (higher = better for actor)
        return -distance

    def compute_logits(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits for InfoNCE loss.

        Uses the form from the paper: f(s,a,g) = ||φ(s,a) - ψ(g)||₂

        For InfoNCE, we want similarity scores, so we use negative L2 distance.

        Args:
            state: State tensor of shape (B, state_dim)
            action: Action tensor of shape (B,) or (B, action_dim)
            goal: Goal tensor of shape (K, goal_dim) for K candidate goals

        Returns:
            Logits tensor of shape (B, K)
        """
        phi = self.encode_state_action(state, action)  # (B, embedding_dim)
        psi = self.encode_goal(goal)  # (K, embedding_dim)

        # Compute pairwise L2 distances: (B, K)
        # phi: (B, 1, D), psi: (1, K, D) -> distances: (B, K)
        phi_expanded = phi.unsqueeze(1)  # (B, 1, D)
        psi_expanded = psi.unsqueeze(0)  # (1, K, D)

        distances = torch.norm(phi_expanded - psi_expanded, dim=-1, p=2)  # (B, K)

        # Return negative distances as logits (higher = more similar)
        return -distances
