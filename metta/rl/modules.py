"""Additional modules for representation learning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive representation learning."""

    def __init__(self, input_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return F.normalize(x, dim=-1)


class DynamicsModel(nn.Module):
    """Simple MLP dynamics model f(z_t, a_t) -> z_{t+1}."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: Tensor, a: Tensor) -> Tensor:
        x = torch.cat([z, a], dim=-1)
        return self.net(x)
