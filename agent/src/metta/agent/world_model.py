"""
Here we will implement a world model to encode the observation.
"""

import torch.nn.functional as F
from torch import Tensor, nn


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # encoder layers
        self.l1 = nn.Linear(200 * 3, 2048)
        self.l2 = nn.Linear(2048, 2048)
        self.l3 = nn.Linear(2048, 1024)
        self.l4 = nn.Linear(1024, latent_dim)

        # decoder layers
        self.l5 = nn.Linear(latent_dim, 1024)
        self.l6 = nn.Linear(1024, 2048)
        self.l7 = nn.Linear(2048, 2048)
        self.l8 = nn.Linear(2048, 200 * 3)

    def encode(self, obs: Tensor) -> Tensor:
        # Accept [B, 200, 3] or already-flattened [B, 600]
        if obs.dim() == 3:
            obs = obs.reshape(obs.shape[0], -1)
        # Ensure floating dtype for linear layers
        obs = obs.float()
        x = self.l1(obs)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        x = F.relu(x)
        x = self.l4(x)
        return x

    def decode(self, latent: Tensor) -> Tensor:
        x = self.l5(latent)
        x = F.relu(x)
        x = self.l6(x)
        x = F.relu(x)
        x = self.l7(x)
        x = F.relu(x)
        x = self.l8(x)
        return x

    def forward(self, obs: Tensor) -> Tensor:
        """
        We will use the forward method for combined encode and decode.
        This method will be used for the supervised training.
        """
        x = self.decode(self.encode(obs))
        # Return the same shape as input if input was [B, 200, 3]
        if obs.dim() == 3:
            x = x.view(obs.shape[0], 200, 3)
        return x
