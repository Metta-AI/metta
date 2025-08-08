import torch
import torch.nn as nn


class TCEncoder(nn.Module):
    """Simple convolutional encoder for temporal consistency."""

    def __init__(self, in_channels: int = 30, latent_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(64),
            nn.Linear(64, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent vectors."""

        x = obs
        if x.dim() == 4 and x.shape[-1] == 30:
            x = x.permute(0, 3, 1, 2)
        return self.head(self.conv(x))
