import torch
import torch.nn as nn


class LatentRollout(nn.Module):
    """Latent dynamics model used for temporal consistency training."""

    def __init__(
        self,
        latent_dim: int = 256,
        act_dim: int = 2,
        hidden: int = 256,
        predict_reward: bool = True,
    ) -> None:
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.predict_reward = predict_reward
        if predict_reward:
            self.r = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def step(self, z: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        z_next = self.f(torch.cat([z, a], dim=-1))
        r = self.r(z_next) if self.predict_reward else None
        return z_next, r

    def rollout(self, z0: torch.Tensor, acts: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        zs: list[torch.Tensor] = []
        rs: list[torch.Tensor] = []
        z = z0
        for i in range(k):
            z, r = self.step(z, acts[:, i])
            zs.append(z)
            if self.predict_reward and r is not None:
                rs.append(r)
        z_stack = torch.stack(zs, dim=1)
        r_stack = torch.stack(rs, dim=1) if self.predict_reward else None
        return z_stack, r_stack
