import torch
from torch import nn


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Exponential moving average update for target network parameters."""

    for tp, op in zip(target.parameters(), online.parameters(), strict=False):
        tp.data.mul_(tau).add_(op.data, alpha=1.0 - tau)
