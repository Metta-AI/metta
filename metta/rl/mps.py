import torch
from torch import Tensor


def advantage(
    values: Tensor,
    rewards: Tensor,
    dones: Tensor,
    importance_sampling_ratio: Tensor,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> Tensor:
    """Native PyTorch implementation (MPS-compatible)"""
    values = values.contiguous().to(device)
    rewards = rewards.contiguous().to(device)
    dones = dones.contiguous().to(device)
    importance_sampling_ratio = importance_sampling_ratio.contiguous().to(device)

    T, B = rewards.shape
    advantages = torch.zeros_like(values, device=device)

    rho = torch.clamp(importance_sampling_ratio, max=vtrace_rho_clip)
    c = torch.clamp(importance_sampling_ratio, max=vtrace_c_clip)

    nextnonterminal = 1.0 - dones[1:]
    delta = rho[:-1] * (rewards[1:] + gamma * values[1:] * nextnonterminal - values[:-1])

    gamma_lambda = gamma * gae_lambda
    lastpufferlam = torch.zeros(B, device=device)

    for t in range(T - 2, -1, -1):
        lastpufferlam = delta[t] + gamma_lambda * c[t] * lastpufferlam * nextnonterminal[t]
        advantages[t] = lastpufferlam

    return advantages
