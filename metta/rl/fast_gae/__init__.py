import numpy as np
import torch

from .fast_gae import compute_gae as compute_gae_cpu  # type: ignore  # noqa: F403


def compute_gae_gpu(dones, values, rewards, gamma, gae_lambda):
    """PyTorch GPU implementation of Generalized Advantage Estimation (GAE)
    
    Parameters:
    -----------
    dones : torch.Tensor
        Binary flags indicating episode termination (1.0 for done, 0.0 for not done)
    values : torch.Tensor
        Value function estimates at each timestep
    rewards : torch.Tensor
        Rewards at each timestep
    gamma : float
        Discount factor
    gae_lambda : float
        GAE lambda parameter for advantage estimation
        
    Returns:
    --------
    advantages : torch.Tensor
        Calculated advantage values
    """
    if not isinstance(dones, torch.Tensor):
        return compute_gae_cpu(dones, values, rewards, gamma, gae_lambda)
        
    device = dones.device
    advantages = torch.zeros_like(values, device=device)
    
    if dones[-1].item() == 1.0:
        advantages[-1] = 0.0
    else:
        advantages[-1] = rewards[-1] - values[-1]
    
    gae = advantages[-1]
    for t in range(len(dones) - 2, -1, -1):
        next_non_terminal = 1.0 - dones[t + 1]
        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
        
    return advantages


def compute_gae(dones, values, rewards, gamma, gae_lambda):
    """Dispatch to the appropriate implementation based on input type"""
    if isinstance(dones, torch.Tensor) and dones.device.type == 'cuda':
        return compute_gae_gpu(dones, values, rewards, gamma, gae_lambda)
    else:
        return compute_gae_cpu(dones, values, rewards, gamma, gae_lambda)
