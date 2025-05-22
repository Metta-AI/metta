from typing import Tuple

import torch
import torch.jit
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def sample_actions(logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits during inference.

    Args:
        logits: Unnormalized log probabilities of shape [B*T, A].

    Returns:
        A tuple of:
        action_index: Tensor of shape [B*T]
        log_prob: Log-probabilities of selected actions (shape [B*T])
        entropy: Entropy of the action distribution (shape [B*T])
        normalized_logits: Log-probabilities (same shape as `logits`)
    """
    # Log-softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: [B*T, A]
    probs = torch.exp(log_probs)  # Shape: [B*T, A]

    # Sample actions
    action_index = torch.multinomial(probs, num_samples=1, replacement=True).reshape(-1)  # Shape: [B*T]

    # Gather log-probs for selected actions
    reshaped_action_index = action_index.reshape(-1, 1)  # Shape: [B*T, 1]
    action_log_prob = log_probs.gather(1, reshaped_action_index).reshape(-1)  # Shape: [B*T]

    # Entropy: -∑p * log(p)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: [B*T]

    return action_index, action_log_prob, entropy, log_probs


@torch.jit.script
def evaluate_actions(logits: Tensor, action_index: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Evaluate provided actions against logits during training.

    Args:
        logits: Unnormalized log probabilities of shape [B*T, A].
        action_index: Tensor of shape [B*T] specifying actions to evaluate.
            Must contain valid indices in the range [0, A).

    Returns:
        A tuple of:
        log_prob: Log-probabilities of provided actions (shape [B*T])
        entropy: Entropy of the action distribution (shape [B*T])
        normalized_logits: Log-probabilities (same shape as `logits`)
    """
    # Log-softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: [B*T, A]
    probs = torch.exp(log_probs)  # Shape: [B*T, A]

    # Gather log-probs for provided actions
    reshaped_action_index = action_index.reshape(-1, 1)  # Shape: [B*T, 1]
    action_log_prob = log_probs.gather(1, reshaped_action_index).reshape(-1)  # Shape: [B*T]

    # Entropy: -∑p * log(p)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: [B*T]

    return action_log_prob, entropy, log_probs
