from typing import Optional, Tuple

import torch
import torch.jit
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def sample_logits(logits: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits and compute log probabilities and entropy.

    Args:
        logits: Unnormalized log probabilities of shape [B*T, 2].
        action: Optional tensor of shape [B*T] specifying actions to evaluate.
               If provided, it must contain valid indices in the range [0, A).

    Returns:
        A tuple of:
        action: Tensor of shape [B*T]
        log_prob: Log-probabilities of selected actions ()
        entropy: Entropy of the action distribution (same shape as `action`)
        normalized_logits: Log-probabilities (same shape as `logits`)
    """
    # Log-softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: [B*T, A]
    probs = torch.exp(log_probs)  # Shape: [B*T, A]

    # Sample actions or use provided ones
    if action is None:
        # Multinomial returns [B*T, 1], reshape to [B*T]
        action_out = torch.multinomial(probs, num_samples=1, replacement=True).reshape(-1)  # Shape: [B*T]
    else:
        # Action is already [B*T]
        action_out = action  # Shape: [B*T]

    # Gather log-probs for selected (or provided) actions
    # First reshape action to [B*T, 1] for gather, then reshape result back to [B*T]
    action_indices = action_out.reshape(-1, 1)  # Shape: [B*T, 1]
    log_prob_out = log_probs.gather(1, action_indices).reshape(-1)  # Shape: [B*T]

    # Entropy: -∑p * log(p)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: [B*T]

    return action_out, log_prob_out, entropy, log_probs
