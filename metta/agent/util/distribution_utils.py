from typing import Optional, Tuple

import torch
import torch.jit
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def sample_logits(logits: Tensor, bptt_logit_index: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits and compute log probabilities and entropy.

    Args:

        logits: Unnormalized log probabilities of shape [B*T, 2].

        bptt_logit_index: Optional tensor of shape [B*T] specifying actions to evaluate.
            If provided, it must contain valid indices in the range [0, A).

    Returns:
        A tuple of:

        action_index: Tensor of shape [B*T]
        log_prob: Log-probabilities of selected actions ()
        entropy: Entropy of the action distribution (same shape as `action`)
        normalized_logits: Log-probabilities (same shape as `logits`)
    """
    # Log-softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: [B*T, A]
    probs = torch.exp(log_probs)  # Shape: [B*T, A]

    # Sample actions or use provided ones
    if bptt_logit_index is None:
        # Multinomial returns [B*T, 1], reshape to [B*T]
        action_index = torch.multinomial(probs, num_samples=1, replacement=True).reshape(-1)  # Shape: [B*T]
    else:
        # Action is already [B*T]
        action_index = bptt_logit_index  # Shape: [B*T]

    # Gather log-probs for selected (or provided) actions
    # First reshape action to [B*T, 1] for gather, then reshape result back to [B*T]
    reshaped_action_index_out = action_index.reshape(-1, 1)  # Shape: [B*T, 1]
    action_log_prob = log_probs.gather(1, reshaped_action_index_out).reshape(-1)  # Shape: [B*T]

    # Entropy: -∑p * log(p)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # Shape: [B*T]

    return action_index, action_log_prob, entropy, log_probs
