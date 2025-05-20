from typing import Optional, Tuple
<<<<<<< HEAD

import torch
import torch.jit
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def sample_logits(logits: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits and compute log probabilities and entropy.

    Supports input of shape [B, T, A] or [B*T, A].

    Args:
        logits: Unnormalized log probabilities, either [B, T, A] or [B*T, A].
        action: Optional tensor of shape [B, T] or [B*T] specifying actions to evaluate.
                If provided, it must contain valid indices in the range [0, A).

    Returns:
        A tuple of:
            action: Tensor of shape [B, T] or [B*T]
            log_prob: Log-probabilities of selected actions (same shape as `action`)
            entropy: Entropy of the action distribution (same shape as `action`)
            normalized_logits: Log-probabilities (same shape as `logits`)
    """

    is_batched = logits.dim() == 3  # Shape: [B, T, A]
    if is_batched:
        B, T, A = logits.shape
        logits_flat = logits.reshape(B * T, A)  # Shape: [B*T, A]
    else:
        B, T = -1, -1
        A = logits.shape[1]
        logits_flat = logits  # Shape: [B*T, A]

    # Log-softmax for numerical stability
    log_probs = F.log_softmax(logits_flat, dim=-1)  # Shape: [B*T, A]
    probs = torch.exp(log_probs)  # Shape: [B*T, A]

    # Sample actions or use provided ones
    if action is None:
        action_flat = torch.multinomial(probs, num_samples=1, replacement=True).select(1, 0)  # Shape: [B*T]
    else:
        action_flat = action.reshape(-1)  # Shape: [B*T]

    # Gather log-probs of selected actions
    log_prob_flat = log_probs.gather(1, action_flat.view(-1, 1)).view(-1)  # Shape: [B*T]

    # Entropy: -∑p * log(p)
    entropy_flat = -torch.sum(probs * log_probs, dim=-1)  # Shape: [B*T]

    # Restore shapes if batched
    if is_batched:
        action_out = action_flat.view(B, T)
        log_prob_out = log_prob_flat.view(B, T)
        entropy_out = entropy_flat.view(B, T)
        log_probs_out = log_probs.view(B, T, A)
    else:
        action_out = action_flat
        log_prob_out = log_prob_flat
        entropy_out = entropy_flat
        log_probs_out = log_probs

    return action_out, log_prob_out, entropy_out, log_probs_out
=======

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
>>>>>>> 13c12a2fdf120e435aa056c95de09aa7ccaa5a87
