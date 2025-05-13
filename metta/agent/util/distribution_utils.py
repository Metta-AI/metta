from typing import Optional, Tuple

import torch
import torch.jit
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def get_min_value(dtype: torch.dtype) -> float:
    """
    Get a safe clamping minimum for a given dtype.
    These are slightly conservative to avoid overflows in TorchScript.
    """
    if dtype == torch.float32:
        return -1e19
    elif dtype == torch.float16:
        return -1e4  # torch.float16 can't safely represent very large values
    elif dtype == torch.float64:
        return -1e300  # Avoid 1.79e308 to stay within safe casting bounds
    else:
        return -1e10  # Safe fallback for other types


@torch.jit.script
def sample_logits(logits: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits and compute log probabilities and entropy.

    Supports logits of shape [B, T, A] or [B*T, A].

    Args:
        logits: Unnormalized log probabilities of shape [B, T, A] or [B*T, A]
        action: Optional action tensor of shape [B, T] or [B*T]

    Returns:
        Tuple:
            action: [B, T] or [B*T]
            log_prob: [B, T] or [B*T]
            entropy: [B, T] or [B*T]
            normalized_logits (log-probs): [B, T, A] or [B*T, A]
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
        max_action = torch.max(action_flat)
        min_action = torch.min(action_flat)
        if bool(max_action >= A) or bool(min_action < 0):
            action_flat = torch.clamp(action_flat, 0, A - 1)

    # Gather log-probs of selected actions (using gather for TorchScript compatibility)
    log_prob_flat = log_probs.gather(1, action_flat.view(-1, 1)).view(-1)  # Shape: [B*T]

    # Clamp for entropy stability
    min_real = get_min_value(log_probs.dtype)
    min_tensor = torch.tensor(min_real, dtype=log_probs.dtype, device=logits.device)
    clamped_log_probs = torch.where(log_probs < min_tensor, min_tensor, log_probs)

    # Entropy: -∑p * log(p)
    entropy_flat = -torch.sum(probs * clamped_log_probs, dim=-1)  # Shape: [B*T]

    # Reshape outputs back to [B, T] if input was batched
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
