from typing import Optional, Tuple

import torch
import torch.jit
from torch import Tensor


@torch.jit.script
def get_min_value(dtype: torch.dtype) -> float:
    """Get minimum value for a dtype in a TorchScript-compatible way."""
    if dtype == torch.float32:
        return -3.4028235e38
    elif dtype == torch.float16:
        return -65504.0
    elif dtype == torch.float64:
        return -1.7976931348623157e308
    else:
        return -1e30  # Reasonable fallback


@torch.jit.script
def sample_logits(logits: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits and compute log probabilities and entropy.

    Args:
        logits: Unnormalized log probabilities of shape [B*T, num_actions].
        action: Optional pre-specified actions of shape [B*T] or [B, T, 1].
                If provided, log probabilities and entropy for these actions are computed.

    Returns:
        Tuple of (action, log_probability, entropy, normalized_logits)
        Shapes: [B*T], [B*T], [B*T], [B*T, A] respectively.
    """
    # Input logits shape: [B*T, A]
    # Input action shape (if provided): [B*T] or [B, T, 1]
    B_times_T = logits.shape[0]
    num_actions = logits.shape[-1]

    # Normalize logits for numerical stability using logsumexp trick
    log_softmax_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = log_softmax_logits.exp()

    if action is None:
        # Sample action if not provided
        action = torch.multinomial(probs, 1, replacement=True).reshape(B_times_T)
    else:
        # Reshape provided action to ensure correct shape [B*T]
        action = action.reshape(-1)

        # Check if any actions are out of valid range
        max_action = action.max().item()
        min_action = action.min().item()

        if max_action >= num_actions or min_action < 0:
            # Clamp actions to valid range to prevent out-of-bounds errors
            action = torch.clamp(action, 0, num_actions - 1)

    # Ensure action has the expected shape [B*T]
    assert action.shape == logits.shape[:-1], f"Action shape mismatch: expected {logits.shape[:-1]}, got {action.shape}"

    # Calculate log probability for the selected action (inlined from get_action_log_prob)
    # Shape: [B*T]
    action_for_gather = action.long().unsqueeze(-1)
    act_logprob = log_softmax_logits.gather(-1, action_for_gather).squeeze(-1)

    # Calculate entropy of the distribution (inlined from entropy)
    # Shape: [B*T]
    min_real = get_min_value(log_softmax_logits.dtype)
    log_softmax_logits_clamped = torch.clamp(log_softmax_logits, min=min_real)
    logits_entropy = -torch.sum(probs * log_softmax_logits_clamped, dim=-1)

    # Return shapes: [B*T], [B*T], [B*T], [B*T, A]
    return action, act_logprob, logits_entropy, log_softmax_logits
