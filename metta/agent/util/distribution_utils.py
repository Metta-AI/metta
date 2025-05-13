from typing import Optional, Tuple

import torch
import torch.jit
from torch import Tensor


@torch.jit.script
def sample_logits(logits: Tensor, action: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from unnormalized logits and compute associated log-probabilities and entropy.
    Optimized for performance, numerical stability, and TorchScript compatibility.
    Uses explicit reshaping operations to maintain strict control over tensor dimensions.

    Args:
        logits: Unnormalized logits of shape [batch_size, num_actions]
        action: Optional pre-specified actions of shape [batch_size] or [batch_size, 1]
                If provided, log probabilities and entropy for these actions are computed.

    Returns:
        Tuple of (actions, log_probability, entropy, normalized_logits)
        Shapes: [batch_size], [batch_size], [batch_size], [batch_size, num_actions] respectively.
    """
    batch_size, num_actions = logits.shape

    # Normalize logits using logsumexp for numerical stability (equivalent to log_softmax)
    log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)

    # Determine actions: either sample from distribution or use provided actions
    if action is None:
        # For sampling, we need the actual probabilities
        probs = torch.exp(log_probs)
        output_action = torch.multinomial(probs, 1)
    else:
        # Always reshape to [batch_size, 1] to ensure consistent dimensionality
        if action.dim() == 1:
            # If [batch_size] -> [batch_size, 1]
            output_action = action.reshape(batch_size, 1)
        else:
            # Handle case where action might be [batch_size, k] or other shape
            output_action = action.reshape(batch_size, -1)
            # Take only first dimension if multi-dimensional
            output_action = output_action[:, :1]

        # Safety check: ensure actions are within valid range
        if torch.jit.is_scripting():
            # Explicit bounds checking for TorchScript
            max_action = output_action.max().item()
            min_action = output_action.min().item()
            if max_action >= num_actions or min_action < 0:
                output_action = torch.clamp(output_action, 0, num_actions - 1)
        else:
            # Can use more efficient operations when not in TorchScript
            output_action = torch.clamp(output_action, 0, num_actions - 1)

    # Ensure indices are proper long type
    indices = output_action.long()

    # Compute log probabilities of selected actions
    # Using gather is more efficient than indexing for batched operations
    joint_logprob_2d = log_probs.gather(dim=-1, index=indices)
    # Explicitly reshape to [batch_size] instead of using squeeze
    joint_logprob = joint_logprob_2d.reshape(batch_size)

    # Compute entropy: -sum(p * log(p))
    # Using a fixed minimum value for numerical stability for torch script
    # -20 is a reasonable minimum for log probabilities (exp(-20) ≈ 2.06e-9)
    min_log_prob = -20.0
    safe_log_probs = torch.clamp(log_probs, min=min_log_prob)
    probs = torch.exp(safe_log_probs)
    joint_entropy = -torch.sum(probs * safe_log_probs, dim=-1)

    # Explicitly reshape output_action to [batch_size] from [batch_size, 1]
    final_action = output_action.reshape(batch_size)

    # Return all outputs with proper shapes
    return final_action, joint_logprob, joint_entropy, log_probs
