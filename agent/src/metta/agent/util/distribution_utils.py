from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.jit
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def _stable_action_distribution(action_logits: Tensor) -> Tuple[Tensor, Tensor]:
    """Return numerically stable action probs & log-probs while honoring masking."""

    illegal_mask = torch.isneginf(action_logits)
    non_masked_nonfinite = (~torch.isfinite(action_logits)) & (~illegal_mask)

    safe_logits = torch.where(non_masked_nonfinite, torch.zeros_like(action_logits), action_logits)

    full_log_probs = F.log_softmax(safe_logits, dim=-1)
    action_probs = torch.exp(full_log_probs)

    probs_sum = torch.sum(action_probs, dim=-1, keepdim=True)
    sum_invalid = (~torch.isfinite(probs_sum)) | (probs_sum <= 0)

    if bool(sum_invalid.any()):
        num_actions = action_probs.shape[-1]
        valid_mask = ~illegal_mask

        valid_counts = torch.sum(valid_mask, dim=-1, keepdim=True)
        valid_counts_clamped = torch.clamp(valid_counts, min=1)
        valid_counts_float = valid_counts_clamped.to(action_probs.dtype)

        fallback_valid = torch.where(
            valid_mask,
            1.0 / valid_counts_float,
            torch.zeros_like(action_probs),
        )

        uniform_all = action_probs.new_full(action_probs.shape, 1.0 / float(num_actions))
        has_valid = valid_counts > 0

        fallback_probs = torch.where(has_valid, fallback_valid, uniform_all)

        expanded_sum_invalid = sum_invalid.expand_as(action_probs)
        action_probs = torch.where(expanded_sum_invalid, fallback_probs, action_probs)

        expanded_has_valid = has_valid.expand_as(action_probs)
        zero_tensor = torch.zeros_like(action_probs)
        action_probs = torch.where(expanded_has_valid & illegal_mask, zero_tensor, action_probs)

        row_sum = torch.sum(action_probs, dim=-1, keepdim=True)
        row_sum = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
        action_probs = action_probs / row_sum

        full_log_probs = torch.log(action_probs)

    return action_probs, full_log_probs


@torch.jit.script
def sample_actions(action_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits during inference.

    Args:
        action_logits: Raw logits from policy network of shape [batch_size, num_actions].
                       These are unnormalized log-probabilities over the action space.

    Returns:
        actions: Sampled action indices of shape [batch_size]. Each element is an
                 integer in [0, num_actions) representing the sampled action.

        act_log_prob: Log-probabilities of the sampled actions, shape [batch_size].

        entropy: Policy entropy at each state, shape [batch_size].

        full_log_probs: Full log-probability distribution over all actions,
                          shape [batch_size, num_actions]. Same as log-softmax of logits.
    """
    action_probs, full_log_probs = _stable_action_distribution(action_logits)

    # Sample actions from categorical distribution (replacement=True is implicit when num_samples=1)
    actions = torch.multinomial(action_probs, num_samples=1).view(-1)  # [batch_size]

    # Extract log-probabilities for sampled actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    act_log_prob = full_log_probs[batch_indices, actions]  # [batch_size]

    # Compute policy entropy: H(π) = -∑π(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * full_log_probs, dim=-1)  # [batch_size]

    return actions, act_log_prob, entropy, full_log_probs


@torch.jit.script
def evaluate_actions(action_logits: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Evaluate provided actions against logits during training.

    Args:
        action_logits: Current policy logits of shape [batch_size, num_actions].
                       These may differ from the logits that originally generated
                       the actions due to policy updates.

        actions: Previously taken action indices of shape [batch_size].
                 Each element must be a valid action index in [0, num_actions).

    Returns:
        log_probs: Log-probabilities of the given actions under current policy,
                   shape [batch_size]. Used for importance sampling: π_new(a|s)/π_old(a|s).

        entropy: Current policy entropy at each state, shape [batch_size].

        action_log_probs: Full log-probability distribution over all actions,
                          shape [batch_size, num_actions]. Same as log-softmax of logits.
    """
    action_probs, action_log_probs = _stable_action_distribution(action_logits)

    # Extract log-probabilities for the provided actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = action_log_probs[batch_indices, actions]  # [batch_size]

    # Compute policy entropy: H(π) = -∑π(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * action_log_probs, dim=-1)  # [batch_size]

    return log_probs, entropy, action_log_probs


def get_from_master(x: Any) -> Any:
    """Broadcast value from rank 0 to all ranks in distributed training. Works for everything that can be pickled."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return x

    if isinstance(x, torch.Tensor):
        # All ranks must provide a tensor of the same shape/dtype; contents will be overwritten with rank 0's
        dist.broadcast(x, src=0)
        return x

    # Generic object path
    rank = dist.get_rank()
    obj_list = [x] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]
