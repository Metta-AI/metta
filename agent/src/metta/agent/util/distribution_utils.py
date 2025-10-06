from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.jit
import torch.nn.functional as F
from torch import Tensor


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
    # Compute probabilities in a numerically stable way and sanitize invalid entries
    action_probs = torch.softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)

    needs_fix = torch.logical_or(~torch.isfinite(action_probs), action_probs < 0)
    if needs_fix.any():
        action_probs = torch.clamp(action_probs, min=0.0)

    prob_sums = action_probs.sum(dim=-1, keepdim=True)
    zero_sum_mask = prob_sums <= 0
    if zero_sum_mask.any():
        fallback = torch.full_like(action_probs, 1.0 / action_probs.shape[-1])
        action_probs = torch.where(zero_sum_mask, fallback, action_probs)
        prob_sums = action_probs.sum(dim=-1, keepdim=True)

    action_probs = action_probs / prob_sums
    full_log_probs = torch.log(action_probs.clamp_min(1e-12))

    actions = torch.multinomial(action_probs, num_samples=1).view(-1)  # [batch_size]
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    act_log_prob = full_log_probs[batch_indices, actions]
    entropy = -torch.sum(action_probs * full_log_probs, dim=-1)

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
    action_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    action_probs = torch.exp(action_log_probs)  # [batch_size, num_actions]

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
