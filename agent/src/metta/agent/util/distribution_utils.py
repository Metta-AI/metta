from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

_INVALID_LOG_PROB = -1e9


def _apply_valid_actions(
    action_logits: Tensor,
    valid_actions: Optional[int],
) -> Tuple[Tensor, Optional[int]]:
    if valid_actions is None or valid_actions >= action_logits.size(-1):
        return action_logits, None
    return action_logits[..., :valid_actions], valid_actions


def sample_actions(action_logits: Tensor, valid_actions: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits during inference.

    Args:
        action_logits: Raw logits from policy network of shape [batch_size, num_actions].
                       These are unnormalized log-probabilities over the action space.
        valid_actions: Optional number of actions to consider from the front of the action dimension.

    Returns:
        actions: Sampled action indices of shape [batch_size]. Each element is an
                 integer in [0, num_actions) representing the sampled action.

        act_log_prob: Log-probabilities of the sampled actions, shape [batch_size].

        entropy: Policy entropy at each state, shape [batch_size].

        full_log_probs: Full log-probability distribution over all actions,
                          shape [batch_size, num_actions]. Same as log-softmax of logits.
    """
    trimmed_logits, valid_actions = _apply_valid_actions(action_logits, valid_actions)
    trimmed_log_probs = F.log_softmax(trimmed_logits, dim=-1)  # [batch_size, valid_actions]
    action_probs = torch.exp(trimmed_log_probs)  # [batch_size, valid_actions]

    # Sample actions from categorical distribution (replacement=True is implicit when num_samples=1)
    actions = torch.multinomial(action_probs, num_samples=1).view(-1)  # [batch_size]

    # Extract log-probabilities for sampled actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    act_log_prob = trimmed_log_probs[batch_indices, actions]  # [batch_size]

    # Compute policy entropy: H(π) = -∑π(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * trimmed_log_probs, dim=-1)  # [batch_size]

    if valid_actions is None:
        full_log_probs = trimmed_log_probs
    else:
        full_log_probs = action_logits.new_full(action_logits.shape, fill_value=_INVALID_LOG_PROB)
        full_log_probs[..., :valid_actions] = trimmed_log_probs

    return actions, act_log_prob, entropy, full_log_probs


def evaluate_actions(
    action_logits: Tensor,
    actions: Tensor,
    valid_actions: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Evaluate provided actions against logits during training.

    Args:
        action_logits: Current policy logits of shape [batch_size, num_actions].
                       These may differ from the logits that originally generated
                       the actions due to policy updates.

        actions: Previously taken action indices of shape [batch_size].
                 Each element must be a valid action index in [0, num_actions).
        valid_actions: Optional number of actions to consider from the front of the action dimension.

    Returns:
        log_probs: Log-probabilities of the given actions under current policy,
                   shape [batch_size]. Used for importance sampling: π_new(a|s)/π_old(a|s).

        entropy: Current policy entropy at each state, shape [batch_size].

        action_log_probs: Full log-probability distribution over all actions,
                          shape [batch_size, num_actions]. Same as log-softmax of logits.
    """
    trimmed_logits, valid_actions = _apply_valid_actions(action_logits, valid_actions)
    trimmed_log_probs = F.log_softmax(trimmed_logits, dim=-1)  # [batch_size, valid_actions]
    action_probs = torch.exp(trimmed_log_probs)  # [batch_size, valid_actions]

    # Extract log-probabilities for the provided actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = trimmed_log_probs[batch_indices, actions]  # [batch_size]

    # Compute policy entropy: H(π) = -∑π(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * trimmed_log_probs, dim=-1)  # [batch_size]

    if valid_actions is None:
        action_log_probs = trimmed_log_probs
    else:
        action_log_probs = action_logits.new_full(action_logits.shape, fill_value=_INVALID_LOG_PROB)
        action_log_probs[..., :valid_actions] = trimmed_log_probs

    return log_probs, entropy, action_log_probs


sample_actions = torch.compile(sample_actions)
evaluate_actions = torch.compile(evaluate_actions)


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
