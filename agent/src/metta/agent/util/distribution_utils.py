from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def _compute_action_distribution(action_logits: Tensor) -> Tuple[Tensor, Tensor]:
    """Direct softmax conversion, matching the pre-components behaviour."""

    full_log_probs = F.log_softmax(action_logits, dim=-1)
    action_probs = torch.exp(full_log_probs)
    return action_probs, full_log_probs


def _finite_stats(t: Tensor) -> Tuple[float, float]:
    mask = torch.isfinite(t)
    if not torch.any(mask):
        return float("nan"), float("nan")
    values = t[mask]
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    return mean, std


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
    if not torch.isfinite(action_logits).all():
        mean, std = _finite_stats(action_logits)
        raise RuntimeError(f"Encountered non-finite action logits prior to sampling; statistics= mean={mean} std={std}")

    action_probs, full_log_probs = _compute_action_distribution(action_logits)

    if not torch.isfinite(action_probs).all():
        mean, std = _finite_stats(action_logits)
        raise RuntimeError(f"Non-finite action probabilities produced; logits stats= mean={mean} std={std}")

    # Sample actions from categorical distribution (replacement=True is implicit when num_samples=1)
    actions = torch.multinomial(action_probs, num_samples=1).view(-1)  # [batch_size]

    # Extract log-probabilities for sampled actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    act_log_prob = full_log_probs[batch_indices, actions]  # [batch_size]

    # Compute policy entropy: H(π) = -∑π(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * full_log_probs, dim=-1)  # [batch_size]

    return actions, act_log_prob, entropy, full_log_probs


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
    if not torch.isfinite(action_logits).all():
        mean, std = _finite_stats(action_logits)
        raise RuntimeError(f"Encountered non-finite action logits during evaluation; statistics= mean={mean} std={std}")

    action_probs, action_log_probs = _compute_action_distribution(action_logits)

    if not torch.isfinite(action_probs).all():
        mean, std = _finite_stats(action_logits)
        raise RuntimeError(
            f"Non-finite action probabilities produced during evaluation; logits stats= mean={mean} std={std}"
        )

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
