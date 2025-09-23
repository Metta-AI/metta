import logging
from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


def _log_invalid_tensor(name: str, tensor: Tensor, logits: Tensor) -> None:
    invalid_mask = ~torch.isfinite(tensor)
    rows = torch.nonzero(invalid_mask.view(invalid_mask.shape[0], -1).any(dim=1), as_tuple=False)
    rows_list = rows[:5].view(-1).cpu().tolist() if rows.numel() else []
    example = None
    if rows_list:
        row_index = rows_list[0]
        example = logits[row_index].detach().cpu().reshape(-1)[:10].tolist()
    logger.error(
        "[distribution_utils] Non-finite %s detected: rows=%s example_logits_row=%s",
        name,
        rows_list,
        example,
    )


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

    full_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    if not torch.isfinite(full_log_probs).all():
        _log_invalid_tensor("full_log_probs", full_log_probs, action_logits)

    action_probs = torch.exp(full_log_probs)  # [batch_size, num_actions]
    if not torch.isfinite(action_probs).all():
        _log_invalid_tensor("action_probs", action_probs, action_logits)

    # Sample actions from categorical distribution (replacement=True is implicit when num_samples=1)
    actions = torch.multinomial(action_probs, num_samples=1).view(-1)  # [batch_size]

    # Extract log-probabilities for sampled actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    act_log_prob = full_log_probs[batch_indices, actions]  # [batch_size]

    # Compute policy entropy: H(π) = -∑π(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * full_log_probs, dim=-1)  # [batch_size]
    if not torch.isfinite(entropy).all():
        _log_invalid_tensor("entropy", entropy.unsqueeze(-1), action_logits)

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

    action_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    if not torch.isfinite(action_log_probs).all():
        _log_invalid_tensor("train_full_log_probs", action_log_probs, action_logits)

    action_probs = torch.exp(action_log_probs)  # [batch_size, num_actions]
    if not torch.isfinite(action_probs).all():
        _log_invalid_tensor("train_action_probs", action_probs, action_logits)

    # Extract log-probabilities for the provided actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = action_log_probs[batch_indices, actions]  # [batch_size]
    if not torch.isfinite(log_probs).all():
        _log_invalid_tensor("train_log_probs", log_probs.unsqueeze(-1), action_logits)

    # Compute policy entropy: H(π) = -∑π(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * action_log_probs, dim=-1)  # [batch_size]
    if not torch.isfinite(entropy).all():
        _log_invalid_tensor("train_entropy", entropy.unsqueeze(-1), action_logits)

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
