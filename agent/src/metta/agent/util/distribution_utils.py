from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.jit
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def sample_actions(action_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample actions from logits and return actions, log-probs, entropy, and full distribution."""

    full_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    action_probs = torch.exp(full_log_probs)  # [batch_size, num_actions]

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
    """Evaluate provided actions against current policy logits for importance sampling."""

    action_log_probs = F.log_softmax(action_logits, dim=-1)  # [batch_size, num_actions]
    action_probs = torch.exp(action_log_probs)  # [batch_size, num_actions]

    # Extract log-probabilities for the provided actions using advanced indexing
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = action_log_probs[batch_indices, actions]  # [batch_size]

    # Compute policy entropy: H(π) = -∑π(a|s)log π(a|s)
    entropy = -torch.sum(action_probs * action_log_probs, dim=-1)  # [batch_size]

    return log_probs, entropy, action_log_probs


def get_from_master(x: Any) -> Any:
    """Broadcast value from rank 0 to all ranks in distributed training."""
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
