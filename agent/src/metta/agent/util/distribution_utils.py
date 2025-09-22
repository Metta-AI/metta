from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def _compute_action_distribution(action_logits: Tensor) -> Tuple[Tensor, Tensor]:
    full_log_probs = F.log_softmax(action_logits, dim=-1)
    action_probs = torch.exp(full_log_probs)
    return action_probs, full_log_probs

def sample_actions(action_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    action_probs, full_log_probs = _compute_action_distribution(action_logits)
    actions = torch.multinomial(action_probs, num_samples=1).view(-1)
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    act_log_prob = full_log_probs[batch_indices, actions]
    entropy = -torch.sum(action_probs * full_log_probs, dim=-1)
    return actions, act_log_prob, entropy, full_log_probs

def evaluate_actions(action_logits: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    action_probs, action_log_probs = _compute_action_distribution(action_logits)
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = action_log_probs[batch_indices, actions]
    entropy = -torch.sum(action_probs * action_log_probs, dim=-1)
    return log_probs, entropy, action_log_probs

def get_from_master(x: Any) -> Any:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    if isinstance(x, torch.Tensor):
        dist.broadcast(x, src=0)
        return x
    rank = dist.get_rank()
    obj_list = [x] if rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]
