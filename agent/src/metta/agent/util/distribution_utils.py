from typing import Any, Callable, Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor

_SAMPLER_BACKENDS: Dict[
    str,
    Tuple[
        Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]],
        Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]],
    ],
] = {}
_CURRENT_SAMPLE_IMPL: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]] | None = None
_CURRENT_EVAL_IMPL: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]] | None = None


def _action_distribution(action_logits: Tensor) -> Tuple[Tensor, Tensor]:
    """Return action probabilities and log-probabilities computed from logits."""

    full_log_probs = F.log_softmax(action_logits, dim=-1)
    action_probs = torch.exp(full_log_probs)
    return action_probs, full_log_probs


def _sample_actions_eager(action_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Eager-mode implementation of action sampling."""

    action_probs, full_log_probs = _action_distribution(action_logits)
    actions = torch.multinomial(action_probs, num_samples=1).view(-1)
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    act_log_prob = full_log_probs[batch_indices, actions]
    entropy = -torch.sum(action_probs * full_log_probs, dim=-1)
    return actions, act_log_prob, entropy, full_log_probs


def _evaluate_actions_eager(action_logits: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Eager-mode implementation of action likelihood evaluation."""

    action_probs, action_log_probs = _action_distribution(action_logits)
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    log_probs = action_log_probs[batch_indices, actions]
    entropy = -torch.sum(action_probs * action_log_probs, dim=-1)
    return log_probs, entropy, action_log_probs


def _resolve_compiled_sampler(
    mode: str,
) -> Tuple[
    Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]],
    Callable[[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]],
]:
    """Return (and memoize) compiled sampling implementations for the requested mode."""

    if mode in _SAMPLER_BACKENDS:
        compiled_sample, compiled_evaluate = _SAMPLER_BACKENDS[mode]
        return compiled_sample, compiled_evaluate

    compiled_sample = torch.compile(_sample_actions_eager, mode=mode)
    compiled_evaluate = torch.compile(_evaluate_actions_eager, mode=mode)
    _SAMPLER_BACKENDS[mode] = (compiled_sample, compiled_evaluate)
    return compiled_sample, compiled_evaluate


def configure_sampling_backend(use_compile: bool, mode: str = "reduce-overhead") -> None:
    """Configure action sampling helpers for eager or compiled execution.

    Args:
        use_compile: Whether to enable ``torch.compile`` for sampling utilities.
        mode: ``torch.compile`` mode to use when compilation is enabled.
    """

    global _CURRENT_SAMPLE_IMPL, _CURRENT_EVAL_IMPL

    if not use_compile:
        _CURRENT_SAMPLE_IMPL = _sample_actions_eager
        _CURRENT_EVAL_IMPL = _evaluate_actions_eager
        return

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build but compile mode was requested")

    compiled_sample, compiled_evaluate = _resolve_compiled_sampler(mode)
    _CURRENT_SAMPLE_IMPL = compiled_sample
    _CURRENT_EVAL_IMPL = compiled_evaluate


def sample_actions(action_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Sample actions from logits using the configured backend."""

    if _CURRENT_SAMPLE_IMPL is None:
        raise RuntimeError(
            "Sampling backend is uninitialized; call configure_sampling_backend or reset_sampling_backend."
        )
    return _CURRENT_SAMPLE_IMPL(action_logits)


def evaluate_actions(action_logits: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Evaluate provided actions under the configured backend."""

    if _CURRENT_EVAL_IMPL is None:
        raise RuntimeError(
            "Sampling backend is uninitialized; call configure_sampling_backend or reset_sampling_backend."
        )
    return _CURRENT_EVAL_IMPL(action_logits, actions)


def reset_sampling_backend() -> None:
    """Reset sampling helpers to the eager implementation."""

    configure_sampling_backend(False)


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


reset_sampling_backend()
