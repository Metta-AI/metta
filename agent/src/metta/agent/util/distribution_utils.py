from typing import Any, Callable, Dict, Tuple

import torch
import torch.distributed as dist
import torch.jit
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


def _sample_actions_eager(action_logits: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Sample actions from logits during inference in eager mode.

    Args:
        action_logits: Raw logits from policy network of shape [batch_size, num_actions].

    Returns:
        actions: Sampled action indices of shape [batch_size].
        act_log_prob: Log-probabilities of the sampled actions, shape [batch_size].
        entropy: Policy entropy for each batch element, shape [batch_size].
        full_log_probs: Full log-probability distribution over all actions, shape [batch_size, num_actions].
    """
    action_probs, full_log_probs = _stable_action_distribution(action_logits)

    actions = torch.multinomial(action_probs, num_samples=1).view(-1)
    batch_indices = torch.arange(actions.shape[0], device=actions.device)
    act_log_prob = full_log_probs[batch_indices, actions]
    entropy = -torch.sum(action_probs * full_log_probs, dim=-1)
    return actions, act_log_prob, entropy, full_log_probs


def _evaluate_actions_eager(action_logits: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Eager-mode implementation of action likelihood evaluation."""

    action_probs, action_log_probs = _stable_action_distribution(action_logits)
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


# Initialize eager backend by default so sampling works out of the box.
_CURRENT_SAMPLE_IMPL = _sample_actions_eager
_CURRENT_EVAL_IMPL = _evaluate_actions_eager


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
        configure_sampling_backend(False)
    return _CURRENT_SAMPLE_IMPL(action_logits)


def evaluate_actions(action_logits: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Evaluate provided actions under the configured backend."""

    if _CURRENT_EVAL_IMPL is None:
        configure_sampling_backend(False)
    return _CURRENT_EVAL_IMPL(action_logits, actions)


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
