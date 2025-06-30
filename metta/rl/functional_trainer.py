"""Functional training utilities for Metta.

This module provides functional implementations of the core training loop components,
extracting the rollout and train logic from MettaTrainer into standalone functions.
"""

import logging
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib
from torch import Tensor

from metta.agent.policy_state import PolicyState
from metta.mettagrid.mettagrid_env import dtype_actions
from metta.mettagrid.util.dict_utils import unroll_nested_dict
from metta.rl.experience import Experience

logger = logging.getLogger(__name__)


def process_rollout_infos(raw_infos: list) -> Dict[str, Any]:
    """Process raw info dictionaries from the environment."""
    infos = defaultdict(list)
    for i in raw_infos:
        for k, v in unroll_nested_dict(i):
            infos[k].append(v)

    stats = {}
    for k, v in infos.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()

        if isinstance(v, list):
            stats.setdefault(k, []).extend(v)
        else:
            stats[k] = [stats[k], v] if k in stats else v
    return stats


def perform_rollout_step(
    policy: torch.nn.Module,
    vecenv: Any,
    experience: Experience,
    device: torch.device,
    timer: Optional[Any],
) -> Tuple[int, list, int]:
    """Performs a single step of the rollout, interacting with the environment."""
    with timer("_rollout.env") if timer else nullcontext():
        o, r, d, t, info, env_id, mask = vecenv.recv()
        training_env_id = slice(env_id[0], env_id[-1] + 1)

    mask = torch.as_tensor(mask)
    num_steps = int(mask.sum().item())

    # Convert to tensors
    tensors = [o, r, d, t]
    o, r, d, t = [torch.as_tensor(x).to(device, non_blocking=True) for x in tensors]

    with torch.no_grad():
        state = PolicyState()
        lstm_h, lstm_c = experience.get_lstm_state(training_env_id.start)
        if lstm_h is not None:
            state.lstm_h = lstm_h
            state.lstm_c = lstm_c

        actions, selected_action_log_probs, _, value, _ = policy(o, state)

        lstm_state_to_store = None
        if state.lstm_h is not None:
            lstm_state_to_store = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c}

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    experience.store(
        obs=o,
        actions=actions,
        logprobs=selected_action_log_probs,
        rewards=r,
        dones=d,
        truncations=t,
        values=value.flatten(),
        env_id=training_env_id,
        mask=mask,
        lstm_state=lstm_state_to_store,
    )

    with timer("_rollout.env") if timer else nullcontext():
        vecenv.send(actions.cpu().numpy().astype(dtype_actions))

    return num_steps, info, 0


def compute_advantage(
    values: Tensor,
    rewards: Tensor,
    dones: Tensor,
    importance_sampling_ratio: Tensor,
    advantages: Tensor,
    gamma: float,
    gae_lambda: float,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
    device: torch.device,
) -> Tensor:
    """CUDA kernel for puffer advantage with automatic CPU fallback."""
    # Move tensors to device
    tensors = [values, rewards, dones, importance_sampling_ratio, advantages]
    tensors = [t.to(device) for t in tensors]
    values, rewards, dones, importance_sampling_ratio, advantages = tensors

    # Create context manager that only applies CUDA device context if needed
    device_context = torch.cuda.device(device) if str(device).startswith("cuda") else nullcontext()
    with device_context:
        torch.ops.pufferlib.compute_puff_advantage(
            values,
            rewards,
            dones,
            importance_sampling_ratio,
            advantages,
            gamma,
            gae_lambda,
            vtrace_rho_clip,
            vtrace_c_clip,
        )

    return advantages


def normalize_advantage_distributed(adv: Tensor, norm_adv: bool = True) -> Tensor:
    """Normalize advantages with distributed training support while preserving shape."""
    if not norm_adv:
        return adv

    if torch.distributed.is_initialized():
        # Compute local statistics
        adv_flat = adv.view(-1)
        local_stats = torch.tensor(
            [adv_flat.sum().item(), (adv_flat * adv_flat).sum().item(), adv_flat.numel()],
            dtype=adv.dtype,
            device=adv.device,
        )

        # All-reduce statistics
        torch.distributed.all_reduce(local_stats, op=torch.distributed.ReduceOp.SUM)

        # Extract global statistics
        global_sum, global_sq_sum, global_count = local_stats
        global_mean = global_sum / global_count
        global_var = (global_sq_sum / global_count) - (global_mean * global_mean)
        global_std = torch.sqrt(global_var.clamp(min=1e-8))

        # Normalize
        adv = (adv - global_mean) / (global_std + 1e-8)
    else:
        # Local normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return adv


def compute_initial_advantages(
    experience: Experience,
    gamma: float,
    gae_lambda: float,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
    device: torch.device,
) -> Tensor:
    """Computes initial advantages before the training loop."""
    advantages = torch.zeros(experience.values.shape, device=device)
    initial_importance_sampling_ratio = torch.ones_like(experience.values)
    return compute_advantage(
        experience.values,
        experience.rewards,
        experience.dones,
        initial_importance_sampling_ratio,
        advantages,
        gamma,
        gae_lambda,
        vtrace_rho_clip,
        vtrace_c_clip,
        device,
    )
