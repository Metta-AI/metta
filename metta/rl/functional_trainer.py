"""Functional training utilities for Metta.

This module provides functional implementations of the core training loop components,
extracting the rollout and train logic from MettaTrainer into standalone functions.
"""

import logging
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import einops
import torch
import torch.distributed
from torch import Tensor

from metta.agent.policy_state import PolicyState
from metta.mettagrid.mettagrid_env import dtype_actions
from metta.mettagrid.util.dict_utils import unroll_nested_dict
from metta.rl.experience import Experience
from metta.rl.losses import Losses

logger = logging.getLogger(__name__)


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
    # Move tensors to device and compute advantage
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
        local_sum = einops.rearrange(adv_flat.sum(), "-> 1")
        local_sq_sum = einops.rearrange((adv_flat * adv_flat).sum(), "-> 1")
        local_count = torch.tensor([adv_flat.numel()], dtype=adv.dtype, device=adv.device)

        # Combine statistics for single all_reduce
        stats = einops.rearrange([local_sum, local_sq_sum, local_count], "one float -> (float one)")
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        # Extract global statistics
        global_sum, global_sq_sum, global_count = stats[0], stats[1], stats[2]
        global_mean = global_sum / global_count
        global_var = (global_sq_sum / global_count) - (global_mean * global_mean)
        global_std = torch.sqrt(global_var.clamp(min=1e-8))

        # Normalize and reshape back
        adv = (adv - global_mean) / (global_std + 1e-8)
    else:
        # Local normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return adv


def rollout(
    policy: torch.nn.Module,
    vecenv: Any,
    experience: Experience,
    device: torch.device,
    agent_step: int,
) -> Tuple[int, Dict[str, Any]]:
    """Perform a rollout to fill the experience buffer.

    Args:
        policy: The policy network
        vecenv: Vectorized environment
        experience: Experience buffer to fill
        device: Device to run on
        agent_step: Current agent step count

    Returns:
        Tuple of (new_agent_step, stats_dict)
    """
    stats = defaultdict(list)
    raw_infos = []
    experience.reset_for_rollout()

    while not experience.ready_for_training:
        o, r, d, t, info, env_id, mask = vecenv.recv()
        training_env_id = slice(env_id[0], env_id[-1] + 1)

        # Convert mask to tensor once
        mask = torch.as_tensor(mask)
        num_steps = int(mask.sum().item())
        agent_step += num_steps

        # Convert to tensors once
        o = torch.as_tensor(o).to(device, non_blocking=True)
        r = torch.as_tensor(r).to(device, non_blocking=True)
        d = torch.as_tensor(d).to(device, non_blocking=True)
        t = torch.as_tensor(t).to(device, non_blocking=True)

        with torch.no_grad():
            state = PolicyState()

            # Use LSTM state access for performance
            lstm_h, lstm_c = experience.get_lstm_state(training_env_id.start)
            if lstm_h is not None:
                state.lstm_h = lstm_h
                state.lstm_c = lstm_c

            # Use pre-moved tensor
            actions, selected_action_log_probs, _, value, _ = policy(o, state)

            # Store LSTM state for performance
            lstm_state_to_store = None
            if state.lstm_h is not None:
                lstm_state_to_store = {"lstm_h": state.lstm_h, "lstm_c": state.lstm_c}

            if str(device).startswith("cuda"):
                torch.cuda.synchronize()

        value = value.flatten()

        # All tensors are already on device, avoid redundant transfers
        experience.store(
            obs=o,
            actions=actions,
            logprobs=selected_action_log_probs,
            rewards=r,
            dones=d,
            truncations=t,
            values=value,
            env_id=training_env_id,
            mask=mask,
            lstm_state=lstm_state_to_store,
        )

        if info:
            raw_infos.extend(info)

        vecenv.send(actions.cpu().numpy().astype(dtype_actions))

    # Batch process info dictionaries after rollout
    for i in raw_infos:
        for k, v in unroll_nested_dict(i):
            stats[k].append(v)

    return agent_step, dict(stats)


def train_ppo(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    experience: Experience,
    device: torch.device,
    losses: Losses,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    norm_adv: bool = True,
    clip_vloss: bool = True,
    vf_clip_coef: float = 0.1,
    update_epochs: int = 4,
    target_kl: Optional[float] = None,
    kickstarter=None,
    agent_step: int = 0,
    l2_reg_loss_coef: float = 0.0,
    l2_init_loss_coef: float = 0.0,
    clip_range: float = 0.0,
    prio_alpha: float = 0.0,
    prio_beta0: float = 0.6,
    total_timesteps: int = 1_000_000,
    vtrace_rho_clip: float = 1.0,
    vtrace_c_clip: float = 1.0,
) -> None:
    """Train the policy using PPO.

    Args:
        policy: The policy network
        optimizer: The optimizer
        experience: Experience buffer with collected data
        device: Device to run on
        losses: Losses object to track training metrics
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_coef: PPO clipping coefficient
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Maximum gradient norm for clipping
        norm_adv: Whether to normalize advantages
        clip_vloss: Whether to clip value loss
        vf_clip_coef: Value function clipping coefficient
        update_epochs: Number of PPO update epochs
        target_kl: Target KL divergence for early stopping
        kickstarter: Kickstarter object for teacher distillation
        agent_step: Current agent step
        l2_reg_loss_coef: L2 regularization coefficient
        l2_init_loss_coef: L2 initialization loss coefficient
        clip_range: Weight clipping range
        prio_alpha: Prioritized experience replay alpha
        prio_beta0: Prioritized experience replay beta0
        total_timesteps: Total training timesteps
        vtrace_rho_clip: V-trace rho clipping
        vtrace_c_clip: V-trace c clipping
    """
    losses.zero()

    # Reset importance sampling ratios
    experience.reset_importance_sampling_ratios()

    # Prioritized sampling parameters
    batch_size = experience.batch_size
    epoch = agent_step // batch_size  # Approximate epoch
    total_epochs = max(1, total_timesteps // batch_size)
    anneal_beta = prio_beta0 + (1 - prio_beta0) * prio_alpha * epoch / total_epochs

    # Compute advantages using puff_advantage
    advantages = torch.zeros(experience.values.shape, device=device)

    # Initial importance sampling ratio is all ones
    initial_importance_sampling_ratio = torch.ones_like(experience.values)

    advantages = compute_advantage(
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

    # Optimizing the policy and value network
    _total_minibatches = experience.num_minibatches * update_epochs
    minibatch_idx = 0

    for _epoch in range(update_epochs):
        for _ in range(experience.num_minibatches):
            minibatch = experience.sample_minibatch(
                advantages=advantages,
                prio_alpha=prio_alpha,
                prio_beta=anneal_beta,
                minibatch_idx=minibatch_idx,
                total_minibatches=_total_minibatches,
            )

            obs = minibatch["obs"]

            lstm_state = PolicyState()
            _, new_logprobs, entropy, newvalue, full_logprobs = policy(obs, lstm_state, action=minibatch["actions"])

            new_logprobs = new_logprobs.reshape(minibatch["logprobs"].shape)
            logratio = new_logprobs - minibatch["logprobs"]
            importance_sampling_ratio = logratio.exp()
            experience.update_ratio(minibatch["indices"], importance_sampling_ratio)

            with torch.no_grad():
                approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
                clipfrac = ((importance_sampling_ratio - 1.0).abs() > clip_coef).float().mean()

            # Re-compute advantages with new ratios (V-trace)
            adv = compute_advantage(
                minibatch["values"],
                minibatch["rewards"],
                minibatch["dones"],
                importance_sampling_ratio,
                minibatch["advantages"],
                gamma,
                gae_lambda,
                vtrace_rho_clip,
                vtrace_c_clip,
                device,
            )

            # Normalize advantages with distributed support, then apply prioritized weights
            adv = normalize_advantage_distributed(adv, norm_adv)
            adv = minibatch["prio_weights"] * adv

            # Policy loss
            pg_loss1 = -adv * importance_sampling_ratio
            pg_loss2 = -adv * torch.clamp(importance_sampling_ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
            if clip_vloss:
                v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
                v_clipped = minibatch["values"] + torch.clamp(
                    newvalue_reshaped - minibatch["values"],
                    -vf_clip_coef,
                    vf_clip_coef,
                )
                v_loss_clipped = (v_clipped - minibatch["returns"]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
            else:
                v_loss = 0.5 * ((newvalue_reshaped - minibatch["returns"]) ** 2).mean()

            entropy_loss = entropy.mean()

            # Kickstarter losses
            ks_action_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            ks_value_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            if kickstarter is not None:
                ks_action_loss, ks_value_loss = kickstarter.loss(
                    agent_step, full_logprobs, newvalue, obs, teacher_lstm_state=[]
                )

            # L2 regularization losses
            l2_reg_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            if l2_reg_loss_coef > 0 and hasattr(policy, "l2_reg_loss"):
                l2_reg_loss = l2_reg_loss_coef * policy.l2_reg_loss().to(device)

            l2_init_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            if l2_init_loss_coef > 0 and hasattr(policy, "l2_init_loss"):
                l2_init_loss = l2_init_loss_coef * policy.l2_init_loss().to(device)

            loss = (
                pg_loss
                - ent_coef * entropy_loss
                + v_loss * vf_coef
                + l2_reg_loss
                + l2_init_loss
                + ks_action_loss
                + ks_value_loss
            )

            # Update values in experience buffer
            experience.update_values(minibatch["indices"], newvalue.view(minibatch["values"].shape))

            # Update loss tracking for logging
            losses.policy_loss_sum += pg_loss.item()
            losses.value_loss_sum += v_loss.item()
            losses.entropy_sum += entropy_loss.item()
            losses.approx_kl_sum += approx_kl.item()
            losses.clipfrac_sum += clipfrac.item()
            losses.l2_reg_loss_sum += l2_reg_loss.item() if torch.is_tensor(l2_reg_loss) else l2_reg_loss
            losses.l2_init_loss_sum += l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
            losses.ks_action_loss_sum += ks_action_loss.item()
            losses.ks_value_loss_sum += ks_value_loss.item()
            losses.importance_sum += importance_sampling_ratio.mean().item()
            losses.minibatches_processed += 1

            optimizer.zero_grad()
            loss.backward()
            if (minibatch_idx + 1) % experience.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimizer.step()

                if clip_range > 0 and hasattr(policy, "clip_weights"):
                    policy.clip_weights()

                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()

            minibatch_idx += 1
            # end loop over minibatches

        # check early exit if we have reached target_kl
        if target_kl is not None:
            average_approx_kl = losses.approx_kl_sum / losses.minibatches_processed
            if average_approx_kl > target_kl:
                break
        # end loop over epochs

    # Calculate explained variance
    y_pred = experience.values.flatten()
    y_true = advantages.flatten() + experience.values.flatten()
    var_y = y_true.var()
    explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
    losses.explained_variance = explained_var.item() if torch.is_tensor(explained_var) else float("nan")
