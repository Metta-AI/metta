"""Functional training utilities for Metta.

This module provides functional implementations of the core training loop components,
extracting the rollout and train logic from MettaTrainer into standalone functions.
"""

import logging
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import einops
import numpy as np
import torch
import torch.distributed
from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib
from torch import Tensor

from metta.agent.policy_state import PolicyState
from metta.agent.util.debug import assert_shape
from metta.mettagrid.util.dict_utils import unroll_nested_dict
from metta.rl.experience import Experience
from metta.rl.losses import Losses

logger = logging.getLogger(__name__)


def get_observation(
    vecenv: Any,
    device: torch.device,
    timer: Any,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, list, slice, Tensor, int]:
    """Get observations and other data from the vectorized environment and convert to tensors.

    Returns:
        Tuple of (observations, rewards, dones, truncations, info, training_env_id, mask, num_steps)
    """
    # Receive environment data
    with timer("_rollout.env"):
        o, r, d, t, info, env_id, mask = vecenv.recv()

    training_env_id = slice(env_id[0], env_id[-1] + 1)

    mask = torch.as_tensor(mask)
    num_steps = int(mask.sum().item())

    # Convert to tensors
    o = torch.as_tensor(o).to(device, non_blocking=True)
    r = torch.as_tensor(r).to(device, non_blocking=True)
    d = torch.as_tensor(d).to(device, non_blocking=True)
    t = torch.as_tensor(t).to(device, non_blocking=True)

    return o, r, d, t, info, training_env_id, mask, num_steps


def run_policy_inference(
    policy: torch.nn.Module,
    observations: Tensor,
    experience: Experience,
    training_env_id_start: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, Tensor]]]:
    """Run the policy to get actions and value estimates.

    Returns:
        Tuple of (actions, selected_action_log_probs, values, lstm_state_to_store)
    """
    with torch.no_grad():
        state = PolicyState()
        lstm_h, lstm_c = experience.get_lstm_state(training_env_id_start)
        if lstm_h is not None:
            state.lstm_h = lstm_h
            state.lstm_c = lstm_c

        actions, selected_action_log_probs, _, value, _ = policy(observations, state)

        if __debug__:
            assert_shape(selected_action_log_probs, ("BT",), "selected_action_log_probs")
            assert_shape(actions, ("BT", 2), "actions")

        lstm_state_to_store = None
        if state.lstm_h is not None and state.lstm_c is not None:
            lstm_state_to_store = {"lstm_h": state.lstm_h.detach(), "lstm_c": state.lstm_c.detach()}

        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    return actions, selected_action_log_probs, value.flatten(), lstm_state_to_store


def compute_ppo_losses(
    minibatch: Dict[str, Tensor],
    new_logprobs: Tensor,
    entropy: Tensor,
    newvalue: Tensor,
    importance_sampling_ratio: Tensor,
    adv: Tensor,
    trainer_cfg: Any,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute PPO losses for policy and value functions."""
    # Policy loss
    pg_loss1 = -adv * importance_sampling_ratio
    pg_loss2 = -adv * torch.clamp(
        importance_sampling_ratio, 1 - trainer_cfg.ppo.clip_coef, 1 + trainer_cfg.ppo.clip_coef
    )
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue_reshaped = newvalue.view(minibatch["returns"].shape)
    if trainer_cfg.ppo.clip_vloss:
        v_loss_unclipped = (newvalue_reshaped - minibatch["returns"]) ** 2
        vf_clip_coef = trainer_cfg.ppo.vf_clip_coef
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

    # Compute metrics
    with torch.no_grad():
        logratio = new_logprobs - minibatch["logprobs"]
        approx_kl = ((importance_sampling_ratio - 1) - logratio).mean()
        clipfrac = ((importance_sampling_ratio - 1.0).abs() > trainer_cfg.ppo.clip_coef).float().mean()

    return pg_loss, v_loss, entropy_loss, approx_kl, clipfrac


def process_minibatch_update(
    policy: torch.nn.Module,
    experience: Experience,
    minibatch: Dict[str, Tensor],
    advantages: Tensor,
    trainer_cfg: Any,
    kickstarter: Any,
    agent_step: int,
    losses: Losses,
    device: torch.device,
) -> Tensor:
    """Process a single minibatch update and return the total loss."""
    obs = minibatch["obs"]

    lstm_state = PolicyState()
    _, new_logprobs, entropy, newvalue, full_logprobs = policy(obs, lstm_state, action=minibatch["actions"])

    new_logprobs = new_logprobs.reshape(minibatch["logprobs"].shape)
    logratio = new_logprobs - minibatch["logprobs"]
    importance_sampling_ratio = logratio.exp()
    experience.update_ratio(minibatch["indices"], importance_sampling_ratio)

    # Re-compute advantages with new ratios (V-trace)
    adv = compute_advantage(
        minibatch["values"],
        minibatch["rewards"],
        minibatch["dones"],
        importance_sampling_ratio,
        minibatch["advantages"],
        trainer_cfg.ppo.gamma,
        trainer_cfg.ppo.gae_lambda,
        trainer_cfg.vtrace.vtrace_rho_clip,
        trainer_cfg.vtrace.vtrace_c_clip,
        device,
    )

    # Normalize advantages with distributed support, then apply prioritized weights
    adv = normalize_advantage_distributed(adv, trainer_cfg.ppo.norm_adv)
    adv = minibatch["prio_weights"] * adv

    # Compute losses
    pg_loss, v_loss, entropy_loss, approx_kl, clipfrac = compute_ppo_losses(
        minibatch, new_logprobs, entropy, newvalue, importance_sampling_ratio, adv, trainer_cfg, device
    )

    # Kickstarter losses
    ks_action_loss, ks_value_loss = kickstarter.loss(agent_step, full_logprobs, newvalue, obs, teacher_lstm_state=[])

    # L2 init loss
    l2_init_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    if trainer_cfg.ppo.l2_init_loss_coef > 0:
        l2_init_loss = trainer_cfg.ppo.l2_init_loss_coef * policy.l2_init_loss().to(device)

    # Total loss
    loss = (
        pg_loss
        - trainer_cfg.ppo.ent_coef * entropy_loss
        + v_loss * trainer_cfg.ppo.vf_coef
        + l2_init_loss
        + ks_action_loss
        + ks_value_loss
    )

    # Update values in experience buffer
    experience.update_values(minibatch["indices"], newvalue.view(minibatch["values"].shape))

    # Update loss tracking
    losses.policy_loss_sum += pg_loss.item()
    losses.value_loss_sum += v_loss.item()
    losses.entropy_sum += entropy_loss.item()
    losses.approx_kl_sum += approx_kl.item()
    losses.clipfrac_sum += clipfrac.item()
    losses.l2_init_loss_sum += l2_init_loss.item() if torch.is_tensor(l2_init_loss) else l2_init_loss
    losses.ks_action_loss_sum += ks_action_loss.item()
    losses.ks_value_loss_sum += ks_value_loss.item()
    losses.importance_sum += importance_sampling_ratio.mean().item()
    losses.minibatches_processed += 1
    losses.current_logprobs_sum += new_logprobs.mean().item()

    return loss


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
    """CUDA kernel for puffer advantage with automatic CPU fallback.

    This matches the trainer.py implementation exactly.
    """
    # Get correct device
    device = torch.device(device) if isinstance(device, str) else device

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
    """Normalize advantages with distributed training support while preserving shape.

    This matches the trainer.py implementation exactly.
    """
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


def calculate_explained_variance(values: Tensor, advantages: Tensor) -> float:
    """Calculate explained variance for value function evaluation."""
    y_pred = values.flatten()
    y_true = advantages.flatten() + values.flatten()
    var_y = y_true.var()
    explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
    return explained_var.item() if torch.is_tensor(explained_var) else float("nan")


def get_lstm_config(policy: Any) -> Tuple[int, int]:
    """Extract LSTM configuration from policy."""
    hidden_size = getattr(policy, "hidden_size", 256)
    num_lstm_layers = 2  # Default value

    # Try to get actual number of LSTM layers from policy
    if hasattr(policy, "components") and "_core_" in policy.components:
        lstm_module = policy.components["_core_"]
        if hasattr(lstm_module, "_net") and hasattr(lstm_module._net, "num_layers"):
            num_lstm_layers = lstm_module._net.num_layers

    return hidden_size, num_lstm_layers


def calculate_batch_sizes(
    forward_pass_minibatch_target_size: int,
    num_agents: int,
    num_workers: int,
    async_factor: int,
) -> Tuple[int, int, int]:
    """Calculate target batch size, actual batch size, and number of environments.

    Returns:
        Tuple of (target_batch_size, batch_size, num_envs)
    """
    target_batch_size = forward_pass_minibatch_target_size // num_agents
    if target_batch_size < max(2, num_workers):  # pufferlib bug requires batch size >= 2
        target_batch_size = num_workers

    batch_size = (target_batch_size // num_workers) * num_workers
    num_envs = batch_size * async_factor

    return target_batch_size, batch_size, num_envs


def calculate_prioritized_sampling_params(
    epoch: int,
    total_timesteps: int,
    batch_size: int,
    prio_alpha: float,
    prio_beta0: float,
) -> float:
    """Calculate annealed beta for prioritized experience replay."""
    total_epochs = max(1, total_timesteps // batch_size)
    anneal_beta = prio_beta0 + (1 - prio_beta0) * prio_alpha * epoch / total_epochs
    return anneal_beta


def accumulate_rollout_stats(
    raw_infos: list,
    stats: Dict[str, Any],
) -> None:
    """Accumulate rollout statistics from info dictionaries."""
    infos = defaultdict(list)

    # Batch process info dictionaries
    for i in raw_infos:
        for k, v in unroll_nested_dict(i):
            # Detach any tensors before accumulating to prevent memory leaks
            if torch.is_tensor(v):
                v = v.detach().cpu().item() if v.numel() == 1 else v.detach().cpu().numpy()
            elif isinstance(v, np.ndarray) and v.size == 1:
                v = v.item()
            infos[k].append(v)

    # Batch process stats
    for k, v in infos.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()

        if isinstance(v, list):
            stats.setdefault(k, []).extend(v)
        else:
            if k not in stats:
                stats[k] = v
            else:
                try:
                    stats[k] += v
                except TypeError:
                    stats[k] = [stats[k], v]  # fallback: bundle as list


def compute_gradient_stats(policy: torch.nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for the policy.

    Returns:
        Dictionary with 'grad/mean', 'grad/variance', and 'grad/norm' keys
    """
    all_gradients = []
    for param in policy.parameters():
        if param.grad is not None:
            all_gradients.append(param.grad.view(-1))

    if not all_gradients:
        return {}

    all_gradients_tensor = torch.cat(all_gradients).to(torch.float32)

    grad_mean = all_gradients_tensor.mean()
    grad_variance = all_gradients_tensor.var()
    grad_norm = all_gradients_tensor.norm(2)

    grad_stats = {
        "grad/mean": grad_mean.item(),
        "grad/variance": grad_variance.item(),
        "grad/norm": grad_norm.item(),
    }

    return grad_stats


def cleanup_old_policies(checkpoint_dir: str, keep_last_n: int = 5) -> None:
    """Clean up old saved policies to prevent memory accumulation.

    Args:
        checkpoint_dir: Directory containing policy checkpoints
        keep_last_n: Number of most recent policies to keep
    """
    from pathlib import Path

    try:
        # Get checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return

        # List all policy files
        policy_files = sorted(checkpoint_path.glob("policy_*.pt"))

        # Keep only the most recent ones
        if len(policy_files) > keep_last_n:
            files_to_remove = policy_files[:-keep_last_n]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove old policy file {file_path}: {e}")

    except Exception as e:
        logger.warning(f"Error during policy cleanup: {e}")


def setup_distributed_vars() -> Tuple[bool, int, int]:
    """Set up distributed training variables.

    Returns:
        Tuple of (_master, _world_size, _rank)
    """
    if torch.distributed.is_initialized():
        _master = torch.distributed.get_rank() == 0
        _world_size = torch.distributed.get_world_size()
        _rank = torch.distributed.get_rank()
    else:
        _master = True
        _world_size = 1
        _rank = 0

    return _master, _world_size, _rank


def should_run_on_interval(
    epoch: int,
    interval: int,
    is_master: bool = True,
    force: bool = False,
) -> bool:
    """Check if a periodic task should run based on interval and master status.

    Args:
        epoch: Current epoch
        interval: Interval to check
        is_master: Whether this is the master rank
        force: Force run regardless of interval

    Returns:
        True if should run, False otherwise
    """
    if not is_master or not interval:
        return False

    if force:
        return True

    return epoch % interval == 0


def maybe_update_l2_weights(
    agent: Any,
    epoch: int,
    interval: int,
    is_master: bool = True,
    force: bool = False,
) -> None:
    """Update L2 weights if on interval.

    Args:
        agent: Policy/agent with update_l2_init_weight_copy method
        epoch: Current epoch
        interval: Update interval (0 to disable)
        is_master: Whether this is the master process
        force: Force update regardless of interval
    """
    if not is_master or not interval:
        return

    if force or epoch % interval == 0:
        if hasattr(agent, "update_l2_init_weight_copy"):
            agent.update_l2_init_weight_copy()
            logger.info(f"Updated L2 init weight copy at epoch {epoch}")


# ============================================================================
# High-Level Training Functions
# ============================================================================


def rollout(
    vecenv: Any,
    policy: Any,
    experience: Any,
    device: torch.device,
    timer: Any,
) -> Tuple[int, list]:
    """Perform a complete rollout phase.

    Returns:
        Tuple of (total_steps, raw_infos)
    """
    raw_infos = []
    experience.reset_for_rollout()
    total_steps = 0

    while not experience.ready_for_training:
        # Get observation
        o, r, d, t, info, training_env_id, mask, num_steps = get_observation(vecenv, device, timer)
        total_steps += num_steps

        # Run policy inference
        actions, selected_action_log_probs, values, lstm_state_to_store = run_policy_inference(
            policy, o, experience, training_env_id.start, device
        )

        # Store experience
        experience.store(
            obs=o,
            actions=actions,
            logprobs=selected_action_log_probs,
            rewards=r,
            dones=d,
            truncations=t,
            values=values,
            env_id=training_env_id,
            mask=mask,
            lstm_state=lstm_state_to_store,
        )

        # Send actions to environment
        with timer("_rollout.env"):
            from metta.mettagrid.mettagrid_env import dtype_actions

            vecenv.send(actions.cpu().numpy().astype(dtype_actions))

        # Collect info
        if info:
            raw_infos.extend(info)

    return total_steps, raw_infos


def train_epoch(
    policy: Any,
    optimizer: Any,
    experience: Any,
    kickstarter: Any,
    losses: Any,
    trainer_cfg: Any,
    agent_step: int,
    epoch: int,
    device: torch.device,
) -> int:
    """Perform training for one or more epochs on collected experience.

    Returns:
        Number of epochs trained
    """
    losses.zero()
    experience.reset_importance_sampling_ratios()

    # Calculate prioritized sampling parameters
    anneal_beta = calculate_prioritized_sampling_params(
        epoch=epoch,
        total_timesteps=trainer_cfg.total_timesteps,
        batch_size=trainer_cfg.batch_size,
        prio_alpha=trainer_cfg.prioritized_experience_replay.prio_alpha,
        prio_beta0=trainer_cfg.prioritized_experience_replay.prio_beta0,
    )

    # Compute initial advantages
    advantages = torch.zeros(experience.values.shape, device=device)
    initial_importance_sampling_ratio = torch.ones_like(experience.values)

    advantages = compute_advantage(
        experience.values,
        experience.rewards,
        experience.dones,
        initial_importance_sampling_ratio,
        advantages,
        trainer_cfg.ppo.gamma,
        trainer_cfg.ppo.gae_lambda,
        trainer_cfg.vtrace.vtrace_rho_clip,
        trainer_cfg.vtrace.vtrace_c_clip,
        device,
    )

    # Train for multiple epochs
    total_minibatches = experience.num_minibatches * trainer_cfg.update_epochs
    minibatch_idx = 0
    epochs_trained = 0

    for _update_epoch in range(trainer_cfg.update_epochs):
        for _ in range(experience.num_minibatches):
            # Sample minibatch
            minibatch = experience.sample_minibatch(
                advantages=advantages,
                prio_alpha=trainer_cfg.prioritized_experience_replay.prio_alpha,
                prio_beta=anneal_beta,
                minibatch_idx=minibatch_idx,
                total_minibatches=total_minibatches,
            )

            # Process minibatch
            loss = process_minibatch_update(
                policy=policy,
                experience=experience,
                minibatch=minibatch,
                advantages=advantages,
                trainer_cfg=trainer_cfg,
                kickstarter=kickstarter,
                agent_step=agent_step,
                losses=losses,
                device=device,
            )

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()

            if (minibatch_idx + 1) % experience.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), trainer_cfg.ppo.max_grad_norm)
                optimizer.step()

                # Optional weight clipping
                if hasattr(policy, "clip_weights"):
                    policy.clip_weights()

                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()

            minibatch_idx += 1

        epochs_trained += 1

        # Early exit if KL divergence is too high
        if trainer_cfg.ppo.target_kl is not None:
            average_approx_kl = losses.approx_kl_sum / losses.minibatches_processed
            if average_approx_kl > trainer_cfg.ppo.target_kl:
                break

    # Calculate explained variance
    losses.explained_variance = calculate_explained_variance(experience.values, advantages)

    return epochs_trained


def process_training_stats(
    raw_stats: Dict[str, Any],
    losses: Any,
    experience: Any,
    trainer_config: Any,
    kickstarter: Any,
) -> Dict[str, Any]:
    """Process training statistics into a clean format.

    Args:
        raw_stats: Raw statistics dictionary (possibly with lists of values)
        losses: Losses object with stats() method
        experience: Experience object with stats() method
        trainer_config: Training configuration
        kickstarter: Kickstarter object

    Returns:
        Dictionary with processed statistics including:
        - mean_stats: Raw stats converted to means
        - losses_stats: Loss statistics
        - experience_stats: Experience buffer statistics
        - environment_stats: Environment-specific stats
        - overview: High-level metrics like average reward
    """
    # Convert lists to means
    mean_stats = {}
    for k, v in raw_stats.items():
        try:
            mean_stats[k] = np.mean(v)
        except (TypeError, ValueError):
            mean_stats[k] = v

    # Get loss and experience statistics
    losses_stats = losses.stats() if hasattr(losses, "stats") else {}
    experience_stats = experience.stats() if hasattr(experience, "stats") else {}

    # Remove unused losses
    if trainer_config.ppo.l2_reg_loss_coef == 0:
        losses_stats.pop("l2_reg_loss", None)
    if trainer_config.ppo.l2_init_loss_coef == 0:
        losses_stats.pop("l2_init_loss", None)
    if not kickstarter.enabled:
        losses_stats.pop("ks_action_loss", None)
        losses_stats.pop("ks_value_loss", None)

    # Calculate environment statistics
    environment_stats = {
        f"env_{k.split('/')[0]}/{'/'.join(k.split('/')[1:])}": v for k, v in mean_stats.items() if "/" in k
    }

    # Calculate overview statistics
    overview = {}

    # Calculate average reward from environment stats
    task_reward_values = [v for k, v in environment_stats.items() if k.startswith("env_task_reward")]
    if task_reward_values:
        mean_reward = sum(task_reward_values) / len(task_reward_values)
        overview["reward"] = mean_reward

    return {
        "mean_stats": mean_stats,
        "losses_stats": losses_stats,
        "experience_stats": experience_stats,
        "environment_stats": environment_stats,
        "overview": overview,
    }


def compute_timing_stats(
    timer: Any,
    agent_step: int,
    world_size: int = 1,
) -> Dict[str, Any]:
    """Compute timing statistics from a Stopwatch timer.

    Args:
        timer: Stopwatch instance
        agent_step: Current agent step count
        world_size: Number of distributed processes

    Returns:
        Dictionary with timing statistics including:
        - lap_times: Per-operation lap times
        - epoch_steps: Steps in this epoch
        - epoch_steps_per_second: Steps per second (epoch)
        - steps_per_second: Overall steps per second
        - timing_stats: Formatted timing statistics for logging
    """
    elapsed_times = timer.get_all_elapsed()
    wall_time = timer.get_elapsed()
    train_time = elapsed_times.get("_rollout", 0) + elapsed_times.get("_train", 0)

    lap_times = timer.lap_all(agent_step, exclude_global=False)
    wall_time_for_lap = lap_times.pop("global", 0)

    epoch_steps = timer.get_lap_steps()
    if epoch_steps is None:
        epoch_steps = 0

    epoch_steps_per_second = epoch_steps / wall_time_for_lap if wall_time_for_lap > 0 else 0
    steps_per_second = timer.get_rate(agent_step) if wall_time > 0 else 0

    # Scale by world size for distributed training
    epoch_steps_per_second *= world_size
    steps_per_second *= world_size

    timing_stats = {
        **{
            f"timing_per_epoch/frac/{op}": lap_elapsed / wall_time_for_lap if wall_time_for_lap > 0 else 0
            for op, lap_elapsed in lap_times.items()
        },
        **{
            f"timing_per_epoch/msec/{op}": lap_elapsed * 1000 if wall_time_for_lap > 0 else 0
            for op, lap_elapsed in lap_times.items()
        },
        "timing_per_epoch/sps": epoch_steps_per_second,
        **{
            f"timing_cumulative/frac/{op}": elapsed / wall_time if wall_time > 0 else 0
            for op, elapsed in elapsed_times.items()
        },
        "timing_cumulative/sps": steps_per_second,
    }

    return {
        "lap_times": lap_times,
        "elapsed_times": elapsed_times,
        "wall_time": wall_time,
        "train_time": train_time,
        "wall_time_for_lap": wall_time_for_lap,
        "epoch_steps": epoch_steps,
        "epoch_steps_per_second": epoch_steps_per_second,
        "steps_per_second": steps_per_second,
        "timing_stats": timing_stats,
    }


def build_wandb_stats(
    processed_stats: Dict[str, Any],
    timing_info: Dict[str, Any],
    weight_stats: Dict[str, Any],
    grad_stats: Dict[str, Any],
    system_stats: Dict[str, Any],
    memory_stats: Dict[str, Any],
    parameters: Dict[str, Any],
    evals: Dict[str, float],
    agent_step: int,
    epoch: int,
    world_size: int = 1,
) -> Dict[str, Any]:
    """Build complete statistics dictionary for wandb logging.

    Args:
        processed_stats: Output from process_training_stats
        timing_info: Output from compute_timing_stats
        weight_stats: Weight analysis statistics
        grad_stats: Gradient statistics
        system_stats: System monitor statistics
        memory_stats: Memory monitor statistics
        parameters: Training parameters
        evals: Evaluation scores
        agent_step: Current agent step
        epoch: Current epoch
        world_size: Number of distributed processes

    Returns:
        Complete dictionary ready for wandb logging
    """
    # Build overview with sps and rewards
    overview = {
        "sps": timing_info["epoch_steps_per_second"],
        **processed_stats["overview"],
    }

    # Add evaluation scores to overview
    category_scores_map = {key.split("/")[0]: value for key, value in evals.items() if key.endswith("/score")}
    for category, score in category_scores_map.items():
        overview[f"{category}_score"] = score

    # Also add reward_vs_total_time if we have reward
    if "reward" in overview:
        overview["reward_vs_total_time"] = overview["reward"]

    # X-axis values for wandb
    metric_stats = {
        "metric/agent_step": agent_step * world_size,
        "metric/epoch": epoch,
        "metric/total_time": timing_info["wall_time"],
        "metric/train_time": timing_info["train_time"],
    }

    # Combine all stats
    return {
        **{f"overview/{k}": v for k, v in overview.items()},
        **{f"losses/{k}": v for k, v in processed_stats["losses_stats"].items()},
        **{f"experience/{k}": v for k, v in processed_stats["experience_stats"].items()},
        **{f"parameters/{k}": v for k, v in parameters.items()},
        **{f"eval_{k}": v for k, v in evals.items()},
        **system_stats,  # Already has monitor/ prefix from SystemMonitor.stats()
        **{f"trainer_memory/{k}": v for k, v in memory_stats.items()},
        **processed_stats["environment_stats"],
        **weight_stats,
        **timing_info["timing_stats"],
        **metric_stats,
        **grad_stats,
    }


def process_stats(
    stats: Dict[str, Any],
    losses: Any,
    evals: Dict[str, float],
    grad_stats: Dict[str, float],
    experience: Any,
    policy: Any,
    timer: Any,
    trainer_cfg: Any,
    agent_step: int,
    epoch: int,
    world_size: int,
    wandb_run: Optional[Any],
    memory_monitor: Optional[Any],
    system_monitor: Optional[Any],
    latest_saved_policy_record: Optional[Any],
    initial_policy_record: Optional[Any],
    optimizer: Optional[Any] = None,
) -> None:
    """Process and log training statistics - kept for backward compatibility with process_stats API."""
    if not wandb_run:
        return

    # Process training stats
    processed_stats = process_training_stats(
        raw_stats=stats,
        losses=losses,
        experience=experience,
        trainer_config=trainer_cfg,
        kickstarter=None,  # Not available in this API
    )

    # Compute timing stats
    timing_info = compute_timing_stats(
        timer=timer,
        agent_step=agent_step,
        world_size=world_size,
    )

    # Compute weight stats if configured
    weight_stats = {}
    if policy and hasattr(trainer_cfg, "agent") and hasattr(trainer_cfg.agent, "analyze_weights_interval"):
        if trainer_cfg.agent.analyze_weights_interval != 0 and epoch % trainer_cfg.agent.analyze_weights_interval == 0:
            for metrics in policy.compute_weight_metrics():
                name = metrics.get("name", "unknown")
                for key, value in metrics.items():
                    if key != "name":
                        weight_stats[f"weights/{key}/{name}"] = value

    # Build parameters
    parameters = {
        "learning_rate": optimizer.param_groups[0]["lr"] if optimizer else trainer_cfg.optimizer.learning_rate,
        "epoch_steps": timing_info["epoch_steps"],
        "num_minibatches": experience.num_minibatches,
        "generation": initial_policy_record.metadata.get("generation", 0) + 1 if initial_policy_record else 0,
        "latest_saved_policy_epoch": latest_saved_policy_record.metadata.epoch if latest_saved_policy_record else 0,
    }

    # Get system stats
    system_stats = system_monitor.stats() if system_monitor else {}
    memory_stats = memory_monitor.stats() if memory_monitor else {}

    # Build complete stats
    all_stats = build_wandb_stats(
        processed_stats=processed_stats,
        timing_info=timing_info,
        weight_stats=weight_stats,
        grad_stats=grad_stats,
        system_stats=system_stats,
        memory_stats=memory_stats,
        parameters=parameters,
        evals=evals,
        agent_step=agent_step,
        epoch=epoch,
        world_size=world_size,
    )

    # Log to wandb
    wandb_run.log(all_stats, step=agent_step)


def evaluate_policy(
    policy_record: Any,
    policy_store: Any,
    sim_suite_config: Any,
    stats_client: Optional[Any],
    stats_run_id: Optional[Any],
    stats_epoch_start: int,
    epoch: int,
    device: torch.device,
    vectorization: str,
    wandb_policy_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Optional[Any]]:
    """Evaluate policy and return scores.

    Returns:
        Tuple of (eval_scores, stats_epoch_id)
    """
    from metta.common.util.heartbeat import record_heartbeat
    from metta.eval.eval_stats_db import EvalStatsDB
    from metta.sim.simulation_suite import SimulationSuite

    stats_epoch_id = None
    if stats_run_id is not None and stats_client is not None:
        stats_epoch_id = stats_client.create_epoch(
            run_id=stats_run_id,
            start_training_epoch=stats_epoch_start,
            end_training_epoch=epoch,
            attributes={},
        ).id

    logger.info(f"Simulating policy: {policy_record.uri} with config: {sim_suite_config}")

    sim_suite = SimulationSuite(
        config=sim_suite_config,
        policy_pr=policy_record,
        policy_store=policy_store,
        device=device,
        vectorization=vectorization,
        stats_dir="/tmp/stats",
        stats_client=stats_client,
        stats_epoch_id=stats_epoch_id,
        wandb_policy_name=wandb_policy_name,
    )

    result = sim_suite.simulate()
    stats_db = EvalStatsDB.from_sim_stats_db(result.stats_db)
    logger.info("Simulation complete")

    # Build evaluation metrics
    eval_scores = {}
    categories = set()
    for sim_name in sim_suite_config.simulations.keys():
        categories.add(sim_name.split("/")[0])

    for category in categories:
        score = stats_db.get_average_metric_by_filter("reward", policy_record, f"sim_name LIKE '%{category}%'")
        logger.info(f"{category} score: {score}")
        record_heartbeat()
        if score is not None:
            eval_scores[f"{category}/score"] = score

    # Get detailed per-simulation scores
    all_scores = stats_db.simulation_scores(policy_record, "reward")
    for (_, sim_name, _), score in all_scores.items():
        category = sim_name.split("/")[0]
        sim_short_name = sim_name.split("/")[-1]
        eval_scores[f"{category}/{sim_short_name}"] = score

    stats_db.close()
    return eval_scores, stats_epoch_id


def generate_replay(
    policy_record: Any,
    policy_store: Any,
    curriculum: Any,
    epoch: int,
    device: torch.device,
    vectorization: str,
    replay_dir: str,
    wandb_run: Optional[Any] = None,
) -> None:
    """Generate and upload replay."""
    from metta.sim.simulation import Simulation
    from metta.sim.simulation_config import SingleEnvSimulationConfig

    replay_sim_config = SingleEnvSimulationConfig(
        env="/env/mettagrid/mettagrid",
        num_episodes=1,
        env_overrides=curriculum.get_task().env_cfg(),
    )

    replay_simulator = Simulation(
        name=f"replay_{epoch}",
        config=replay_sim_config,
        policy_pr=policy_record,
        policy_store=policy_store,
        device=device,
        vectorization=vectorization,
        replay_dir=replay_dir,
    )

    results = replay_simulator.simulate()

    if wandb_run is not None:
        key, version = results.stats_db.key_and_version(policy_record)
        replay_urls = results.stats_db.get_replay_urls(key, version)
        if len(replay_urls) > 0:
            replay_url = replay_urls[0]
            player_url = f"https://metta-ai.github.io/metta/?replayUrl={replay_url}"
            import wandb

            link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
            wandb_run.log(link_summary)

    results.stats_db.close()


def save_policy_with_metadata(
    policy: Any,
    policy_store: Any,
    epoch: int,
    agent_step: int,
    evals: Dict[str, float],
    timer: Any,
    vecenv: Any,
    initial_policy_record: Optional[Any],
    run_name: str,
    is_master: bool = True,
) -> Optional[Any]:
    """Save policy with metadata.

    Returns:
        Saved policy record or None if not master
    """
    if not is_master:
        return None

    from metta.agent.metta_agent import DistributedMettaAgent
    from metta.agent.policy_metadata import PolicyMetadata
    from metta.mettagrid.mettagrid_env import MettaGridEnv

    name = policy_store.make_model_name(epoch)

    metta_grid_env: MettaGridEnv = vecenv.driver_env
    assert isinstance(metta_grid_env, MettaGridEnv), "vecenv.driver_env must be a MettaGridEnv"

    training_time = timer.get_elapsed("_rollout") + timer.get_elapsed("_train")

    category_scores_map = {key.split("/")[0]: value for key, value in evals.items() if key.endswith("/score")}
    category_score_values = [v for k, v in category_scores_map.items()]
    overall_score = sum(category_score_values) / len(category_score_values) if category_score_values else 0

    metadata = PolicyMetadata(
        agent_step=agent_step,
        epoch=epoch,
        run=run_name,
        action_names=metta_grid_env.action_names,
        generation=initial_policy_record.metadata.get("generation", 0) + 1 if initial_policy_record else 0,
        initial_uri=initial_policy_record.uri if initial_policy_record else None,
        train_time=training_time,
        score=overall_score,
        eval_scores=category_scores_map,
    )

    # Extract actual policy from distributed wrapper
    policy_to_save = policy
    if isinstance(policy, DistributedMettaAgent):
        policy_to_save = policy.module

    # Save original feature mapping
    if hasattr(policy_to_save, "get_original_feature_mapping"):
        original_feature_mapping = policy_to_save.get_original_feature_mapping()
        if original_feature_mapping is not None:
            metadata["original_feature_mapping"] = original_feature_mapping
            logger.info(f"Saving original_feature_mapping with {len(original_feature_mapping)} features to metadata")

    # Create and save policy record
    policy_record = policy_store.create_empty_policy_record(name)
    policy_record.metadata = metadata
    policy_record.policy = policy_to_save

    saved_record = policy_store.save(policy_record)
    logger.info(f"Successfully saved policy at epoch {epoch}")

    return saved_record


def save_training_state(
    checkpoint_dir: str,
    agent_step: int,
    epoch: int,
    optimizer: Any,
    timer: Any,
    latest_saved_policy_uri: Optional[str],
    kickstarter: Any,
    world_size: int,
    is_master: bool = True,
) -> None:
    """Save training checkpoint state.

    Only master saves, but all ranks should call this for distributed sync.
    """
    if not is_master:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    from metta.rl.trainer_checkpoint import TrainerCheckpoint

    extra_args = {}
    if kickstarter.enabled and kickstarter.teacher_uri is not None:
        extra_args["teacher_pr_uri"] = kickstarter.teacher_uri

    checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        total_agent_step=agent_step * world_size,
        optimizer_state_dict=optimizer.state_dict(),
        stopwatch_state=timer.save_state(),
        policy_path=latest_saved_policy_uri,
        extra_args=extra_args,
    )
    checkpoint.save(checkpoint_dir)
    logger.info(f"Saved training state at epoch {epoch}")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
