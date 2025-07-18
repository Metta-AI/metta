"""Training phase functions for Metta."""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import wandb
from torch import Tensor

from metta.agent.policy_state import PolicyState
from metta.rl.experience import Experience
from metta.rl.functions.advantage import compute_advantage, normalize_advantage_distributed
from metta.rl.losses import Losses

logger = logging.getLogger(__name__)


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


def calculate_explained_variance(values: Tensor, advantages: Tensor) -> float:
    """Calculate explained variance for value function evaluation."""
    y_pred = values.flatten()
    y_true = advantages.flatten() + values.flatten()
    var_y = y_true.var()
    explained_var = torch.nan if var_y == 0 else 1 - (y_true - y_pred).var() / var_y
    return explained_var.item() if torch.is_tensor(explained_var) else float("nan")


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


def setup_device_and_distributed(base_device: str = "cuda") -> Tuple[torch.device, bool, int, int]:
    """Set up device and initialize distributed training if needed.

    This function handles:
    - Device selection based on LOCAL_RANK environment variable
    - Distributed process group initialization with appropriate backend
    - Fallback to CPU if CUDA requested but not available
    - Returns distributed training variables (is_master, world_size, rank)

    Args:
        base_device: Base device type ("cuda" or "cpu")

    Returns:
        Tuple of (device, is_master, world_size, rank)
    """
    # Check CUDA availability
    if base_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        base_device = "cpu"

    # Handle distributed setup
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])

        if base_device.startswith("cuda"):
            # CUDA distributed training
            device = torch.device(f"{base_device}:{local_rank}")
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
                logger.info(f"Initialized NCCL distributed training on {device}")
        else:
            # CPU distributed training
            device = torch.device(base_device)
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="gloo")
                logger.info(f"Initialized Gloo distributed training on {device}")
    else:
        # Single device training
        device = torch.device(base_device)
        logger.info(f"Single device training on {device}")

    # Get distributed vars using the shared function
    is_master, world_size, rank = setup_distributed_vars()

    return device, is_master, world_size, rank


def should_run(
    epoch: int,
    interval: int,
    is_master: bool = True,
    force: bool = False,
) -> bool:
    """Check if a periodic task should run based on interval and master status."""
    if not is_master or not interval:
        return False

    if force:
        return True

    return epoch % interval == 0


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
            link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
            wandb_run.log(link_summary)

    results.stats_db.close()
