import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed
import wandb
from heavyball import ForeachMuon
from omegaconf import DictConfig, OmegaConf

from metta.agent.metta_agent import DistributedMettaAgent
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_service import evaluate_policy as eval_service_evaluate_policy
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.mettagrid.mettagrid_env import MettaGridEnv, dtype_actions
from metta.rl.experience import Experience
from metta.rl.hyperparameter_scheduler import HyperparameterScheduler
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import create_trainer_config
from metta.rl.util.advantage import compute_advantage
from metta.rl.util.batch_utils import (
    calculate_batch_sizes,
    calculate_prioritized_sampling_params,
)
from metta.rl.util.distributed import setup_distributed_vars
from metta.rl.util.evaluation import generate_replay
from metta.rl.util.losses import process_minibatch_update
from metta.rl.util.optimization import (
    calculate_explained_variance,
    compute_gradient_stats,
    maybe_update_l2_weights,
)
from metta.rl.util.policy_management import (
    cleanup_old_policies,
    maybe_load_checkpoint,
    save_policy_with_metadata,
    validate_policy_environment_match,
)
from metta.rl.util.rollout import (
    get_lstm_config,
    get_observation,
    run_policy_inference,
)
from metta.rl.util.stats import (
    accumulate_rollout_stats,
    process_stats,
)
from metta.rl.util.utils import should_run
from metta.rl.vecenv import make_vecenv
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig

try:
    from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, "
        "try installing with --no-build-isolation"
    ) from None

torch.set_float32_matmul_precision("high")

# Get rank for logger name
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
logger = logging.getLogger(f"trainer-{rank}-{local_rank}")


@dataclass
class TrainerState:
    """Mutable state for training that gets passed between functions."""

    agent_step: int = 0
    epoch: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    grad_stats: Dict[str, float] = field(default_factory=dict)
    evals: Any = field(default_factory=dict)  # Will be EvalRewardSummary
    latest_saved_policy_record: Optional[Any] = None
    initial_policy_record: Optional[Any] = None
    # Stats tracking
    stats_epoch_start: int = 0
    stats_epoch_id: Optional[Any] = None
    stats_run_id: Optional[Any] = None


def _maybe_save_training_state(
    checkpoint_dir: str,
    agent_step: int,
    epoch: int,
    optimizer: Any,
    timer: Any,
    latest_saved_policy_uri: Optional[str],
    kickstarter: Any,
    trainer_cfg: Any,
    is_master: bool = True,
    force: bool = False,
) -> None:
    """Save training checkpoint state.

    Only master saves, but all ranks should call this for distributed sync.
    """
    # Check interval for all ranks to ensure synchronization
    if not force and trainer_cfg.checkpoint.checkpoint_interval:
        if epoch % trainer_cfg.checkpoint.checkpoint_interval != 0:
            return

    # Only master saves training state, but all ranks must participate in barrier
    if not is_master:
        # Non-master ranks need to participate in the barrier below
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    extra_args = {}
    if kickstarter.enabled and kickstarter.teacher_uri is not None:
        extra_args["teacher_pr_uri"] = kickstarter.teacher_uri

    checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        optimizer_state_dict=optimizer.state_dict(),
        stopwatch_state=timer.save_state(),
        policy_path=latest_saved_policy_uri,
        extra_args=extra_args,
    )
    checkpoint.save(checkpoint_dir)
    logger.info(f"Saved training state at epoch {epoch}")

    # Synchronize all ranks to ensure the checkpoint is fully saved before continuing
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _rollout(
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

        # Send actions back to environment
        with timer("_rollout.env"):
            vecenv.send(actions.cpu().numpy().astype(dtype_actions))

        # Collect info for batch processing
        if info:
            raw_infos.extend(info)

    return total_steps, raw_infos


def _train(
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


def _maybe_save_policy(
    policy: Any,
    policy_store: Any,
    state: TrainerState,
    timer: Any,
    vecenv: Any,
    run_name: str,
    is_master: bool,
    trainer_cfg: Any,
    force: bool = False,
) -> Optional[Any]:
    """Save policy with distributed synchronization."""
    # Check if should save
    should_save = force or (
        trainer_cfg.checkpoint.checkpoint_interval and state.epoch % trainer_cfg.checkpoint.checkpoint_interval == 0
    )
    if not should_save:
        return None

    # All ranks participate in barrier for distributed sync
    if not is_master:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return None

    # Save policy with metadata
    saved_record = save_policy_with_metadata(
        policy=policy,
        policy_store=policy_store,
        epoch=state.epoch,
        agent_step=state.agent_step,
        evals=state.evals,
        timer=timer,
        initial_policy_record=state.initial_policy_record,
        run_name=run_name,
        is_master=is_master,
    )

    if saved_record:
        # Clean up old policies periodically
        if state.epoch % 10 == 0:
            cleanup_old_policies(trainer_cfg.checkpoint.checkpoint_dir, keep_last_n=5)

    # Sync all ranks after save
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    return saved_record


def _upload_policy_to_wandb(
    wandb_run: Any, policy_store: Any, policy_record: Any, force: bool = False
) -> Optional[str]:
    """Upload policy to wandb."""
    if not wandb_run or not policy_record:
        return None

    if not wandb_run.name:
        logger.warning("No wandb run name was provided")
        return None

    result = policy_store.add_to_wandb_run(wandb_run.name, policy_record)
    logger.info(f"Uploaded policy to wandb at epoch {policy_record.metadata.get('epoch', 'unknown')}")
    return result


def _maybe_evaluate_policy(
    policy_record: Any,
    sim_suite_config: SimulationSuiteConfig,
    curriculum: Any,
    stats_client: Optional[Any],
    state: TrainerState,
    device: torch.device,
    vectorization: str,
    replay_dir: str,
    wandb_policy_name: Optional[str],
    policy_store: Any,
    cfg: Any,
    wandb_run: Optional[Any],
    logger: Any,
) -> EvalRewardSummary:
    """Evaluate policy using the new eval service."""
    # Create an extended simulation suite that includes the training task
    extended_suite_config = SimulationSuiteConfig(
        name=sim_suite_config.name,
        simulations=dict(sim_suite_config.simulations),
        env_overrides=sim_suite_config.env_overrides,
        num_episodes=sim_suite_config.num_episodes,
    )

    # Add training task to the suite
    training_task_config = SingleEnvSimulationConfig(
        env="/env/mettagrid/mettagrid",
        num_episodes=1,
        env_overrides=curriculum.get_task().env_cfg(),
    )
    extended_suite_config.simulations["eval/training_task"] = training_task_config

    logger.info("Simulating policy with extended config including training task")

    # Use the eval service evaluate_policy function
    evaluation_results = eval_service_evaluate_policy(
        policy_record=policy_record,
        simulation_suite=extended_suite_config,
        device=device,
        vectorization=vectorization,
        replay_dir=replay_dir,
        stats_epoch_id=state.stats_epoch_id,
        wandb_policy_name=wandb_policy_name,
        policy_store=policy_store,
        stats_client=stats_client,
        logger=logger,
    )

    logger.info("Simulation complete")

    # Set policy metadata score for sweep_eval.py
    target_metric = getattr(cfg, "sweep", {}).get("metric", "reward")  # fallback to reward
    category_scores = list(evaluation_results.scores.category_scores.values())
    if category_scores and policy_record:
        policy_record.metadata["score"] = float(np.mean(category_scores))
        logger.info(f"Set policy metadata score to {policy_record.metadata['score']} using {target_metric} metric")

    # Upload replay HTML if we have wandb
    if wandb_run is not None and evaluation_results.replay_urls:
        _upload_replay_html(
            replay_urls=evaluation_results.replay_urls,
            epoch=state.epoch,
            agent_step=state.agent_step,
            wandb_run=wandb_run,
        )

    return evaluation_results.scores


def _upload_replay_html(
    replay_urls: Dict[str, list[str]],
    epoch: int,
    agent_step: int,
    wandb_run: Any,
) -> None:
    """Upload replay HTML to wandb with unified view of all replay links."""

    # Create unified HTML with all replay links on a single line
    if replay_urls:
        # Group replays by base name
        replay_groups = {}

        for sim_name, urls in sorted(replay_urls.items()):
            if "training_task" in sim_name:
                # Training replays
                if "training" not in replay_groups:
                    replay_groups["training"] = []
                replay_groups["training"].extend(urls)
            else:
                # Evaluation replays - clean up the display name
                display_name = sim_name.replace("eval/", "")
                if display_name not in replay_groups:
                    replay_groups[display_name] = []
                replay_groups[display_name].extend(urls)

        # Build HTML with episode numbers
        links = []
        for name, urls in replay_groups.items():
            if len(urls) == 1:
                # Single episode - just show the name
                player_url = "https://metta-ai.github.io/metta/?replayUrl=" + urls[0]
                links.append(f'<a href="{player_url}" target="_blank">{name}</a>')
            else:
                # Multiple episodes - show with numbers
                episode_links = []
                for i, url in enumerate(urls, 1):
                    player_url = "https://metta-ai.github.io/metta/?replayUrl=" + url
                    episode_links.append(f'<a href="{player_url}" target="_blank">{i}</a>')
                links.append(f"{name} [{' '.join(episode_links)}]")

        # Join all links with " | " separator and add epoch prefix
        html_content = f"epoch {epoch}: " + " | ".join(links)
    else:
        html_content = f"epoch {epoch}: No replays available."

    # Log the unified HTML with step parameter for wandb's epoch slider
    link_summary = {"replays/all_links": wandb.Html(html_content)}
    wandb_run.log(link_summary, step=agent_step)

    # Also log individual link for backward compatibility
    if "eval/training_task" in replay_urls and replay_urls["eval/training_task"]:
        training_url = replay_urls["eval/training_task"][0]  # Use first URL for backward compatibility
        player_url = "https://metta-ai.github.io/metta/?replayUrl=" + training_url
        link_summary = {"replays/link": wandb.Html(f'<a href="{player_url}">MetaScope Replay (Epoch {epoch})</a>')}
        wandb_run.log(link_summary, step=agent_step)


def _check_abort(wandb_run: Optional[Any], trainer_cfg: Any, agent_step: int) -> bool:
    """Check for abort tag in wandb run."""
    if wandb_run is None:
        return False

    try:
        if "abort" not in wandb.Api().run(wandb_run.path).tags:
            return False

        logger.info("Abort tag detected. Stopping the run.")
        trainer_cfg.total_timesteps = int(agent_step)
        wandb_run.config.update({"trainer.total_timesteps": trainer_cfg.total_timesteps}, allow_val_change=True)
        return True
    except Exception:
        return False


def _initialize_stats_tracking(
    state: TrainerState,
    stats_client: Optional[Any],
    wandb_run: Optional[Any],
) -> None:
    """Initialize stats tracking for training run."""
    if stats_client is None:
        return

    if wandb_run is not None:
        name = wandb_run.name if wandb_run.name is not None else "unknown"
        url = wandb_run.url
        tags = list(wandb_run.tags) if wandb_run.tags is not None else None
        description = wandb_run.notes
    else:
        name = "unknown"
        url = None
        tags = None
        description = None

    try:
        state.stats_run_id = stats_client.create_training_run(
            name=name, attributes={}, url=url, description=description, tags=tags
        ).id
    except Exception as e:
        logger.warning(f"Failed to create training run: {e}")


def train(
    cfg: DictConfig,
    wandb_run: Any | None,
    policy_store: Any,
    sim_suite_config: Any,
    stats_client: Any | None,
    **kwargs: Any,
) -> None:
    """Functional training loop replacing MettaTrainer.train()."""
    logger.info("Starting training")

    # Create all components individually
    (
        vecenv,
        policy,
        optimizer,
        experience,
        kickstarter,
        lr_scheduler,
        hyperparameter_scheduler,
        losses,
        timer,
        torch_profiler,
        memory_monitor,
        system_monitor,
        trainer_cfg,
        device,
        is_master,
        world_size,
        rank,
        state,
        curriculum,
    ) = create_training_components(
        cfg=cfg,
        wandb_run=wandb_run,
        policy_store=policy_store,
        sim_suite_config=sim_suite_config,
        stats_client=stats_client,
    )

    # Initialize stats tracking
    _initialize_stats_tracking(state, stats_client, wandb_run)

    logger.info(f"Training on {device}")
    wandb_policy_name: str | None = None

    # Main training loop
    while state.agent_step < trainer_cfg.total_timesteps:
        steps_before = state.agent_step

        with torch_profiler:
            # Rollout phase
            with timer("_rollout"):
                num_steps, raw_infos = _rollout(
                    vecenv=vecenv,
                    policy=policy,
                    experience=experience,
                    device=device,
                    timer=timer,
                )
                state.agent_step += num_steps

                # Process rollout stats
                accumulate_rollout_stats(raw_infos, state.stats)

            # Training phase
            with timer("_train"):
                epochs_trained = _train(
                    policy=policy,
                    optimizer=optimizer,
                    experience=experience,
                    kickstarter=kickstarter,
                    losses=losses,
                    trainer_cfg=trainer_cfg,
                    agent_step=state.agent_step,
                    epoch=state.epoch,
                    device=device,
                )
                state.epoch += epochs_trained

                # Update learning rate scheduler
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Update hyperparameter scheduler
                hyperparameter_scheduler.step(state.agent_step)

        torch_profiler.on_epoch_end(state.epoch)

        # Process stats
        with timer("_process_stats"):
            if is_master and wandb_run:
                process_stats(
                    stats=state.stats,
                    losses=losses,
                    evals=state.evals,
                    grad_stats=state.grad_stats,
                    experience=experience,
                    policy=policy,
                    timer=timer,
                    trainer_cfg=trainer_cfg,
                    agent_step=state.agent_step,
                    epoch=state.epoch,
                    world_size=world_size,
                    wandb_run=wandb_run,
                    memory_monitor=memory_monitor,
                    system_monitor=system_monitor,
                    latest_saved_policy_record=state.latest_saved_policy_record,
                    initial_policy_record=state.initial_policy_record,
                    optimizer=optimizer,
                    kickstarter=kickstarter,
                )
            # Clear stats after processing
            state.stats.clear()
            state.grad_stats.clear()

        # Calculate performance metrics
        rollout_time = timer.get_last_elapsed("_rollout")
        train_time = timer.get_last_elapsed("_train")
        stats_time = timer.get_last_elapsed("_process_stats")
        steps_calculated = state.agent_step - steps_before

        total_time = train_time + rollout_time + stats_time
        steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

        train_pct = (train_time / total_time) * 100
        rollout_pct = (rollout_time / total_time) * 100
        stats_pct = (stats_time / total_time) * 100

        # Format total timesteps with scientific notation for large numbers
        total_timesteps = trainer_cfg.total_timesteps
        if total_timesteps >= 1e9:
            total_steps_str = f"{total_timesteps:.0e}"
        else:
            total_steps_str = f"{total_timesteps:,}"

        logger.info(
            f"Epoch {state.epoch}, "
            f"{steps_per_sec * world_size:.0f} steps/sec, "
            f"Agent step {state.agent_step}/{total_steps_str} "
            f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats)"
        )

        # Periodic tasks
        if should_run(state.epoch, 10, is_master):
            record_heartbeat()

        # Update L2 weights if configured
        if hasattr(policy, "l2_init_weight_update_interval"):
            maybe_update_l2_weights(
                agent=policy,
                epoch=state.epoch,
                interval=getattr(policy, "l2_init_weight_update_interval", 0),
                is_master=is_master,
            )

        # Save policy
        if should_run(state.epoch, trainer_cfg.checkpoint.checkpoint_interval):
            saved_record = _maybe_save_policy(
                policy, policy_store, state, timer, vecenv, cfg.run, is_master, trainer_cfg
            )
            if saved_record:
                state.latest_saved_policy_record = saved_record

        # Save training state
        if should_run(state.epoch, trainer_cfg.checkpoint.checkpoint_interval):
            _maybe_save_training_state(
                checkpoint_dir=cfg.run_dir,
                agent_step=state.agent_step,
                epoch=state.epoch,
                optimizer=optimizer,
                timer=timer,
                latest_saved_policy_uri=state.latest_saved_policy_record.uri
                if state.latest_saved_policy_record
                else None,
                kickstarter=kickstarter,
                trainer_cfg=trainer_cfg,
                is_master=is_master,
            )

        # Upload to wandb
        if should_run(state.epoch, trainer_cfg.checkpoint.wandb_checkpoint_interval, is_master):
            wandb_policy_name = _upload_policy_to_wandb(wandb_run, policy_store, state.latest_saved_policy_record)

        # Evaluate policy
        if should_run(state.epoch, trainer_cfg.simulation.evaluate_interval, is_master):
            if state.latest_saved_policy_record:
                # Create stats epoch if needed
                if stats_client is not None and state.stats_run_id is not None:
                    state.stats_epoch_id = stats_client.create_epoch(
                        run_id=state.stats_run_id,
                        start_training_epoch=state.stats_epoch_start,
                        end_training_epoch=state.epoch,
                        attributes={},
                    ).id

                eval_scores = _maybe_evaluate_policy(
                    state.latest_saved_policy_record,
                    sim_suite_config,
                    curriculum,
                    stats_client,
                    state,
                    device,
                    cfg.vectorization,
                    trainer_cfg.simulation.replay_dir,
                    wandb_policy_name,
                    policy_store,
                    cfg,
                    wandb_run,
                    logger,
                )
                state.evals = eval_scores
                state.stats_epoch_start = state.epoch + 1

        # Generate replay
        if should_run(state.epoch, trainer_cfg.simulation.evaluate_interval, is_master):
            if state.latest_saved_policy_record:
                # Get curriculum from trainer config
                curriculum = curriculum_from_config_path(
                    trainer_cfg.curriculum_or_env, DictConfig(trainer_cfg.env_overrides)
                )

                generate_replay(
                    policy_record=state.latest_saved_policy_record,
                    policy_store=policy_store,
                    curriculum=curriculum,
                    epoch=state.epoch,
                    device=device,
                    vectorization=cfg.vectorization,
                    replay_dir=trainer_cfg.simulation.replay_dir,
                    wandb_run=wandb_run,
                )

        # Compute gradient stats
        if should_run(state.epoch, trainer_cfg.grad_mean_variance_interval, is_master):
            with timer("grad_stats"):
                state.grad_stats = compute_gradient_stats(policy)

        # Check for abort
        if _check_abort(wandb_run, trainer_cfg, state.agent_step):
            break

    logger.info("Training complete!")
    timing_summary = timer.get_all_summaries()

    for name, summary in timing_summary.items():
        logger.info(f"  {name}: {timer.format_time(summary['total_elapsed'])}")

    # Force final saves
    if is_master:
        saved_record = _maybe_save_policy(
            policy, policy_store, state, timer, vecenv, cfg.run, is_master, trainer_cfg, force=True
        )
        if saved_record:
            state.latest_saved_policy_record = saved_record

    _maybe_save_training_state(
        checkpoint_dir=cfg.run_dir,
        agent_step=state.agent_step,
        epoch=state.epoch,
        optimizer=optimizer,
        timer=timer,
        latest_saved_policy_uri=state.latest_saved_policy_record.uri if state.latest_saved_policy_record else None,
        kickstarter=kickstarter,
        trainer_cfg=trainer_cfg,
        is_master=is_master,
        force=True,
    )

    if wandb_run and state.latest_saved_policy_record:
        _upload_policy_to_wandb(wandb_run, policy_store, state.latest_saved_policy_record, force=True)

    # Cleanup
    vecenv.close()
    if is_master:
        if memory_monitor:
            memory_monitor.clear()
        if system_monitor:
            system_monitor.stop()


def create_training_components(
    cfg: Any,
    wandb_run: Optional[Any],
    policy_store: Any,
    sim_suite_config: Any,
    stats_client: Optional[Any] = None,
) -> Tuple[Any, ...]:
    """Create training components individually, similar to run.py."""
    logger.info(f"run_dir = {cfg.run_dir}")

    # Log recent checkpoints like the MettaTrainer did
    checkpoints_dir = Path(cfg.run_dir) / "checkpoints"
    if checkpoints_dir.exists():
        files = sorted(os.listdir(checkpoints_dir))
        recent_files = files[-3:] if len(files) >= 3 else files
        logger.info(f"Recent checkpoints: {', '.join(recent_files)}")

    # Apply batch size scaling BEFORE creating trainer config
    # This matches the behavior in tools/train.py
    if torch.distributed.is_initialized() and cfg.trainer.get("scale_batches_by_world_size", False):
        world_size = torch.distributed.get_world_size()
        # Make a mutable copy of the config to modify
        OmegaConf.set_struct(cfg, False)
        cfg.trainer.forward_pass_minibatch_target_size = cfg.trainer.forward_pass_minibatch_target_size // world_size
        cfg.trainer.batch_size = cfg.trainer.batch_size // world_size
        OmegaConf.set_struct(cfg, True)

    trainer_cfg = create_trainer_config(cfg)

    # Set up distributed
    is_master, world_size, rank = setup_distributed_vars()
    device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device

    # Create utilities
    timer = Stopwatch(logger)
    timer.start()

    memory_monitor = None
    if is_master:
        memory_monitor = MemoryMonitor()

    # Instantiate system monitor (master only)
    system_monitor = None
    if is_master:
        system_monitor = SystemMonitor(
            sampling_interval_sec=1.0,
            history_size=100,
            logger=logger,
            auto_start=True,
            external_timer=timer,
        )

    # Instantiate losses tracker and torch profiler
    losses = Losses()
    torch_profiler = TorchProfiler(is_master, trainer_cfg.profiler, wandb_run, cfg.run_dir)

    # Create curriculum and vecenv
    curriculum = curriculum_from_config_path(trainer_cfg.curriculum_or_env, DictConfig(trainer_cfg.env_overrides))

    # Calculate batch sizes
    num_agents = curriculum.get_task().env_cfg().game.num_agents
    target_batch_size, batch_size, num_envs = calculate_batch_sizes(
        trainer_cfg.forward_pass_minibatch_target_size,
        num_agents,
        trainer_cfg.num_workers,
        trainer_cfg.async_factor,
    )

    vecenv = make_vecenv(
        curriculum,
        cfg.vectorization,
        num_envs=num_envs,
        batch_size=batch_size,
        num_workers=trainer_cfg.num_workers,
        zero_copy=trainer_cfg.zero_copy,
        is_training=True,
    )

    seed = cfg.get("seed", np.random.randint(0, 1000000))
    vecenv.async_reset(seed + rank)

    metta_grid_env: MettaGridEnv = vecenv.driver_env  # type: ignore[attr-defined]

    # Initialize state
    state = TrainerState()
    state.evals = EvalRewardSummary()  # Initialize with empty scores
    state.stats = defaultdict(list)  # Initialize stats dict
    state.grad_stats = {}  # Initialize grad stats

    # Load checkpoint and policy
    checkpoint, policy_record, agent_step, epoch = maybe_load_checkpoint(
        run_dir=cfg.run_dir,
        policy_store=policy_store,
        trainer_cfg=trainer_cfg,
        metta_grid_env=metta_grid_env,
        cfg=cfg,
        device=device,
        is_master=is_master,
        rank=rank,
    )

    state.agent_step = agent_step
    state.epoch = epoch

    # Restore timer state if checkpoint exists
    if checkpoint and checkpoint.stopwatch_state is not None:
        timer.load_state(checkpoint.stopwatch_state, resume_running=True)

    state.initial_policy_record = policy_record
    state.latest_saved_policy_record = policy_record
    policy = policy_record.policy

    # Initialize policy to environment
    features = metta_grid_env.get_observation_features()
    policy.initialize_to_environment(features, metta_grid_env.action_names, metta_grid_env.max_action_args, device)

    # Validate that policy matches environment
    validate_policy_environment_match(policy, metta_grid_env)

    if trainer_cfg.compile:
        logger.info("Compiling policy")
        policy = torch.compile(policy, mode=trainer_cfg.compile_mode)

    # Create kickstarter
    kickstarter = Kickstarter(
        trainer_cfg.kickstart,
        str(device),
        policy_store,
        metta_grid_env,
    )

    # Wrap in DDP if distributed
    if torch.distributed.is_initialized():
        logger.info(f"Initializing DistributedDataParallel on device {device}")
        policy = DistributedMettaAgent(policy, device)
        torch.distributed.barrier()

    # Create experience buffer
    hidden_size, num_lstm_layers = get_lstm_config(policy)
    experience = Experience(
        total_agents=vecenv.num_agents,  # type: ignore[attr-defined]
        batch_size=trainer_cfg.batch_size,  # Already scaled if needed
        bptt_horizon=trainer_cfg.bptt_horizon,
        minibatch_size=trainer_cfg.minibatch_size,
        max_minibatch_size=trainer_cfg.minibatch_size,
        obs_space=vecenv.single_observation_space,  # type: ignore[attr-defined]
        atn_space=vecenv.single_action_space,  # type: ignore[attr-defined]
        device=device,
        hidden_size=hidden_size,
        cpu_offload=trainer_cfg.cpu_offload,
        num_lstm_layers=num_lstm_layers,
        agents_per_batch=getattr(vecenv, "agents_per_batch", None),
    )

    # Create optimizer

    optimizer_type = trainer_cfg.optimizer.type
    opt_cls = torch.optim.Adam if optimizer_type == "adam" else ForeachMuon
    # ForeachMuon expects int for weight_decay, Adam expects float
    weight_decay = trainer_cfg.optimizer.weight_decay
    if optimizer_type != "adam":
        # Ensure weight_decay is int for ForeachMuon
        weight_decay = int(weight_decay)

    optimizer = opt_cls(
        policy.parameters(),
        lr=trainer_cfg.optimizer.learning_rate,
        betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
        eps=trainer_cfg.optimizer.eps,
        weight_decay=weight_decay,  # type: ignore - ForeachMuon type annotation issue
    )

    if checkpoint and checkpoint.optimizer_state_dict:
        try:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            logger.info("Successfully loaded optimizer state from checkpoint")
        except ValueError:
            logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

    # Create lr scheduler
    lr_scheduler = None
    if trainer_cfg.lr_scheduler.enabled:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=trainer_cfg.total_timesteps // trainer_cfg.batch_size
        )

    # Set up wandb metrics
    if wandb_run and is_master:
        metrics = ["agent_step", "epoch", "total_time", "train_time"]
        for metric in metrics:
            wandb_run.define_metric(f"metric/{metric}")
        wandb_run.define_metric("*", step_metric="metric/agent_step")
        wandb_run.define_metric("overview/reward_vs_total_time", step_metric="metric/total_time")

        # Log model parameters
        num_params = sum(p.numel() for p in policy.parameters())
        if wandb_run.summary:
            wandb_run.summary["model/total_parameters"] = num_params

    # Add memory monitor tracking
    if is_master and memory_monitor:
        memory_monitor.add(experience, name="Experience", track_attributes=True)
        memory_monitor.add(policy, name="Policy", track_attributes=False)

    # Create hyperparameter scheduler
    hyperparameter_scheduler = HyperparameterScheduler(trainer_cfg, optimizer, trainer_cfg.total_timesteps, logging)

    # Return all components in the expected order
    return (
        vecenv,
        policy,
        optimizer,
        experience,
        kickstarter,
        lr_scheduler,
        hyperparameter_scheduler,
        losses,
        timer,
        torch_profiler,
        memory_monitor,
        system_monitor,
        trainer_cfg,
        device,
        is_master,
        world_size,
        rank,
        state,
        curriculum,
    )
