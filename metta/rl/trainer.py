import logging
import os
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed
import wandb
from omegaconf import DictConfig

from metta.api import TrainerState
from metta.common.util.heartbeat import record_heartbeat
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.mettagrid.mettagrid_env import dtype_actions
from metta.rl.functions import (
    accumulate_rollout_stats,
    calculate_batch_sizes,
    calculate_explained_variance,
    calculate_prioritized_sampling_params,
    cleanup_old_policies,
    compute_advantage,
    evaluate_policy,
    generate_replay,
    get_lstm_config,
    get_observation,
    maybe_update_l2_weights,
    process_minibatch_update,
    process_stats,
    run_policy_inference,
    save_policy_with_metadata,
    setup_distributed_vars,
    validate_policy_environment_match,
)
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import create_trainer_config
from metta.rl.vecenv import make_vecenv

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


def _should_run(
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


def _maybe_compute_grad_stats(policy: torch.nn.Module) -> Dict[str, float]:
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


def _maybe_save_training_state(
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

        # Send actions to environment
        with timer("_rollout.env"):
            vecenv.send(actions.cpu().numpy().astype(dtype_actions))

        # Collect info
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
    world_size: int,
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
        vecenv=vecenv,
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
    logger.info(f"Uploaded policy to wandb at epoch {policy_record.metadata.epoch}")
    return result


def _maybe_evaluate_policy(
    policy_record: Any,
    policy_store: Any,
    sim_suite_config: Any,
    stats_client: Optional[Any],
    state: TrainerState,
    device: torch.device,
    vectorization: str,
    wandb_policy_name: Optional[str] = None,
) -> Tuple[Dict[str, float], Optional[Any]]:
    """Evaluate policy."""
    try:
        eval_scores, stats_epoch_id = evaluate_policy(
            policy_record=policy_record,
            policy_store=policy_store,
            sim_suite_config=sim_suite_config,
            stats_client=stats_client,
            stats_run_id=state.stats_run_id,
            stats_epoch_start=state.stats_epoch_start,
            epoch=state.epoch,
            device=device,
            vectorization=vectorization,
            wandb_policy_name=wandb_policy_name,
        )
        state.stats_epoch_start = state.epoch + 1
        return eval_scores, stats_epoch_id
    except Exception as e:
        logger.error(f"Error evaluating policy: {e}")
        logger.error(traceback.format_exc())
        return {}, None


def _maybe_generate_replay(
    policy_record: Any,
    trainer_cfg: Any,
    policy_store: Any,
    epoch: int,
    device: torch.device,
    vectorization: str,
    wandb_run: Optional[Any] = None,
) -> None:
    """Generate replay."""
    # Get curriculum from trainer config
    curriculum = curriculum_from_config_path(trainer_cfg.curriculum_or_env, DictConfig(trainer_cfg.env_overrides))

    generate_replay(
        policy_record=policy_record,
        policy_store=policy_store,
        curriculum=curriculum,
        epoch=epoch,
        device=device,
        vectorization=vectorization,
        replay_dir=trainer_cfg.simulation.replay_dir,
        wandb_run=wandb_run,
    )


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

        logger.info(
            f"Epoch {state.epoch} - "
            f"{steps_per_sec * world_size:.0f} steps/sec "
            f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats)"
        )

        # Periodic tasks
        if _should_run(state.epoch, 10, is_master):
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
        if _should_run(state.epoch, trainer_cfg.checkpoint.checkpoint_interval):
            saved_record = _maybe_save_policy(
                policy, policy_store, state, timer, vecenv, cfg.run, is_master, trainer_cfg, world_size
            )
            if saved_record:
                state.latest_saved_policy_record = saved_record

        # Save training state
        if _should_run(state.epoch, trainer_cfg.checkpoint.checkpoint_interval):
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
                world_size=world_size,
                is_master=is_master,
            )

        # Upload to wandb
        if _should_run(state.epoch, trainer_cfg.checkpoint.wandb_checkpoint_interval, is_master):
            wandb_policy_name = _upload_policy_to_wandb(wandb_run, policy_store, state.latest_saved_policy_record)

        # Evaluate policy
        if _should_run(state.epoch, trainer_cfg.simulation.evaluate_interval, is_master):
            if state.latest_saved_policy_record:
                eval_scores, stats_epoch_id = _maybe_evaluate_policy(
                    state.latest_saved_policy_record,
                    policy_store,
                    sim_suite_config,
                    stats_client,
                    state,
                    device,
                    cfg.vectorization,
                    wandb_policy_name,
                )
                state.evals = eval_scores
                state.stats_epoch_id = stats_epoch_id

        # Generate replay
        if _should_run(state.epoch, trainer_cfg.simulation.replay_interval, is_master):
            if state.latest_saved_policy_record:
                _maybe_generate_replay(
                    state.latest_saved_policy_record,
                    trainer_cfg,
                    policy_store,
                    state.epoch,
                    device,
                    cfg.vectorization,
                    wandb_run,
                )

        # Compute gradient stats
        if _should_run(state.epoch, trainer_cfg.grad_mean_variance_interval, is_master):
            with timer("grad_stats"):
                state.grad_stats = _maybe_compute_grad_stats(policy)

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
            policy, policy_store, state, timer, vecenv, cfg.run, is_master, trainer_cfg, world_size, force=True
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
        world_size=world_size,
        is_master=is_master,
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
    from metta.agent.metta_agent import DistributedMettaAgent
    from metta.common.profiling.memory_monitor import MemoryMonitor
    from metta.common.profiling.stopwatch import Stopwatch
    from metta.common.util.system_monitor import SystemMonitor
    from metta.mettagrid.curriculum.util import curriculum_from_config_path
    from metta.mettagrid.mettagrid_env import MettaGridEnv
    from metta.rl.experience import Experience

    logger.info(f"run_dir = {cfg.run_dir}")

    # Apply batch size scaling BEFORE creating trainer config
    # This matches the behavior in tools/train.py
    if torch.distributed.is_initialized() and cfg.trainer.get("scale_batches_by_world_size", False):
        world_size = torch.distributed.get_world_size()
        # Make a mutable copy of the config to modify
        from omegaconf import OmegaConf

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
    losses = Losses()
    torch_profiler = TorchProfiler(is_master, trainer_cfg.profiler, wandb_run, cfg.run_dir)

    memory_monitor = None
    system_monitor = None
    if is_master:
        memory_monitor = MemoryMonitor()
        system_monitor = SystemMonitor(
            sampling_interval_sec=1.0,
            history_size=100,
            logger=logger,
            auto_start=True,
        )

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

    metta_grid_env: MettaGridEnv = vecenv.driver_env

    # Initialize state
    state = TrainerState()

    # Load checkpoint if exists
    checkpoint = TrainerCheckpoint.load(cfg.run_dir)
    if checkpoint:
        state.agent_step = checkpoint.agent_step
        state.epoch = checkpoint.epoch
        logger.info(f"Restored from checkpoint at {state.agent_step} steps")
        if checkpoint.stopwatch_state is not None:
            timer.load_state(checkpoint.stopwatch_state, resume_running=True)

    # Load or create policy
    policy_record = _load_or_create_policy(
        checkpoint, policy_store, trainer_cfg, metta_grid_env, cfg, device, is_master, rank
    )

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
        total_agents=vecenv.num_agents,
        batch_size=trainer_cfg.batch_size,  # Already scaled if needed
        bptt_horizon=trainer_cfg.bptt_horizon,
        minibatch_size=trainer_cfg.minibatch_size,
        max_minibatch_size=trainer_cfg.minibatch_size,
        obs_space=vecenv.single_observation_space,
        atn_space=vecenv.single_action_space,
        device=device,
        hidden_size=hidden_size,
        cpu_offload=trainer_cfg.cpu_offload,
        num_lstm_layers=num_lstm_layers,
        agents_per_batch=getattr(vecenv, "agents_per_batch", None),
    )

    # Create optimizer
    from heavyball import ForeachMuon

    optimizer_type = trainer_cfg.optimizer.type
    opt_cls = torch.optim.Adam if optimizer_type == "adam" else ForeachMuon
    optimizer = opt_cls(
        policy.parameters(),
        lr=trainer_cfg.optimizer.learning_rate,
        betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
        eps=trainer_cfg.optimizer.eps,
        weight_decay=trainer_cfg.optimizer.weight_decay,
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

    # Add memory monitor tracking
    if is_master and memory_monitor:
        memory_monitor.add(experience, name="Experience", track_attributes=True)
        memory_monitor.add(policy, name="Policy", track_attributes=False)

    return (
        vecenv,
        policy,
        optimizer,
        experience,
        kickstarter,
        lr_scheduler,
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
    )


def _load_or_create_policy(
    checkpoint: Optional[Any],
    policy_store: Any,
    trainer_cfg: Any,
    metta_grid_env: Any,
    cfg: Any,
    device: torch.device,
    is_master: bool,
    rank: int,
) -> Any:
    """Load existing policy or create new one with distributed coordination."""
    from metta.agent.metta_agent import make_policy
    from metta.common.util.fs import wait_for_file

    # Try to load from checkpoint or config
    if checkpoint and checkpoint.policy_path:
        logger.info(f"Loading policy from checkpoint: {checkpoint.policy_path}")
        policy_record = policy_store.policy_record(checkpoint.policy_path)

        # Restore original_feature_mapping from metadata if available
        if (
            hasattr(policy_record.policy, "restore_original_feature_mapping")
            and "original_feature_mapping" in policy_record.metadata
        ):
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping from checkpoint")

        return policy_record

    if trainer_cfg.initial_policy and trainer_cfg.initial_policy.uri:
        logger.info(f"Loading initial policy URI: {trainer_cfg.initial_policy.uri}")
        policy_record = policy_store.policy_record(trainer_cfg.initial_policy.uri)

        # Restore original_feature_mapping from metadata if available
        if (
            hasattr(policy_record.policy, "restore_original_feature_mapping")
            and "original_feature_mapping" in policy_record.metadata
        ):
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping from initial policy")

        return policy_record

    # Check for existing policy at default path
    default_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0))
    if os.path.exists(default_path):
        logger.info(f"Loading policy from default path: {default_path}")
        policy_record = policy_store.policy_record(default_path)

        # Restore original_feature_mapping from metadata if available
        if (
            hasattr(policy_record.policy, "restore_original_feature_mapping")
            and "original_feature_mapping" in policy_record.metadata
        ):
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info("Restored original_feature_mapping from default path")

        return policy_record

    # Create new policy
    if torch.distributed.is_initialized() and not is_master:
        # Non-master waits for master to create
        logger.info(f"Rank {rank}: Waiting for master to create policy at {default_path}")
        torch.distributed.barrier()

        if not wait_for_file(default_path, timeout=300):
            raise RuntimeError(f"Rank {rank}: Timeout waiting for policy at {default_path}")

        policy_record = policy_store.policy_record(default_path)

        # Restore original_feature_mapping from metadata if available
        if (
            hasattr(policy_record.policy, "restore_original_feature_mapping")
            and "original_feature_mapping" in policy_record.metadata
        ):
            policy_record.policy.restore_original_feature_mapping(policy_record.metadata["original_feature_mapping"])
            logger.info(f"Rank {rank}: Restored original_feature_mapping")

        return policy_record
    else:
        # Master creates new policy
        name = policy_store.make_model_name(0)
        pr = policy_store.create_empty_policy_record(name)
        pr.policy = make_policy(metta_grid_env, cfg)
        saved_pr = policy_store.save(pr)
        logger.info(f"Created and saved new policy to {saved_pr.uri}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return saved_pr
