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
from metta.rl.functions import (
    accumulate_rollout_stats,
    cleanup_old_policies,
    compute_gradient_stats,
    evaluate_policy,
    generate_replay,
    process_stats,
    rollout,
    save_policy_with_metadata,
    save_training_state,
    should_run_on_interval,
    train_epoch,
)

torch.set_float32_matmul_precision("high")

# Get rank for logger name
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
logger = logging.getLogger(f"trainer-{rank}-{local_rank}")


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
    if stats_client is not None:
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

    logger.info(f"Training on {device}")
    wandb_policy_name: str | None = None

    # Main training loop
    while state.agent_step < trainer_cfg.total_timesteps:
        steps_before = state.agent_step

        with torch_profiler:
            # Rollout phase
            with timer("_rollout"):
                num_steps, raw_infos = rollout(
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
                epochs_trained = train_epoch(
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
        if should_run_on_interval(state.epoch, 10, is_master):
            record_heartbeat()

        # Save policy
        if _should_save_policy(state.epoch, trainer_cfg.checkpoint.checkpoint_interval):
            saved_record = _save_policy(
                policy, policy_store, state, timer, vecenv, cfg.run, is_master, trainer_cfg, world_size
            )
            if saved_record:
                state.latest_saved_policy_record = saved_record

        # Save training state
        if _should_save_policy(state.epoch, trainer_cfg.checkpoint.checkpoint_interval):
            save_training_state(
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
        if should_run_on_interval(state.epoch, trainer_cfg.checkpoint.wandb_checkpoint_interval, is_master):
            wandb_policy_name = _upload_policy_to_wandb(wandb_run, policy_store, state.latest_saved_policy_record)

        # Evaluate policy
        if should_run_on_interval(state.epoch, trainer_cfg.simulation.evaluate_interval, is_master):
            if state.latest_saved_policy_record:
                eval_scores, stats_epoch_id = _evaluate_policy(
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
        if should_run_on_interval(state.epoch, trainer_cfg.simulation.replay_interval, is_master):
            if state.latest_saved_policy_record:
                _generate_replay(
                    state.latest_saved_policy_record,
                    trainer_cfg,
                    policy_store,
                    state.epoch,
                    device,
                    cfg.vectorization,
                    wandb_run,
                )

        # Compute gradient stats
        if should_run_on_interval(state.epoch, trainer_cfg.grad_mean_variance_interval, is_master):
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
        saved_record = _save_policy(
            policy, policy_store, state, timer, vecenv, cfg.run, is_master, trainer_cfg, world_size, force=True
        )
        if saved_record:
            state.latest_saved_policy_record = saved_record

    save_training_state(
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
    from metta.rl.functions import calculate_batch_sizes, get_lstm_config, setup_distributed_vars
    from metta.rl.kickstarter import Kickstarter
    from metta.rl.losses import Losses
    from metta.rl.torch_profiler import TorchProfiler
    from metta.rl.trainer_checkpoint import TrainerCheckpoint
    from metta.rl.trainer_config import parse_trainer_config
    from metta.rl.vecenv import make_vecenv

    logger.info(f"run_dir = {cfg.run_dir}")

    trainer_cfg = parse_trainer_config(cfg)

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

    if trainer_cfg.compile:
        logger.info("Compiling policy")
        policy = torch.compile(policy, mode=trainer_cfg.compile_mode)

    # Create kickstarter
    kickstarter = Kickstarter(
        trainer_cfg.kickstart,
        device,
        policy_store,
        metta_grid_env.action_names,
        metta_grid_env.max_action_args,
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
        batch_size=trainer_cfg.batch_size
        if not trainer_cfg.scale_batches_by_world_size
        else trainer_cfg.batch_size // world_size,
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


def _should_save_policy(epoch: int, interval: int, force: bool = False) -> bool:
    """Check if policy should be saved."""
    return force or (interval and epoch % interval == 0)


def _save_policy(
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
    if not _should_save_policy(state.epoch, trainer_cfg.checkpoint.checkpoint_interval, force):
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


def _evaluate_policy(
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


def _generate_replay(
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
        return policy_store.policy_record(checkpoint.policy_path)

    if trainer_cfg.initial_policy and trainer_cfg.initial_policy.uri:
        logger.info(f"Loading initial policy URI: {trainer_cfg.initial_policy.uri}")
        return policy_store.policy_record(trainer_cfg.initial_policy.uri)

    # Check for existing policy at default path
    default_path = os.path.join(trainer_cfg.checkpoint.checkpoint_dir, policy_store.make_model_name(0))
    if os.path.exists(default_path):
        logger.info(f"Loading policy from default path: {default_path}")
        return policy_store.policy_record(default_path)

    # Create new policy
    if torch.distributed.is_initialized() and not is_master:
        # Non-master waits for master to create
        logger.info(f"Rank {rank}: Waiting for master to create policy at {default_path}")
        torch.distributed.barrier()

        if not wait_for_file(default_path, timeout=300):
            raise RuntimeError(f"Rank {rank}: Timeout waiting for policy at {default_path}")

        return policy_store.policy_record(default_path)
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
