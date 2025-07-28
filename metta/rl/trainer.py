import asyncio
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from heavyball import ForeachMuon
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table

from metta.agent.metta_agent import DistributedMettaAgent
from metta.app_backend.routes.eval_task_routes import TaskCreateRequest
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.eval.eval_request_config import EvalRewardSummary
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.evaluate import evaluate_policy, generate_policy_replay
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.rollout import rollout
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.train import train_ppo
from metta.rl.trainer_config import create_trainer_config
from metta.rl.util.batch_utils import calculate_batch_sizes
from metta.rl.util.distributed import setup_distributed_vars
from metta.rl.util.optimization import (
    compute_gradient_stats,
    maybe_update_l2_weights,
)
from metta.rl.util.policy_management import (
    load_or_initialize_policy,
    validate_policy_environment_match,
)
from metta.rl.util.rollout import get_lstm_config
from metta.rl.util.stats import (
    StatsTracker,
    accumulate_rollout_stats,
    process_stats,
)
from metta.rl.util.utils import check_abort, should_run
from metta.rl.vecenv import make_vecenv
from metta.rl.wandb import log_model_parameters, setup_wandb_metrics, upload_policy_to_wandb, upload_replay_html
from metta.sim.utils import get_or_create_policy_ids, wandb_policy_name_to_uri

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


def create_training_components(
    cfg: Any,
    wandb_run: Optional[Any],
    policy_store: Any,
    sim_suite_config: Any,
    stats_client: Optional[Any] = None,
) -> Tuple[Any, ...]:
    """Create training components needed on all ranks."""
    logger.info(f"run_dir = {cfg.run_dir}")

    # Log recent checkpoints like the MettaTrainer did
    checkpoints_dir = Path(cfg.run_dir) / "checkpoints"
    if checkpoints_dir.exists():
        files = sorted(os.listdir(checkpoints_dir))
        recent_files = files[-3:] if len(files) >= 3 else files
        logger.info(f"Recent checkpoints: {', '.join(recent_files)}")

    trainer_cfg = create_trainer_config(cfg)

    # Set up distributed
    is_master, world_size, rank = setup_distributed_vars()
    device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device

    # Create utilities
    timer = Stopwatch(logger)
    timer.start()

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

    # Initialize state containers
    stats_tracker = StatsTracker(rollout_stats=defaultdict(list))
    eval_scores = EvalRewardSummary()  # Initialize eval_scores with empty summary

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=trainer_cfg.checkpoint.checkpoint_dir,
        policy_store=policy_store,
        trainer_cfg=trainer_cfg,
        device=device,
        is_master=is_master,
        rank=rank,
        run_name=cfg.run,
    )

    # Load checkpoint and policy with distributed coordination
    from metta.rl.trainer_checkpoint import TrainerCheckpoint

    checkpoint = TrainerCheckpoint.load(cfg.run_dir)
    agent_step = 0
    epoch = 0

    if checkpoint:
        agent_step = checkpoint.agent_step
        epoch = checkpoint.epoch
        logger.info(f"Restored from checkpoint at {agent_step} steps")

    # Restore timer state if checkpoint exists
    if checkpoint and checkpoint.stopwatch_state is not None:
        timer.load_state(checkpoint.stopwatch_state, resume_running=True)

    # Load or initialize policy with distributed coordination
    policy, initial_policy_record, latest_saved_policy_record = load_or_initialize_policy(
        cfg=cfg,
        checkpoint=checkpoint,
        policy_store=policy_store,
        metta_grid_env=metta_grid_env,
        device=device,
        is_master=is_master,
        rank=rank,
    )

    # Extract initial policy info
    initial_policy_uri = initial_policy_record.uri if initial_policy_record else None
    initial_generation = initial_policy_record.metadata.get("generation", 0) if initial_policy_record else 0

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

    hyperparameter_scheduler = None  # Disabled for now

    # Return all components in the expected order
    return (
        vecenv,
        policy,
        optimizer,
        experience,
        kickstarter,
        hyperparameter_scheduler,
        losses,
        timer,
        torch_profiler,
        trainer_cfg,
        device,
        is_master,
        world_size,
        rank,
        agent_step,
        epoch,
        stats_tracker,
        latest_saved_policy_record,
        initial_policy_uri,
        initial_generation,
        eval_scores,
        curriculum,
        checkpoint_manager,
    )


def create_master_trainer_components(
    policy: Any,
    experience: Experience,
    wandb_run: Optional[Any],
    is_master: bool,
    timer: Optional[Stopwatch] = None,
) -> Tuple[Optional[MemoryMonitor], Optional[SystemMonitor]]:
    """Create components only needed on the master rank.

    Args:
        policy: The policy model
        experience: The experience buffer
        wandb_run: Weights & Biases run object
        is_master: Whether this is the master rank
        timer: Stopwatch timer instance (optional)

    Returns:
        Tuple of (memory_monitor, system_monitor)
    """
    memory_monitor = None
    system_monitor = None

    if is_master:
        # Create memory monitor
        memory_monitor = MemoryMonitor()
        memory_monitor.add(experience, name="Experience", track_attributes=True)
        memory_monitor.add(policy, name="Policy", track_attributes=False)

        # Create system monitor
        system_monitor = SystemMonitor(
            sampling_interval_sec=1.0,
            history_size=100,
            logger=logger,
            auto_start=True,
            external_timer=timer,
        )

        # Set up wandb metrics
        if wandb_run and is_master:
            setup_wandb_metrics(wandb_run)
            log_model_parameters(policy, wandb_run)

    return memory_monitor, system_monitor


def log_master(message: str, level: str = "info", is_master: bool = True) -> None:
    """Log a message only on the master node.

    Args:
        message: The message to log
        level: The log level ('debug', 'info', 'warning', 'error', 'critical')
        is_master: Whether this is the master rank
    """
    if not is_master:
        return

    log_func = getattr(logger, level, logger.info)
    log_func(message)


def log_training_status(
    epoch: int,
    agent_step: int,
    total_timesteps: int,
    timer: Stopwatch,
    steps_before: int,
    is_master: bool,
) -> None:
    """Log training status with rich console output."""
    if not is_master:
        return

    rollout_time = timer.get_last_elapsed("_rollout")
    train_time = timer.get_last_elapsed("_train")
    stats_time = timer.get_last_elapsed("_process_stats")
    steps_calculated = agent_step - steps_before

    total_time = train_time + rollout_time + stats_time
    steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

    train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
    rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0
    stats_pct = (stats_time / total_time) * 100 if total_time > 0 else 0

    console = Console()
    table = Table(
        title=f"[bold cyan]Training Progress - Epoch {epoch}[/bold cyan]",
        show_header=True,
        header_style="bold magenta",
    )

    # Add columns
    table.add_column("Metric", style="cyan", justify="left")
    table.add_column("Progress", style="green", justify="right")
    table.add_column("Rate", style="yellow", justify="left")

    # Add rows
    progress_pct = (agent_step / total_timesteps) * 100
    table.add_row(
        "Agent Steps",
        f"{agent_step:,} / {total_timesteps:,} ({progress_pct:.1f}%)",
        f"[dim]{steps_per_sec:.0f} steps/sec[/dim]",
    )

    table.add_row(
        "Epoch Time",
        f"{total_time:.2f}s",
        f"[dim]Train: {train_pct:.0f}% | Rollout: {rollout_pct:.0f}% | Stats: {stats_pct:.0f}%[/dim]",
    )

    # Log the table
    console.print(table)


def train(
    cfg: DictConfig,
    wandb_run: Any | None,
    policy_store: Any,
    sim_suite_config: Any,
    stats_client: Any | None,
    **kwargs: Any,
) -> None:
    """Functional training loop replacing MettaTrainer.train()."""
    # Create all components individually first to get is_master value
    (
        vecenv,
        policy,
        optimizer,
        experience,
        kickstarter,
        hyperparameter_scheduler,
        losses,
        timer,
        torch_profiler,
        trainer_cfg,
        device,
        is_master,
        world_size,
        rank,
        agent_step,
        epoch,
        stats_tracker,
        latest_saved_policy_record,
        initial_policy_uri,
        initial_generation,
        eval_scores,
        curriculum,
        checkpoint_manager,
    ) = create_training_components(
        cfg=cfg,
        wandb_run=wandb_run,
        policy_store=policy_store,
        sim_suite_config=sim_suite_config,
        stats_client=stats_client,
    )

    log_master("Starting training", is_master=is_master)

    # Create master-only components
    memory_monitor, system_monitor = create_master_trainer_components(
        policy=policy,
        experience=experience,
        wandb_run=wandb_run,
        is_master=is_master,
        timer=timer,
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
            stats_tracker.stats_run_id = stats_client.create_training_run(
                name=name, attributes={}, url=url, description=description, tags=tags
            ).id
        except Exception as e:
            logger.warning(f"Failed to create training run: {e}")

    log_master(f"Training on {device}", is_master=is_master)
    wandb_policy_name: str | None = None

    # Main training loop
    while agent_step < trainer_cfg.total_timesteps:
        steps_before = agent_step
        record_heartbeat()

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
                agent_step += num_steps * world_size

                # Process rollout stats
                accumulate_rollout_stats(raw_infos, stats_tracker.rollout_stats)

            # Training phase
            with timer("_train"):
                epochs_trained = train_ppo(
                    policy=policy,
                    optimizer=optimizer,
                    experience=experience,
                    kickstarter=kickstarter,
                    losses=losses,
                    trainer_cfg=trainer_cfg,
                    agent_step=agent_step,
                    epoch=epoch,
                    device=device,
                )
                epoch += epochs_trained

        torch_profiler.on_epoch_end(epoch)

        # Process stats
        with timer("_process_stats"):
            if is_master and wandb_run:
                # Create temporary initial_policy_record for process_stats
                from metta.agent.policy_metadata import PolicyMetadata
                from metta.agent.policy_record import PolicyRecord

                initial_policy_record = None
                if initial_policy_uri:
                    # Create a minimal PolicyRecord for stats tracking
                    metadata = PolicyMetadata(generation=initial_generation)
                    initial_policy_record = PolicyRecord(
                        policy_store=policy_store, run_name="", uri=initial_policy_uri, metadata=metadata
                    )

                process_stats(
                    stats=stats_tracker.rollout_stats,
                    losses=losses,
                    evals=eval_scores,
                    grad_stats=stats_tracker.grad_stats,
                    experience=experience,
                    policy=policy,
                    timer=timer,
                    trainer_cfg=trainer_cfg,
                    agent_step=agent_step,
                    epoch=epoch,
                    world_size=world_size,
                    wandb_run=wandb_run,
                    memory_monitor=memory_monitor,
                    system_monitor=system_monitor,
                    latest_saved_policy_record=latest_saved_policy_record,
                    initial_policy_record=initial_policy_record,
                    optimizer=optimizer,
                    kickstarter=kickstarter,
                )
            # Clear stats after processing
            stats_tracker.clear_rollout_stats()
            stats_tracker.clear_grad_stats()

        # Log training status
        log_training_status(
            epoch=epoch,
            agent_step=agent_step,
            total_timesteps=trainer_cfg.total_timesteps,
            timer=timer,
            steps_before=steps_before,
            is_master=is_master,
        )

        # Update L2 weights if configured
        if hasattr(policy, "l2_init_weight_update_interval"):
            maybe_update_l2_weights(
                agent=policy,
                epoch=epoch,
                interval=getattr(policy, "l2_init_weight_update_interval", 0),
                is_master=is_master,
            )

        # Save policy
        if checkpoint_manager.should_checkpoint(epoch):
            # Create initial policy record for metadata if needed
            from metta.agent.policy_metadata import PolicyMetadata
            from metta.agent.policy_record import PolicyRecord

            initial_policy_record = None
            if initial_policy_uri:
                metadata = PolicyMetadata(generation=initial_generation)
                initial_policy_record = PolicyRecord(
                    policy_store=policy_store, run_name="", uri=initial_policy_uri, metadata=metadata
                )

            saved_record = checkpoint_manager.save_policy(
                policy=policy,
                epoch=epoch,
                agent_step=agent_step,
                evals=eval_scores,
                timer=timer,
                initial_policy_record=initial_policy_record,
            )
            if saved_record:
                latest_saved_policy_record = saved_record

                # Save training state with the new policy path
                checkpoint_manager.save_checkpoint(
                    agent_step=agent_step,
                    epoch=epoch,
                    optimizer=optimizer,
                    policy_path=saved_record.uri,
                    timer=timer,
                    run_dir=cfg.run_dir,
                    kickstarter=kickstarter,
                )

        # Upload to wandb
        if should_run(epoch, trainer_cfg.checkpoint.wandb_checkpoint_interval, is_master):
            wandb_policy_name = upload_policy_to_wandb(wandb_run, policy_store, latest_saved_policy_record)

        # Evaluate policy (with remote evaluation support)
        if should_run(epoch, trainer_cfg.simulation.evaluate_interval, is_master):
            if latest_saved_policy_record:
                # Create stats epoch if needed
                if stats_client is not None and stats_tracker.stats_run_id is not None:
                    stats_tracker.stats_epoch_id = stats_client.create_epoch(
                        run_id=stats_tracker.stats_run_id,
                        start_training_epoch=stats_tracker.stats_epoch_start,
                        end_training_epoch=epoch,
                        attributes={},
                    ).id

                # Check for remote evaluation
                if (
                    trainer_cfg.simulation.evaluate_remote
                    and wandb_run
                    and stats_client
                    and wandb_policy_name  # ensures it was uploaded to wandb
                ):
                    # Remote evaluation
                    if ":" not in wandb_policy_name:
                        logger.warning(f"Remote evaluation: {wandb_policy_name} does not specify a version")
                    else:
                        internal_wandb_policy_name, wandb_uri = wandb_policy_name_to_uri(wandb_policy_name)
                        stats_server_policy_id = get_or_create_policy_ids(
                            stats_client,
                            [(internal_wandb_policy_name, wandb_uri, wandb_run.notes)],
                            stats_tracker.stats_epoch_id,
                        ).get(internal_wandb_policy_name)
                        if not stats_server_policy_id:
                            logger.warning(
                                f"Remote evaluation: failed to get or register policy ID for {wandb_policy_name}"
                            )
                        else:
                            task = asyncio.run(
                                stats_client.create_task(
                                    TaskCreateRequest(
                                        policy_id=stats_server_policy_id,
                                        git_hash=trainer_cfg.simulation.git_hash,
                                        sim_suite=sim_suite_config.name,
                                    )
                                )
                            )
                            logger.info(f"Remote evaluation: created task {task.id} for policy {wandb_policy_name}")

                # Local evaluation
                eval_scores = evaluate_policy(
                    latest_saved_policy_record,
                    sim_suite_config,
                    curriculum,
                    stats_client,
                    stats_tracker,
                    agent_step,
                    epoch,
                    device,
                    cfg.vectorization,
                    trainer_cfg.simulation.replay_dir,
                    wandb_policy_name,
                    policy_store,
                    cfg,
                    wandb_run,
                    logger,
                )
                stats_tracker.update_epoch_tracking(epoch + 1)

        # Generate replay
        if should_run(epoch, trainer_cfg.simulation.evaluate_interval, is_master):
            if latest_saved_policy_record:
                replay_url = generate_policy_replay(
                    policy_record=latest_saved_policy_record,
                    policy_store=policy_store,
                    trainer_cfg=trainer_cfg,
                    epoch=epoch,
                    device=device,
                    vectorization=cfg.vectorization,
                    wandb_run=wandb_run,
                )
                if replay_url:
                    upload_replay_html(
                        replay_urls={"replay": [replay_url]},
                        agent_step=agent_step,
                        epoch=epoch,
                        wandb_run=wandb_run,
                    )

        # Compute gradient stats
        if should_run(epoch, trainer_cfg.grad_mean_variance_interval, is_master):
            with timer("grad_stats"):
                stats_tracker.grad_stats = compute_gradient_stats(policy)

        # Check for abort
        if check_abort(wandb_run, trainer_cfg, agent_step):
            break

    log_master("Training complete!", is_master=is_master)
    timing_summary = timer.get_all_summaries()

    for name, summary in timing_summary.items():
        log_master(f"  {name}: {timer.format_time(summary['total_elapsed'])}", is_master=is_master)

    # Force final saves
    if is_master:
        # Create initial policy record for metadata if needed
        from metta.agent.policy_metadata import PolicyMetadata
        from metta.agent.policy_record import PolicyRecord

        initial_policy_record = None
        if initial_policy_uri:
            metadata = PolicyMetadata(generation=initial_generation)
            initial_policy_record = PolicyRecord(
                policy_store=policy_store, run_name="", uri=initial_policy_uri, metadata=metadata
            )

        saved_record = checkpoint_manager.save_policy(
            policy=policy,
            epoch=epoch,
            agent_step=agent_step,
            evals=eval_scores,
            timer=timer,
            initial_policy_record=initial_policy_record,
            force=True,
        )
        if saved_record:
            latest_saved_policy_record = saved_record

            # Save final training state
            checkpoint_manager.save_checkpoint(
                agent_step=agent_step,
                epoch=epoch,
                optimizer=optimizer,
                policy_path=saved_record.uri,
                timer=timer,
                run_dir=cfg.run_dir,
                kickstarter=kickstarter,
                force=True,
            )

    if wandb_run and latest_saved_policy_record:
        upload_policy_to_wandb(wandb_run, policy_store, latest_saved_policy_record, force=True)

    # Cleanup
    vecenv.close()
    if is_master:
        if memory_monitor:
            memory_monitor.clear()
        if system_monitor:
            system_monitor.stop()
