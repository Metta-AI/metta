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

from metta.agent.metta_agent import DistributedMettaAgent
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
    maybe_load_checkpoint,
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

    # Restore timer state if checkpoint exists
    if checkpoint and checkpoint.stopwatch_state is not None:
        timer.load_state(checkpoint.stopwatch_state, resume_running=True)

    # Extract initial policy info
    latest_saved_policy_record = policy_record
    initial_policy_uri = policy_record.uri if policy_record else None
    initial_generation = policy_record.metadata.get("generation", 0) if policy_record else 0

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

    # Create master-only components
    memory_monitor, system_monitor = create_master_trainer_components(
        policy=policy,
        experience=experience,
        wandb_run=wandb_run,
        is_master=is_master,
        timer=timer,
    )

    # Initialize stats tracking
    _initialize_stats_tracking(stats_tracker, stats_client, wandb_run)

    logger.info(f"Training on {device}")
    wandb_policy_name: str | None = None

    # Main training loop
    while agent_step < trainer_cfg.total_timesteps:
        steps_before = agent_step

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
                initial_policy_record = None
                if initial_policy_uri:
                    initial_policy_record = type(
                        "obj", (object,), {"uri": initial_policy_uri, "metadata": {"generation": initial_generation}}
                    )()

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

        # Calculate performance metrics
        rollout_time = timer.get_last_elapsed("_rollout")
        train_time = timer.get_last_elapsed("_train")
        stats_time = timer.get_last_elapsed("_process_stats")
        steps_calculated = agent_step - steps_before

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
            f"Epoch {epoch}- "
            f"{steps_per_sec:.0f} SPS- "
            f"step {agent_step}/{total_steps_str}- "
            f"({train_pct:.0f}% train- {rollout_pct:.0f}% rollout- {stats_pct:.0f}% stats)"
        )

        # Periodic tasks
        if should_run(epoch, 10, is_master):
            record_heartbeat()

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
            initial_policy_record = None
            if initial_policy_uri:
                initial_policy_record = type(
                    "obj", (object,), {"uri": initial_policy_uri, "metadata": {"generation": initial_generation}}
                )()

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

        # Evaluate policy
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

    logger.info("Training complete!")
    timing_summary = timer.get_all_summaries()

    for name, summary in timing_summary.items():
        logger.info(f"  {name}: {timer.format_time(summary['total_elapsed'])}")

    # Force final saves
    if is_master:
        # Create initial policy record for metadata if needed
        initial_policy_record = None
        if initial_policy_uri:
            initial_policy_record = type(
                "obj", (object,), {"uri": initial_policy_uri, "metadata": {"generation": initial_generation}}
            )()

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


def _initialize_stats_tracking(
    stats_tracker: StatsTracker,
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
        stats_tracker.stats_run_id = stats_client.create_training_run(
            name=name, attributes={}, url=url, description=description, tags=tags
        ).id
    except Exception as e:
        logger.warning(f"Failed to create training run: {e}")
