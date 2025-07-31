import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed
from heavyball import ForeachMuon
from omegaconf import DictConfig

from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.core.distributed import setup_distributed_vars
from metta.core.monitoring import (
    cleanup_monitoring,
    setup_monitoring,
)
from metta.eval.eval_request_config import EvalRewardSummary
from metta.mettagrid import MettaGridEnv
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.evaluate import evaluate_policy
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.optimization import (
    compute_gradient_stats,
    maybe_update_l2_weights,
)
from metta.rl.policy_management import (
    load_or_initialize_policy,
    validate_policy_environment_match,
    wrap_agent_distributed,
)
from metta.rl.ppo import ppo
from metta.rl.rollout import get_lstm_config, rollout
from metta.rl.stats import (
    StatsTracker,
    accumulate_rollout_stats,
    process_stats,
)
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import create_trainer_config
from metta.rl.utils import (
    log_training_progress,
    should_run,
)
from metta.rl.vecenv import make_vecenv
from metta.rl.wandb import (
    abort_requested,
    log_model_parameters,
    setup_wandb_metrics,
    upload_policy_artifact,
)
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.utils.batch import calculate_batch_sizes

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


def train(
    cfg: DictConfig,
    wandb_run: Any | None,
    policy_store: Any,
    sim_suite_config: Any,
    stats_client: Any | None,
    **kwargs: Any,
) -> None:
    """Functional training loop."""
    logger.info(f"run_dir = {cfg.run_dir}")

    # Log recent checkpoints for debugging
    checkpoints_dir = Path(cfg.run_dir) / "checkpoints"
    if checkpoints_dir.exists():
        files = sorted(os.listdir(checkpoints_dir))[-3:]
        if files:
            logger.info(f"Recent checkpoints: {', '.join(files)}")

    # Create trainer config from Hydra config
    trainer_cfg = create_trainer_config(cfg)

    # Set up distributed
    is_master, world_size, rank = setup_distributed_vars()
    device = torch.device(cfg.device) if isinstance(cfg.device, str) else cfg.device

    # Create timer, Losses, profiler, curriculum
    timer = Stopwatch(logger)
    timer.start()
    losses = Losses()
    torch_profiler = TorchProfiler(is_master, trainer_cfg.profiler, wandb_run, cfg.run_dir)
    curriculum = curriculum_from_config_path(trainer_cfg.curriculum_or_env, DictConfig(trainer_cfg.env_overrides))

    # Calculate batch sizes
    num_agents = curriculum.get_task().env_cfg().game.num_agents
    target_batch_size, batch_size, num_envs = calculate_batch_sizes(
        trainer_cfg.forward_pass_minibatch_target_size,
        num_agents,
        trainer_cfg.num_workers,
        trainer_cfg.async_factor,
    )

    # Create vectorized environment
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

    # Load checkpoint if it exists
    checkpoint = TrainerCheckpoint.load(cfg.run_dir)
    agent_step = checkpoint.agent_step if checkpoint else 0
    epoch = checkpoint.epoch if checkpoint else 0

    if checkpoint:
        logger.info(f"Restored from checkpoint at {agent_step} steps")
        if checkpoint.stopwatch_state is not None:
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
        torch.distributed.barrier()
        policy = wrap_agent_distributed(policy, device)
        torch.distributed.barrier()

    # Initialize policy to environment after distributed wrapping
    # This must happen after wrapping to ensure all ranks do it at the same time
    if hasattr(policy, "initialize_to_environment"):
        features = metta_grid_env.get_observation_features()
        policy.initialize_to_environment(features, metta_grid_env.action_names, metta_grid_env.max_action_args, device)
    else:
        policy.activate_actions(metta_grid_env.action_names, metta_grid_env.max_action_args, device)

    # Get LSTM configuration
    hidden_size, num_lstm_layers = get_lstm_config(policy)

    # Create experience buffer
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
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=trainer_cfg.optimizer.learning_rate,
            betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
            eps=trainer_cfg.optimizer.eps,
            weight_decay=trainer_cfg.optimizer.weight_decay,
        )
    elif optimizer_type == "muon":
        # ForeachMuon expects int for weight_decay
        optimizer = ForeachMuon(
            policy.parameters(),
            lr=trainer_cfg.optimizer.learning_rate,
            betas=(trainer_cfg.optimizer.beta1, trainer_cfg.optimizer.beta2),
            eps=trainer_cfg.optimizer.eps,
            weight_decay=int(trainer_cfg.optimizer.weight_decay),
        )
    else:
        raise ValueError(f"Optimizer type must be 'adam' or 'muon', got {optimizer_type}")

    if checkpoint and checkpoint.optimizer_state_dict:
        try:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            logger.info("Successfully loaded optimizer state from checkpoint")
        except ValueError:
            logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

    # Set up monitoring (master only)
    if is_master:
        logger.info("Starting training")
        memory_monitor, system_monitor = setup_monitoring(
            policy=policy,
            experience=experience,
            timer=timer,
        )
    else:
        memory_monitor, system_monitor = None, None

    # Set up wandb metrics (master only)
    if wandb_run and is_master:
        setup_wandb_metrics(wandb_run)
        log_model_parameters(policy, wandb_run)

    # Initialize stats tracking
    stats_tracker = StatsTracker(rollout_stats=defaultdict(list))
    if stats_client is not None:
        # Extract wandb attributes with defaults
        name = getattr(wandb_run, "name", "unknown") or "unknown"
        url = getattr(wandb_run, "url", None)
        tags = list(wandb_run.tags) if wandb_run and wandb_run.tags else None
        description = getattr(wandb_run, "notes", None)

        try:
            stats_tracker.stats_run_id = stats_client.create_training_run(
                name=name, attributes={}, url=url, description=description, tags=tags
            ).id
        except Exception as e:
            logger.warning(f"Failed to create training run: {e}")

    if is_master:
        logger.info(f"Training on {device}")
    wandb_policy_name: str | None = None

    # Main training loop
    while agent_step < trainer_cfg.total_timesteps:
        steps_before = agent_step
        record_heartbeat()

        with torch_profiler:
            # ===== ROLLOUT PHASE =====
            with timer("_rollout"):
                num_steps, raw_infos = rollout(
                    vecenv=vecenv,
                    policy=policy,
                    experience=experience,
                    device=device,
                    timer=timer,
                )
                agent_step += num_steps * world_size
            accumulate_rollout_stats(raw_infos, stats_tracker.rollout_stats)

            # ===== TRAINING PHASE =====
            with timer("_train"):
                epochs_trained = ppo(
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
        if is_master:
            rollout_time = timer.get_last_elapsed("_rollout")
            train_time = timer.get_last_elapsed("_train")
            stats_time = timer.get_last_elapsed("_process_stats")
            steps_calculated = agent_step - steps_before

            total_time = train_time + rollout_time + stats_time
            steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

            log_training_progress(
                epoch=epoch,
                agent_step=agent_step,
                total_timesteps=trainer_cfg.total_timesteps,
                steps_per_sec=steps_per_sec,
                train_time=train_time,
                rollout_time=rollout_time,
                stats_time=stats_time,
                is_master=is_master,
            )

        # Update L2 weights if configured
        if interval := getattr(policy, "l2_init_weight_update_interval", 0):
            maybe_update_l2_weights(policy, epoch, interval, is_master)

        # Save policy - all ranks must participate in checkpoint decision
        if checkpoint_manager.should_checkpoint(epoch):
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

                # Only master saves training state
                if is_master:
                    checkpoint_manager.save_checkpoint(
                        agent_step=agent_step,
                        epoch=epoch,
                        optimizer=optimizer,
                        policy_path=saved_record.uri,
                        timer=timer,
                        run_dir=cfg.run_dir,
                        kickstarter=kickstarter,
                    )

            # All ranks must synchronize after checkpoint operations
            # This barrier must be outside the if saved_record block so all ranks hit it
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        # Upload to wandb
        if should_run(epoch, trainer_cfg.checkpoint.wandb_checkpoint_interval, is_master):
            wandb_policy_name = upload_policy_artifact(wandb_run, policy_store, latest_saved_policy_record)

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

                # Create extended simulation suite that includes the training task
                extended_suite_config = SimulationSuiteConfig(
                    name=sim_suite_config.name,
                    simulations=dict(sim_suite_config.simulations),
                    env_overrides=sim_suite_config.env_overrides,
                    num_episodes=sim_suite_config.num_episodes,
                )

                # Add training task to the suite
                # Pass the config as _pre_built_env_config to avoid Hydra loading
                task_cfg = curriculum.get_task().env_cfg()
                training_task_config = SingleEnvSimulationConfig(
                    env="eval/training_task",  # Just a descriptive name
                    num_episodes=1,
                    env_overrides={"_pre_built_env_config": task_cfg},
                )
                extended_suite_config.simulations["eval/training_task"] = training_task_config

                # Evaluate policy using the extracted evaluation function
                eval_scores = evaluate_policy(
                    policy_record=latest_saved_policy_record,
                    policy_uri=latest_saved_policy_record.uri,
                    sim_suite_config=extended_suite_config,
                    device=device,
                    vectorization=cfg.vectorization,
                    replay_dir=trainer_cfg.simulation.replay_dir,
                    stats_epoch_id=stats_tracker.stats_epoch_id,
                    wandb_policy_name=wandb_policy_name,
                    policy_store=policy_store,
                    stats_client=stats_client,
                    cfg=cfg,
                    wandb_run=wandb_run,
                    trainer_cfg=trainer_cfg,
                    agent_step=agent_step,
                    epoch=epoch,
                )

                stats_tracker.update_epoch_tracking(epoch + 1)

        # Compute gradient stats
        if should_run(epoch, trainer_cfg.grad_mean_variance_interval, is_master):
            with timer("grad_stats"):
                stats_tracker.grad_stats = compute_gradient_stats(policy)

        # Check for abort every 5 epochs
        if is_master and wandb_run and epoch % 5 == 0:
            if abort_requested(wandb_run, min_interval_sec=60):
                logger.info("Abort tag detected. Stopping the run.")
                trainer_cfg.total_timesteps = int(agent_step)
                wandb_run.config.update({"trainer.total_timesteps": trainer_cfg.total_timesteps}, allow_val_change=True)
                break

    if is_master:
        logger.info("Training complete!")
        timing_summary = timer.get_all_summaries()
        for name, summary in timing_summary.items():
            logger.info(f"  {name}: {timer.format_time(summary['total_elapsed'])}")

    # Force final saves - all ranks must participate
    if is_master:
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

    # All ranks must synchronize after final save operations
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if wandb_run and latest_saved_policy_record:
        upload_policy_artifact(wandb_run, policy_store, latest_saved_policy_record, force=True)

    # Final synchronization before cleanup
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Cleanup
    vecenv.close()
    cleanup_monitoring(memory_monitor, system_monitor)
