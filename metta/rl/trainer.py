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
from metta.rl.evaluate import evaluate_policy, generate_policy_replay
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.rollout import rollout
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.train import train_ppo
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import create_trainer_config
from metta.rl.util.batch_utils import calculate_batch_sizes
from metta.rl.util.distributed import setup_distributed_vars
from metta.rl.util.evaluation import upload_replay_html
from metta.rl.util.optimization import (
    compute_gradient_stats,
    maybe_update_l2_weights,
)
from metta.rl.util.policy_management import (
    cleanup_old_policies,
    maybe_load_checkpoint,
    save_policy_with_metadata,
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
    """Create training components individually, similar to run.py."""
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

    # Initialize state containers
    stats_tracker = StatsTracker(rollout_stats=defaultdict(list))
    eval_scores = EvalRewardSummary()  # Initialize eval_scores with empty summary

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

    hyperparameter_scheduler = None  # Disabled for now

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
        agent_step,
        epoch,
        stats_tracker,
        latest_saved_policy_record,
        initial_policy_uri,
        initial_generation,
        eval_scores,
        curriculum,
    )


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
        agent_step,
        epoch,
        stats_tracker,
        latest_saved_policy_record,
        initial_policy_uri,
        initial_generation,
        eval_scores,
        curriculum,
    ) = create_training_components(
        cfg=cfg,
        wandb_run=wandb_run,
        policy_store=policy_store,
        sim_suite_config=sim_suite_config,
        stats_client=stats_client,
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

                # Update learning rate scheduler
                if lr_scheduler is not None:
                    lr_scheduler.step()

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
        if should_run(epoch, trainer_cfg.checkpoint.checkpoint_interval):
            saved_record = _maybe_save_policy(
                policy,
                policy_store,
                latest_saved_policy_record,
                initial_policy_uri,
                initial_generation,
                agent_step,
                epoch,
                eval_scores,
                timer,
                cfg.run,
                is_master,
                trainer_cfg,
            )
            if saved_record:
                latest_saved_policy_record = saved_record

        # Save training state
        if should_run(epoch, trainer_cfg.checkpoint.checkpoint_interval):
            _maybe_save_training_state(
                checkpoint_dir=cfg.run_dir,
                agent_step=agent_step,
                epoch=epoch,
                optimizer=optimizer,
                timer=timer,
                latest_saved_policy_uri=latest_saved_policy_record.uri if latest_saved_policy_record else None,
                kickstarter=kickstarter,
                trainer_cfg=trainer_cfg,
                is_master=is_master,
            )

        # Upload to wandb
        if should_run(epoch, trainer_cfg.checkpoint.wandb_checkpoint_interval, is_master):
            wandb_policy_name = _upload_policy_to_wandb(wandb_run, policy_store, latest_saved_policy_record)

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
        saved_record = _maybe_save_policy(
            policy,
            policy_store,
            latest_saved_policy_record,
            initial_policy_uri,
            initial_generation,
            agent_step,
            epoch,
            eval_scores,
            timer,
            cfg.run,
            is_master,
            trainer_cfg,
            force=True,
        )
        if saved_record:
            latest_saved_policy_record = saved_record

    _maybe_save_training_state(
        checkpoint_dir=cfg.run_dir,
        agent_step=agent_step,
        epoch=epoch,
        optimizer=optimizer,
        timer=timer,
        latest_saved_policy_uri=latest_saved_policy_record.uri if latest_saved_policy_record else None,
        kickstarter=kickstarter,
        trainer_cfg=trainer_cfg,
        is_master=is_master,
        force=True,
    )

    if wandb_run and latest_saved_policy_record:
        _upload_policy_to_wandb(wandb_run, policy_store, latest_saved_policy_record, force=True)

    # Cleanup
    vecenv.close()
    if is_master:
        if memory_monitor:
            memory_monitor.clear()
        if system_monitor:
            system_monitor.stop()


def _maybe_save_policy(
    policy: Any,
    policy_store: Any,
    latest_saved_policy_record: Optional[Any],
    initial_policy_uri: Optional[str],
    initial_generation: int,
    agent_step: int,
    epoch: int,
    evals: Any,  # EvalRewardSummary
    timer: Any,
    run_name: str,
    is_master: bool,
    trainer_cfg: Any,
    force: bool = False,
) -> Optional[Any]:
    """Save policy with distributed synchronization."""
    # Check if should save
    should_save = force or (
        trainer_cfg.checkpoint.checkpoint_interval and epoch % trainer_cfg.checkpoint.checkpoint_interval == 0
    )
    if not should_save:
        return None

    # All ranks participate in barrier for distributed sync
    if not is_master:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return None

    # Create a temporary initial_policy_record for save_policy_with_metadata
    # This is a bit of a hack to maintain compatibility with existing function
    initial_policy_record = None
    if initial_policy_uri:
        initial_policy_record = type(
            "obj", (object,), {"uri": initial_policy_uri, "metadata": {"generation": initial_generation}}
        )()

    # Save policy with metadata
    saved_record = save_policy_with_metadata(
        policy=policy,
        policy_store=policy_store,
        epoch=epoch,
        agent_step=agent_step,
        evals=evals,
        timer=timer,
        initial_policy_record=initial_policy_record,
        run_name=run_name,
        is_master=is_master,
    )

    if saved_record:
        # Clean up old policies periodically
        if epoch % 10 == 0:
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

    try:
        result = policy_store.add_to_wandb_run(wandb_run.id, policy_record)
        logger.info(f"Uploaded policy to wandb at epoch {policy_record.metadata.get('epoch', 'unknown')}")
        return result
    except Exception as e:
        logger.warning(f"Failed to upload policy to wandb: {e}")
        return None


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
