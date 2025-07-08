import logging
import os
import traceback
from typing import Any, Optional

import torch
import torch.distributed
import wandb
from omegaconf import DictConfig

from metta.api import (
    TrainerState,
    TrainingComponents,
    create_training_components,
)
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

    # Create components and state
    components, state = create_training_components(
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

    logger.info(f"Training on {components.device}")
    wandb_policy_name: str | None = None

    # Main training loop
    while state.agent_step < components.trainer_cfg.total_timesteps:
        steps_before = state.agent_step

        with components.torch_profiler:
            # Rollout phase
            with components.timer("_rollout"):
                num_steps, raw_infos = rollout(
                    vecenv=components.vecenv,
                    policy=components.policy,
                    experience=components.experience,
                    device=components.device,
                    timer=components.timer,
                )
                state.agent_step += num_steps

                # Process rollout stats
                accumulate_rollout_stats(raw_infos, state.stats)

            # Training phase
            with components.timer("_train"):
                epochs_trained = train_epoch(
                    policy=components.policy,
                    optimizer=components.optimizer,
                    experience=components.experience,
                    kickstarter=components.kickstarter,
                    losses=components.losses,
                    trainer_cfg=components.trainer_cfg,
                    agent_step=state.agent_step,
                    epoch=state.epoch,
                    device=components.device,
                )
                state.epoch += epochs_trained

                # Update learning rate scheduler
                if components.lr_scheduler is not None:
                    components.lr_scheduler.step()

        components.torch_profiler.on_epoch_end(state.epoch)

        # Process stats
        with components.timer("_process_stats"):
            _process_stats_functional(components, state)

        # Calculate performance metrics
        rollout_time = components.timer.get_last_elapsed("_rollout")
        train_time = components.timer.get_last_elapsed("_train")
        stats_time = components.timer.get_last_elapsed("_process_stats")
        steps_calculated = state.agent_step - steps_before

        total_time = train_time + rollout_time + stats_time
        steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

        train_pct = (train_time / total_time) * 100
        rollout_pct = (rollout_time / total_time) * 100
        stats_pct = (stats_time / total_time) * 100

        logger.info(
            f"Epoch {state.epoch} - "
            f"{steps_per_sec * components.world_size:.0f} steps/sec "
            f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats)"
        )

        # Periodic tasks
        _maybe_record_heartbeat(state.epoch, components.is_master)
        _maybe_save_policy(components, state)
        _maybe_save_training_state(components, state)
        wandb_policy_name = _maybe_upload_policy_to_wandb(components, state)
        _maybe_evaluate_policy(components, state, wandb_policy_name)
        _maybe_generate_replay(components, state)
        _maybe_compute_grad_stats(components, state)
        _maybe_update_l2_weights(components, state)

        # Check for abort (if using AbortingTrainer behavior)
        if _check_abort(components, state):
            break

    logger.info("Training complete!")
    timing_summary = components.timer.get_all_summaries()

    for name, summary in timing_summary.items():
        logger.info(f"  {name}: {components.timer.format_time(summary['total_elapsed'])}")

    # Force final saves
    _maybe_save_policy(components, state, force=True)
    _maybe_save_training_state(components, state, force=True)
    _maybe_upload_policy_to_wandb(components, state, force=True)

    # Cleanup
    components.vecenv.close()
    if components.is_master:
        if components.memory_monitor:
            components.memory_monitor.clear()
        if components.system_monitor:
            components.system_monitor.stop()


def _process_stats_functional(components: TrainingComponents, state: TrainerState) -> None:
    """Process stats for the functional trainer."""
    if not components.is_master or not components.wandb_run:
        state.stats.clear()
        state.grad_stats.clear()
        return

    process_stats(
        stats=state.stats,
        losses=components.losses,
        evals=state.evals,
        grad_stats=state.grad_stats,
        experience=components.experience,
        policy=components.policy,
        timer=components.timer,
        trainer_cfg=components.trainer_cfg,
        agent_step=state.agent_step,
        epoch=state.epoch,
        world_size=components.world_size,
        wandb_run=components.wandb_run,
        memory_monitor=components.memory_monitor,
        system_monitor=components.system_monitor,
        latest_saved_policy_record=state.latest_saved_policy_record,
        initial_policy_record=state.initial_policy_record,
        optimizer=components.optimizer,
    )

    # Clear stats after processing
    state.stats.clear()
    state.grad_stats.clear()


def _maybe_record_heartbeat(epoch: int, is_master: bool, force: bool = False) -> None:
    """Record heartbeat if on interval."""
    if not should_run_on_interval(epoch, 10, is_master, force):
        return
    record_heartbeat()


def _maybe_save_policy(components: TrainingComponents, state: TrainerState, force: bool = False) -> None:
    """Save policy if on checkpoint interval."""
    interval = components.trainer_cfg.checkpoint.checkpoint_interval
    if not force and interval and state.epoch % interval != 0:
        return

    # All ranks participate in barrier for distributed sync
    if not components.is_master:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return

    # Save policy with metadata
    saved_record = save_policy_with_metadata(
        policy=components.policy,
        policy_store=components.policy_store,
        epoch=state.epoch,
        agent_step=state.agent_step,
        evals=state.evals,
        timer=components.timer,
        vecenv=components.vecenv,
        initial_policy_record=state.initial_policy_record,
        run_name=getattr(components.cfg, "run", "unknown"),
        is_master=components.is_master,
    )

    if saved_record:
        state.latest_saved_policy_record = saved_record

        # Clean up old policies periodically
        if state.epoch % 10 == 0:
            cleanup_old_policies(components.trainer_cfg.checkpoint.checkpoint_dir, keep_last_n=5)

    # Sync all ranks after save
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _maybe_save_training_state(components: TrainingComponents, state: TrainerState, force: bool = False) -> None:
    """Save training state if on checkpoint interval."""
    interval = components.trainer_cfg.checkpoint.checkpoint_interval
    if not force and interval and state.epoch % interval != 0:
        return

    latest_uri = state.latest_saved_policy_record.uri if state.latest_saved_policy_record else None

    save_training_state(
        checkpoint_dir=components.cfg.run_dir,
        agent_step=state.agent_step,
        epoch=state.epoch,
        optimizer=components.optimizer,
        timer=components.timer,
        latest_saved_policy_uri=latest_uri,
        kickstarter=components.kickstarter,
        world_size=components.world_size,
        is_master=components.is_master,
    )


def _maybe_upload_policy_to_wandb(
    components: TrainingComponents, state: TrainerState, force: bool = False
) -> Optional[str]:
    """Upload policy to wandb if on interval."""
    interval = components.trainer_cfg.checkpoint.wandb_checkpoint_interval
    if not should_run_on_interval(state.epoch, interval, components.is_master, force):
        return None

    if not components.wandb_run or not state.latest_saved_policy_record:
        return None

    if not components.wandb_run.name:
        logger.warning("No wandb run name was provided")
        return None

    result = components.policy_store.add_to_wandb_run(components.wandb_run.name, state.latest_saved_policy_record)
    logger.info(f"Uploaded policy to wandb at epoch {state.epoch}")
    return result


def _maybe_evaluate_policy(
    components: TrainingComponents, state: TrainerState, wandb_policy_name: Optional[str] = None, force: bool = False
) -> None:
    """Evaluate policy if on evaluation interval."""
    interval = components.trainer_cfg.simulation.evaluate_interval
    if not should_run_on_interval(state.epoch, interval, components.is_master, force):
        return

    try:
        eval_scores, stats_epoch_id = evaluate_policy(
            policy_record=state.latest_saved_policy_record,
            policy_store=components.policy_store,
            sim_suite_config=components.sim_suite_config,
            stats_client=components.stats_client,
            stats_run_id=state.stats_run_id,
            stats_epoch_start=state.stats_epoch_start,
            epoch=state.epoch,
            device=components.device,
            vectorization=components.cfg.vectorization,
            wandb_policy_name=wandb_policy_name,
        )
        state.evals = eval_scores
        state.stats_epoch_id = stats_epoch_id
        state.stats_epoch_start = state.epoch + 1
    except Exception as e:
        logger.error(f"Error evaluating policy: {e}")
        logger.error(traceback.format_exc())


def _maybe_generate_replay(components: TrainingComponents, state: TrainerState, force: bool = False) -> None:
    """Generate replay if on replay interval."""
    interval = components.trainer_cfg.simulation.replay_interval
    if not should_run_on_interval(state.epoch, interval, components.is_master, force):
        return

    # Get curriculum from components
    curriculum = curriculum_from_config_path(
        components.trainer_cfg.curriculum_or_env, DictConfig(components.trainer_cfg.env_overrides)
    )

    generate_replay(
        policy_record=state.latest_saved_policy_record,
        policy_store=components.policy_store,
        curriculum=curriculum,
        epoch=state.epoch,
        device=components.device,
        vectorization=components.cfg.vectorization,
        replay_dir=components.trainer_cfg.simulation.replay_dir,
        wandb_run=components.wandb_run,
    )


def _maybe_compute_grad_stats(components: TrainingComponents, state: TrainerState, force: bool = False) -> None:
    """Compute gradient stats if on interval."""
    interval = components.trainer_cfg.grad_mean_variance_interval
    if not should_run_on_interval(state.epoch, interval, components.is_master, force):
        return

    with components.timer("grad_stats"):
        state.grad_stats = compute_gradient_stats(components.policy)


def _maybe_update_l2_weights(components: TrainingComponents, state: TrainerState, force: bool = False) -> None:
    """Update L2 init weights if on interval."""
    interval = components.cfg.agent.l2_init_weight_update_interval
    if not should_run_on_interval(state.epoch, interval, components.is_master, force):
        return

    if hasattr(components.policy, "update_l2_init_weight_copy"):
        components.policy.update_l2_init_weight_copy()


def _check_abort(components: TrainingComponents, state: TrainerState) -> bool:
    """Check for abort tag in wandb run (AbortingTrainer behavior)."""
    if components.wandb_run is None:
        return False

    try:
        if "abort" not in wandb.Api().run(components.wandb_run.path).tags:
            return False

        logger.info("Abort tag detected. Stopping the run.")
        components.trainer_cfg.total_timesteps = int(state.agent_step)
        components.wandb_run.config.update(
            {"trainer.total_timesteps": components.trainer_cfg.total_timesteps}, allow_val_change=True
        )
        return True
    except Exception:
        return False
