import os
from collections import defaultdict
from typing import cast

import numpy as np
import torch
import torch.distributed
from heavyball import ForeachMuon
from torchrl.data import Composite

from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import MettaAgent, PolicyAgent
from metta.app_backend.clients.stats_client import StatsClient
from metta.cogworks.curriculum.curriculum import Curriculum
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.log_config import getRankAwareLogger
from metta.common.wandb.wandb_context import WandbRun
from metta.core.distributed import TorchDistributedConfig
from metta.core.monitoring import (
    cleanup_monitoring,
    setup_monitoring,
)
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.eval.eval_service import evaluate_policy
from metta.mettagrid import MettaGridEnv, dtype_actions
from metta.mettagrid.profiling.stopwatch import Stopwatch
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.evaluate import evaluate_policy_remote_with_checkpoint_manager, upload_replay_html
from metta.rl.experience import Experience
from metta.rl.hyperparameter_scheduler import step_hyperparameters
from metta.rl.losses import get_loss_experience_spec
from metta.rl.optimization import (
    compute_gradient_stats,
)
from metta.rl.policy_management import (
    wrap_agent_distributed,
)
from metta.rl.rollout import get_observation, send_observation
from metta.rl.stats import (
    StatsTracker,
    accumulate_rollout_stats,
    process_stats,
)
from metta.rl.system_config import SystemConfig
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_config import TrainerConfig
from metta.rl.trainer_state import TrainerState
from metta.rl.utils import (
    log_training_progress,
    should_run,
)
from metta.rl.vecenv import make_vecenv
from metta.rl.wandb import (
    abort_requested,
    log_model_parameters,
    setup_wandb_metrics,
)
from metta.sim.simulation_config import SimulationConfig
from metta.utils.batch import calculate_batch_sizes

try:
    from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib  # type: ignore[reportUnusedImport]
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, "
        "try installing with --no-build-isolation"
    ) from None

torch.set_float32_matmul_precision("high")


logger = getRankAwareLogger(__name__)


def distributed_barrier(description=""):
    """Simple barrier wrapper with timeout warnings."""
    if not torch.distributed.is_initialized():
        return
    import math
    import threading
    import time

    start = time.time()
    done = threading.Event()

    def _barrier():
        torch.distributed.barrier()
        done.set()

    threading.Thread(target=_barrier, daemon=True).start()

    warned = False  # Track first warning
    while not done.wait(1.0):
        elapsed = time.time() - start
        int_elapsed = math.ceil(elapsed)
        if int_elapsed % 100 == 0:
            logger.warning(f"Barrier '{description}' very slow: {elapsed:.0f}s")
        elif int_elapsed > 2 and not warned:
            logger.warning(f"Barrier '{description}' slow: {elapsed:.0f}s")
            warned = True


def _update_training_status_on_failure(stats_client: StatsClient | None, stats_run_id, logger) -> None:
    """Helper to update training run status to 'failed' when training encounters an error."""
    if stats_client and stats_run_id:
        try:
            stats_client.update_training_run_status(stats_run_id, "failed")
            logger.info("Training run status updated to 'failed'")
        except Exception as e:
            logger.warning(f"Failed to update training run status to failed: {e}", exc_info=True)


def train(
    run_dir: str,
    run: str,
    system_cfg: SystemConfig,
    agent_cfg: AgentConfig,
    device: torch.device,
    trainer_cfg: TrainerConfig,
    wandb_run: WandbRun | None,
    checkpoint_manager: CheckpointManager,
    stats_client: StatsClient | None,
    torch_dist_cfg: TorchDistributedConfig,
) -> None:
    """Main training loop for Metta agents."""
    logger.info(f"run_dir = {run_dir}")

    checkpoints_dir = trainer_cfg.checkpoint.checkpoint_dir
    if os.path.exists(checkpoints_dir):
        files = sorted(os.listdir(checkpoints_dir))[-3:]
        if files:
            logger.info(f"Recent checkpoints: {', '.join(files)}")

    # Create timer, Losses, profiler, curriculum
    timer = Stopwatch(log_level=logger.getEffectiveLevel())
    timer.start()
    torch_profiler = TorchProfiler(torch_dist_cfg.is_master, trainer_cfg.profiler, wandb_run, run_dir)
    curriculum = Curriculum(trainer_cfg.curriculum)

    # Calculate batch sizes
    num_agents = curriculum.get_task().get_env_cfg().game.num_agents
    target_batch_size, batch_size, num_envs = calculate_batch_sizes(
        trainer_cfg.forward_pass_minibatch_target_size,
        num_agents,
        trainer_cfg.rollout_workers,
        trainer_cfg.async_factor,
    )

    # Create vectorized environment
    vecenv = make_vecenv(
        curriculum,
        system_cfg.vectorization,
        num_envs=num_envs,
        batch_size=batch_size,
        num_workers=trainer_cfg.rollout_workers,
        zero_copy=trainer_cfg.zero_copy,
        is_training=True,
    )

    vecenv.async_reset(system_cfg.seed + torch_dist_cfg.rank)

    metta_grid_env: MettaGridEnv = vecenv.driver_env  # type: ignore[attr-defined]

    # Initialize state containers
    eval_scores = EvalRewardSummary()  # Initialize eval_scores with empty summary

    # Distributed checkpoint loading coordination
    # Master determines which checkpoint to load, then all ranks load the same one
    if torch_dist_cfg.is_master:
        # Check if checkpoints exist and load trainer state
        trainer_state = checkpoint_manager.load_trainer_state()

        # Determine checkpoint epoch to load (None if no checkpoints)
        checkpoint_epoch = trainer_state["epoch"] if trainer_state else None

        # Load the agent if checkpoint exists
        if checkpoint_epoch is not None:
            existing_agent = checkpoint_manager.load_agent(epoch=checkpoint_epoch, device=device)
        else:
            existing_agent = None
    else:
        trainer_state = None
        checkpoint_epoch = None
        existing_agent = None

    # Synchronize checkpoint epoch and trainer state across all ranks
    if torch.distributed.is_initialized():
        from metta.agent.util.distribution_utils import get_from_master

        checkpoint_epoch = get_from_master(checkpoint_epoch)
        trainer_state = get_from_master(trainer_state)

    agent_step = trainer_state["agent_step"] if trainer_state else 0
    epoch = trainer_state["epoch"] if trainer_state else 0
    latest_saved_epoch = epoch  # Track the epoch of the latest saved checkpoint
    latest_wandb_uri = None  # Track the last uploaded wandb artifact URI

    if trainer_state:
        logger.info(f"Restored from checkpoint at {agent_step} steps")
        # Restore stopwatch state if available
        if "stopwatch_state" in trainer_state:
            timer.load_state(trainer_state["stopwatch_state"], resume_running=True)

    # Create or load agent with distributed coordination
    # Now all ranks know the exact checkpoint epoch to load
    if checkpoint_epoch is not None:
        if torch_dist_cfg.is_master:
            policy_agent = existing_agent
        else:
            policy_agent = checkpoint_manager.load_agent(epoch=checkpoint_epoch, device=device)
    else:
        policy_agent = MettaAgent(metta_grid_env, system_cfg, agent_cfg)

    # Ensure all ranks have created/loaded their policy before continuing
    if torch.distributed.is_initialized():
        distributed_barrier("after policies created/loaded")

    policy: PolicyAgent = policy_agent

    if trainer_cfg.compile:
        logger.info("Compiling policy")
        # torch.compile gives a CallbackFunctionType, but it preserves the interface of the original policy
        policy = cast(PolicyAgent, torch.compile(policy, mode=trainer_cfg.compile_mode))

    # Wrap in DDP if distributed
    if torch.distributed.is_initialized():
        if torch_dist_cfg.is_master:
            logger.info("Initializing DistributedDataParallel")
        distributed_barrier("before DDP wrapper")
        policy = wrap_agent_distributed(policy, device)
        distributed_barrier("after DDP wrapper")

    # Initialize policy to environment after distributed wrapping
    # This must happen after wrapping to ensure all ranks do it at the same time
    policy.train()  # Set to training mode for training
    features = metta_grid_env.get_observation_features()
    policy.initialize_to_environment(features, metta_grid_env.action_names, metta_grid_env.max_action_args, device)

    # Instantiate configured composable losses dynamically
    loss_instances = trainer_cfg.losses.init_losses(policy, trainer_cfg, vecenv, device, checkpoint_manager)

    # Get the experience buffer specification from the policy
    policy_spec = policy.get_agent_experience_spec()
    act_space = vecenv.single_action_space
    act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
    loss_spec = get_loss_experience_spec(act_space.nvec, act_dtype)

    # Merge experience specs required by all losses
    merged_spec_dict: dict = dict(policy_spec.items())
    for inst in loss_instances.values():
        spec = inst.get_experience_spec()
        merged_spec_dict.update(dict(spec.items()))
    merged_spec_dict.update(dict(loss_spec.items()))

    # Create experience buffer
    experience = Experience(
        total_agents=vecenv.num_agents,
        batch_size=trainer_cfg.batch_size,
        bptt_horizon=trainer_cfg.bptt_horizon,
        minibatch_size=trainer_cfg.minibatch_size,
        max_minibatch_size=trainer_cfg.minibatch_size,
        experience_spec=Composite(merged_spec_dict),
        device=device,
    )

    for loss_instance in loss_instances.values():
        loss_instance.attach_replay_buffer(experience)
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

    if trainer_state and "optimizer_state" in trainer_state:
        try:
            optimizer.load_state_dict(trainer_state["optimizer_state"])
            logger.info("Successfully loaded optimizer state from checkpoint")
        except ValueError:
            logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

    # Set up monitoring (master only)
    if torch_dist_cfg.is_master:
        logger.info("Starting training")
        memory_monitor, system_monitor = setup_monitoring(
            policy=policy,
            experience=experience,
            timer=timer,
        )
    else:
        memory_monitor, system_monitor = None, None

    # Set up wandb metrics (master only)
    if wandb_run and torch_dist_cfg.is_master:
        setup_wandb_metrics(wandb_run)
        log_model_parameters(policy, wandb_run)

    # Initialize stats tracking
    stats_tracker = StatsTracker(rollout_stats=defaultdict(list))
    if stats_client is not None:
        # Extract wandb attributes with defaults
        name = url = "unknown"
        description: str | None = None
        tags: list[str] | None = None
        if wandb_run:
            name = wandb_run.name or name
            url = wandb_run.url
            if wandb_run.tags:
                tags = list(wandb_run.tags)
            description = wandb_run.notes

        try:
            stats_tracker.stats_run_id = stats_client.create_training_run(
                name=name, url=url, description=description, tags=tags
            ).id
        except Exception as e:
            logger.warning(f"Failed to create training run: {e}", exc_info=True)

    if torch_dist_cfg.is_master:
        logger.info(f"Training on {device}")

    # Main training loop
    trainer_state = TrainerState(
        agent_step=agent_step,
        epoch=0,
        update_epoch=0,
        mb_idx=0,
        optimizer=optimizer,
    )
    try:
        while agent_step < trainer_cfg.total_timesteps:
            distributed_barrier(f"on agent_step {agent_step}")

            steps_before = agent_step
            trainer_state.agent_step = agent_step
            trainer_state.epoch = epoch
            all_losses = list(loss_instances.keys())
            shared_loss_mb_data = experience.give_me_empty_md_td()
            policy.on_new_training_run()
            for _loss_name in loss_instances.keys():
                loss_instances[_loss_name].on_new_training_run(trainer_state)
                shared_loss_mb_data[_loss_name] = experience.give_me_empty_md_td()

            # Initialize main's traditional loss system alongside composable system
            record_heartbeat()

            with torch_profiler:
                # Rollout phase
                with timer("_rollout"):
                    raw_infos = []
                    experience.reset_for_rollout()
                    for _loss_name in list(all_losses):
                        loss_instances[_loss_name].on_rollout_start(trainer_state)

                    buffer_step = experience.buffer[experience.ep_indices, experience.ep_lengths - 1]
                    buffer_step = buffer_step.select(*policy_spec.keys())

                    while not experience.ready_for_training and not trainer_state.stop_rollout:
                        o, r, d, t, info, training_env_id, _, num_steps = get_observation(vecenv, device, timer)

                        trainer_state.training_env_id = training_env_id
                        td = buffer_step[training_env_id].clone()
                        td["env_obs"] = o
                        td["rewards"] = r
                        td["dones"] = d.float()
                        td["truncateds"] = t.float()
                        td.set(
                            "training_env_id_start",
                            torch.full(
                                td.batch_size,
                                training_env_id.start,
                                device=td.device,
                                dtype=torch.long,
                            ),
                        )

                        # Inference - hybrid approach: run composable losses rollout hooks first
                        # note that each loss will modify the td, the same one that is passed to other losses.
                        # We want this because this allows other parts of the network to only run what's needed on
                        # these obs, efficiently reusing hiddens within the network. Other losses should clear fields
                        # and/or clone as necessary.
                        for _lname in list(all_losses):
                            loss_obj = loss_instances[_lname]
                            loss_obj.rollout(td, trainer_state)

                        # If no composable losses did inference, do it here
                        # This fallback is needed when losses are disabled by scheduling or no losses do inference
                        if "actions" not in td:
                            with torch.no_grad():
                                policy(td)
                            # Store experience since no loss did it
                            experience.store(data_td=td, env_id=training_env_id)

                        send_observation(vecenv, td["actions"], dtype_actions, timer)

                        if info:
                            raw_infos.extend(info)

                        agent_step += num_steps * torch_dist_cfg.world_size
                    accumulate_rollout_stats(raw_infos, stats_tracker.rollout_stats)

                # Training phase
                with timer("_train"):
                    # Reset loss tracking
                    shared_loss_mb_data.zero_()
                    experience.reset_importance_sampling_ratios()

                    epochs_trained = 0
                    for _lname in list(all_losses):
                        loss_obj = loss_instances[_lname]
                        loss_obj.zero_loss_tracker()

                    for _update_epoch in range(trainer_cfg.update_epochs):
                        trainer_state.update_epoch = _update_epoch
                        for mb_idx in range(experience.num_minibatches):
                            trainer_state.mb_idx = mb_idx
                            trainer_state.stop_update_epoch = False

                            # Use composable losses system
                            total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                            for _lname in list(all_losses):
                                loss_obj = loss_instances[_lname]
                                loss_val, shared_loss_mb_data = loss_obj.train(shared_loss_mb_data, trainer_state)
                                total_loss = total_loss + loss_val

                            if trainer_state.stop_update_epoch:
                                break

                            optimizer.zero_grad()

                            # This also serves as a barrier for all ranks
                            total_loss.backward()

                            if (mb_idx + 1) % experience.accumulate_minibatches == 0:
                                # Get max_grad_norm from first loss config that has it (typically PPO)
                                max_grad_norm = 0.5  # default fallback
                                for loss_inst in loss_instances.values():
                                    if hasattr(loss_inst.loss_cfg, "max_grad_norm"):
                                        max_grad_norm = loss_inst.loss_cfg.max_grad_norm
                                        break
                                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                                optimizer.step()

                                policy.clip_weights()

                                if device.type == "cuda":
                                    torch.cuda.synchronize()

                            for _lname in list(all_losses):
                                loss_obj = loss_instances[_lname]
                                loss_obj.on_mb_end(trainer_state)

                        epochs_trained += 1

                    for _lname in list(all_losses):
                        loss_obj = loss_instances[_lname]
                        loss_obj.on_train_phase_end(trainer_state)

                epoch += epochs_trained
                trainer_state.epoch = epoch
                trainer_state.agent_step = agent_step  # update agent_step count state not in between rollout and train

            # Update hyperparameters based on current training step (master only)
            if torch_dist_cfg.is_master:
                step_hyperparameters(trainer_cfg, optimizer, agent_step, trainer_cfg.total_timesteps, logger)

            # Safe to proceed to next rollout phase only once all ranks have completed training
            if torch.distributed.is_initialized():
                distributed_barrier("after training phase")

            if not torch_dist_cfg.is_master:
                # Only master needs to do bookkeeping
                continue

            torch_profiler.on_epoch_end(epoch)

            losses_stats = {}
            for _lname in list(all_losses):
                loss_obj = loss_instances[_lname]
                losses_stats.update(loss_obj.stats())

            with timer("_process_stats"):
                if wandb_run:
                    process_stats(
                        agent_cfg=agent_cfg,
                        stats=stats_tracker.rollout_stats,
                        losses_stats=losses_stats,
                        evals=eval_scores,
                        grad_stats=stats_tracker.grad_stats,
                        experience=experience,
                        policy=policy,
                        timer=timer,
                        trainer_cfg=trainer_cfg,
                        agent_step=agent_step,
                        epoch=epoch,
                        wandb_run=wandb_run,
                        # We know these exist within master
                        memory_monitor=memory_monitor,  # type: ignore[arg-type]
                        system_monitor=system_monitor,  # type: ignore[arg-type]
                        optimizer=optimizer,
                        latest_saved_epoch=latest_saved_epoch,
                    )
                # Clear stats after processing
                stats_tracker.clear_rollout_stats()
                stats_tracker.clear_grad_stats()

            log_training_progress(
                epoch=epoch,
                agent_step=agent_step,
                prev_agent_step=steps_before,
                total_timesteps=trainer_cfg.total_timesteps,
                train_time=timer.get_last_elapsed("_train"),
                rollout_time=timer.get_last_elapsed("_rollout"),
                stats_time=timer.get_last_elapsed("_process_stats"),
                run_name=run,
            )
            if should_run(epoch, trainer_cfg.checkpoint.checkpoint_interval):
                # Extract the actual agent from distributed wrapper if needed
                agent_to_save = policy.module if torch.distributed.is_initialized() else policy

                # Build metadata from evaluation scores
                metadata = {
                    "agent_step": agent_step,
                    "total_time": timer.get_elapsed(),
                    "total_train_time": timer.get_all_elapsed().get("_rollout", 0)
                    + timer.get_all_elapsed().get("_train", 0),
                }

                # Add evaluation scores if available
                if eval_scores.category_scores or eval_scores.simulation_scores:
                    metadata.update(
                        {
                            "score": eval_scores.avg_simulation_score,
                            "avg_reward": eval_scores.avg_category_score,
                            "category_scores": eval_scores.category_scores,
                            "simulation_scores": {
                                f"{cat}/{sim}": score for (cat, sim), score in eval_scores.simulation_scores.items()
                            },
                        }
                    )

                # Save agent and trainer state
                # Only upload to wandb if we're at the right interval
                should_upload_wandb = wandb_run and should_run(epoch, trainer_cfg.checkpoint.wandb_checkpoint_interval)
                metadata["upload_to_wandb"] = should_upload_wandb

                wandb_uri = checkpoint_manager.save_agent(
                    agent_to_save, epoch, metadata, wandb_run=wandb_run if should_upload_wandb else None
                )
                checkpoint_manager.save_trainer_state(optimizer, epoch, agent_step, timer.save_state())
                latest_saved_epoch = epoch

                if wandb_uri:
                    latest_wandb_uri = wandb_uri
                    logger.info(f"Saved checkpoint to wandb: {latest_wandb_uri}")

            if trainer_cfg.evaluation and should_run(epoch, trainer_cfg.evaluation.evaluate_interval):
                # Evaluation with CheckpointManager - use current policy directly
                if stats_client and stats_tracker.stats_run_id:
                    stats_tracker.stats_epoch_id = stats_client.create_epoch(
                        run_id=stats_tracker.stats_run_id,
                        start_training_epoch=stats_tracker.stats_epoch_start,
                        end_training_epoch=epoch,
                    ).id

                sims = [
                    SimulationConfig(
                        name=f"train_task_{i}",
                        env=curriculum.get_task().get_env_cfg(),
                    )
                    for i in range(trainer_cfg.evaluation.num_training_tasks)
                ]
                sims.extend(trainer_cfg.evaluation.simulations)

                evaluate_local = trainer_cfg.evaluation.evaluate_local
                if latest_wandb_uri:
                    policy_uri = latest_wandb_uri  # Already a wandb:// URI
                else:
                    checkpoint_uris = checkpoint_manager.select_checkpoints("latest", count=1)
                    policy_uri = checkpoint_uris[0] if checkpoint_uris else None
                if trainer_cfg.evaluation.evaluate_remote:
                    try:
                        # Get the most recent checkpoint URI for remote evaluation
                        # Prefer wandb artifact if available, otherwise use local file
                        if policy_uri:
                            logger.info(f"Evaluating policy remotely from {policy_uri}")
                            evaluate_policy_remote_with_checkpoint_manager(
                                policy_uri=policy_uri,
                                simulations=sims,
                                stats_epoch_id=stats_tracker.stats_epoch_id,
                                stats_client=stats_client,
                                wandb_run=wandb_run,
                                trainer_cfg=trainer_cfg,
                            )
                        else:
                            logger.warning("No checkpoint available for remote evaluation")
                    except Exception as e:
                        logger.error(f"Failed to evaluate policy remotely: {e}", exc_info=True)
                        logger.error("Falling back to local evaluation")
                        evaluate_local = True
                if evaluate_local:
                    if policy_uri:
                        evaluation_results = evaluate_policy(
                            checkpoint_uri=policy_uri,
                            simulations=sims,
                            device=device,
                            vectorization=system_cfg.vectorization,
                            replay_dir=trainer_cfg.evaluation.replay_dir if trainer_cfg.evaluation else None,
                            stats_epoch_id=stats_tracker.stats_epoch_id,
                            stats_client=stats_client,
                        )
                        logger.info("Simulation complete")
                        eval_scores = evaluation_results.scores
                        if wandb_run is not None and evaluation_results.replay_urls:
                            upload_replay_html(
                                replay_urls=evaluation_results.replay_urls,
                                agent_step=agent_step,
                                epoch=epoch,
                                wandb_run=wandb_run,
                                step_metric_key="metric/epoch",
                                epoch_metric_key="metric/epoch",
                            )
                    else:
                        logger.warning("No checkpoint available for local evaluation")
                        evaluation_results = EvalResults(scores=EvalRewardSummary(), replay_urls={})
                        eval_scores = evaluation_results.scores

                stats_tracker.update_epoch_tracking(epoch + 1)

            # Compute gradient stats
            if should_run(epoch, trainer_cfg.grad_mean_variance_interval):
                with timer("grad_stats"):
                    stats_tracker.grad_stats = compute_gradient_stats(policy)

            # Check for abort every 5 epochs
            if should_run(epoch, 5):
                if wandb_run and abort_requested(wandb_run, min_interval_sec=60):
                    logger.info("Abort tag detected. Stopping the run.")
                    trainer_cfg.total_timesteps = int(agent_step)
                    wandb_run.config.update(
                        {"trainer.total_timesteps": trainer_cfg.total_timesteps}, allow_val_change=True
                    )
                    break

        # All ranks wait until training is complete before closing vecenv
        if torch.distributed.is_initialized():
            distributed_barrier("before closing vecenv")
    except:
        _update_training_status_on_failure(stats_client, stats_tracker.stats_run_id, logger)
        raise

    vecenv.close()

    if not torch_dist_cfg.is_master:
        return

    logger.info("Training complete!")

    # Update training run status to completed
    if stats_client and stats_tracker.stats_run_id:
        try:
            stats_client.update_training_run_status(stats_tracker.stats_run_id, "completed")
            logger.info("Training run status updated to 'completed'")
        except Exception as e:
            logger.warning(f"Failed to update training run status to completed: {e}", exc_info=True)

    timing_summary = timer.get_all_summaries()
    for name, summary in timing_summary.items():
        logger.info(f"  {name}: {timer.format_time(summary['total_elapsed'])}")

    # Final checkpoint save at end of training
    agent_to_save = policy.module if torch.distributed.is_initialized() else policy

    final_metadata = {
        "agent_step": agent_step,
        "total_time": timer.get_elapsed(),
        "total_train_time": timer.get_all_elapsed().get("_rollout", 0) + timer.get_all_elapsed().get("_train", 0),
    }

    # Add final evaluation scores if available
    if eval_scores.category_scores or eval_scores.simulation_scores:
        final_metadata.update(
            {
                "score": eval_scores.avg_simulation_score,
                "avg_reward": eval_scores.avg_category_score,
            }
        )

    # Mark as final checkpoint and always upload to wandb if wandb is configured
    final_metadata["is_final"] = True
    final_metadata["upload_to_wandb"] = bool(wandb_run)  # Always upload final checkpoint

    checkpoint_manager.save_agent(
        agent_to_save,
        epoch,
        final_metadata,
        wandb_run=wandb_run,  # Upload final checkpoint if wandb is available
    )
    checkpoint_manager.save_trainer_state(optimizer, epoch, agent_step)

    cleanup_monitoring(memory_monitor, system_monitor)

    # Return stats info for exception handling at higher levels
    if stats_client and stats_tracker and stats_tracker.stats_run_id:
        return {"stats_run_id": stats_tracker.stats_run_id, "stats_client": stats_client, "logger": logger}
    return None
