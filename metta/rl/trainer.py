import importlib
import logging
import os
from collections import defaultdict
from typing import cast

import torch
import torch.distributed
from heavyball import ForeachMuon
from omegaconf import DictConfig, OmegaConf
from torchrl.data import Composite

from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_store import PolicyStore
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.wandb.wandb_context import WandbRun
from metta.core.distributed import setup_distributed_vars
from metta.core.monitoring import (
    cleanup_monitoring,
    setup_monitoring,
)
from metta.eval.eval_request_config import EvalRewardSummary
from metta.mettagrid import MettaGridEnv, dtype_actions
from metta.mettagrid.curriculum.util import curriculum_from_config_path
from metta.rl.checkpoint_manager import CheckpointManager, maybe_establish_checkpoint
from metta.rl.env_config import EnvConfig
from metta.rl.evaluate import evaluate_policy, evaluate_policy_remote
from metta.rl.experience import Experience
from metta.rl.loss.base_loss import BaseLoss
from metta.rl.loss.loss_tracker import LossTracker
from metta.rl.optimization import (
    compute_gradient_stats,
)
from metta.rl.policy_management import (
    initialize_policy_for_environment,
    wrap_agent_distributed,
)
from metta.rl.rollout import get_observation, send_observation
from metta.rl.stats import (
    StatsTracker,
    accumulate_rollout_stats,
    process_stats,
)
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
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
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.utils.batch import calculate_batch_sizes

try:
    from pufferlib import _C  # noqa: F401 - Required for torch.ops.pufferlib  # type: ignore[reportUnusedImport]
except ImportError:
    raise ImportError(
        "Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, "
        "try installing with --no-build-isolation"
    ) from None

torch.set_float32_matmul_precision("high")

# Get rank for logger name
_rank = int(os.environ.get("RANK", 0))
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
logger = logging.getLogger(f"trainer-{_rank}-{_local_rank}")


def train(
    run_dir: str,
    run: str,
    env_cfg: EnvConfig,
    agent_cfg: DictConfig,
    device: torch.device,
    trainer_cfg: TrainerConfig,
    wandb_run: WandbRun | None,
    policy_store: PolicyStore,
    sim_suite_config: SimulationSuiteConfig,
    stats_client: StatsClient | None,
) -> None:
    """Main training loop for Metta agents."""
    torch.autograd.set_detect_anomaly(True)
    logger.info(f"run_dir = {run_dir}")

    # Log recent checkpoints for debugging
    checkpoints_dir = trainer_cfg.checkpoint.checkpoint_dir
    if os.path.exists(checkpoints_dir):
        files = sorted(os.listdir(checkpoints_dir))[-3:]
        if files:
            logger.info(f"Recent checkpoints: {', '.join(files)}")

    # Set up distributed
    is_master, world_size, rank = setup_distributed_vars()

    # Create timer, LossTracker, profiler, curriculum
    timer = Stopwatch(logger)
    timer.start()
    loss_tracker = LossTracker()
    torch_profiler = TorchProfiler(is_master, trainer_cfg.profiler, wandb_run, run_dir)
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
        env_cfg.vectorization,
        num_envs=num_envs,
        batch_size=batch_size,
        num_workers=trainer_cfg.num_workers,
        zero_copy=trainer_cfg.zero_copy,
        is_training=True,
    )

    vecenv.async_reset(env_cfg.seed + rank)

    metta_grid_env: MettaGridEnv = vecenv.driver_env  # type: ignore[attr-defined]

    # Initialize state containers
    eval_scores = EvalRewardSummary()  # Initialize eval_scores with empty summary

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        policy_store=policy_store,
        checkpoint_config=trainer_cfg.checkpoint,
        device=device,
        is_master=is_master,
        rank=rank,
        run_name=run,
    )

    # Load checkpoint if it exists
    checkpoint = TrainerCheckpoint.load(run_dir)
    agent_step = checkpoint.agent_step if checkpoint else 0
    epoch = checkpoint.epoch if checkpoint else 0

    if checkpoint:
        logger.info(f"Restored from checkpoint at {agent_step} steps")
        if checkpoint.stopwatch_state is not None:
            timer.load_state(checkpoint.stopwatch_state, resume_running=True)

    # Load or initialize policy with distributed coordination
    initial_policy_record = latest_saved_policy_record = checkpoint_manager.load_or_create_policy(
        agent_cfg=agent_cfg,
        env_cfg=env_cfg,
        trainer_cfg=trainer_cfg,
        checkpoint=checkpoint,
        metta_grid_env=metta_grid_env,
    )

    # Don't proceed until all ranks have the policy
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    policy: PolicyAgent = latest_saved_policy_record.policy
    policy_cfg = policy.get_cfg()

    if trainer_cfg.compile:
        logger.info("Compiling policy")
        # torch.compile gives a CallbackFunctionType, but it preserves the interface of the original policy
        policy = cast(PolicyAgent, torch.compile(policy, mode=trainer_cfg.compile_mode))

    # Wrap in DDP if distributed
    if torch.distributed.is_initialized():
        logger.info(f"Initializing DistributedDataParallel on device {device}")
        torch.distributed.barrier()
        policy = wrap_agent_distributed(policy, device)
        torch.distributed.barrier()

    # Initialize policy to environment after distributed wrapping
    # This must happen after wrapping to ensure all ranks do it at the same time
    initialize_policy_for_environment(
        policy_record=latest_saved_policy_record,
        metta_grid_env=metta_grid_env,
        device=device,
        restore_feature_mapping=True,
    )

    # Create optimizer
    optimizer_type = trainer_cfg.optimizer.type
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=policy_cfg.optimizer.learning_rate,
            betas=(policy_cfg.optimizer.beta1, policy_cfg.optimizer.beta2),
            eps=policy_cfg.optimizer.eps,
            weight_decay=policy_cfg.optimizer.weight_decay,
        )
    elif optimizer_type == "muon":
        # ForeachMuon expects int for weight_decay
        optimizer = ForeachMuon(
            policy.parameters(),
            lr=policy_cfg.optimizer.learning_rate,
            betas=(policy_cfg.optimizer.beta1, policy_cfg.optimizer.beta2),
            eps=policy_cfg.optimizer.eps,
            weight_decay=int(policy_cfg.optimizer.weight_decay),
        )
    else:
        raise ValueError(f"Optimizer type must be 'adam' or 'muon', got {optimizer_type}")

    # Instantiate configured losses dynamically by class name
    loss_instances: dict[str, BaseLoss] = {}
    for loss_instance_name, loss_config in policy_cfg.losses.items():
        module_path, class_name = loss_config["path"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        loss_cls = getattr(module, class_name)
        loss_instances[loss_instance_name] = loss_cls(
            policy,
            trainer_cfg,
            vecenv,
            device,
            loss_tracker,
            policy_store,
            loss_instance_name,
        )
    loss_tracker.configure_from_losses(list(loss_instances.values()))

    # Get the experience buffer specification from the policy
    policy_spec = policy.get_agent_experience_spec()
    # Merge experience specs required by all losses
    merged_spec_dict: dict = dict(policy_spec.items())
    for inst in loss_instances.values():
        spec = inst.get_experience_spec()
        merged_spec_dict.update(dict(spec.items()))

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
    policy.attach_replay_buffer(experience)

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

    if is_master:
        logger.info(f"Training on {device}")
    wandb_policy_name: str | None = None

    # Main training loop
    trainer_state = TrainerState(
        agent_step=agent_step,
        epoch=0,
        update_epoch=0,
        mb_idx=0,
        num_mbs=0,
        optimizer=optimizer,
    )

    while agent_step < trainer_cfg.total_timesteps:
        steps_before = agent_step
        trainer_state.agent_step = agent_step
        trainer_state.epoch = epoch
        policy_losses = policy.get_cfg().losses
        shared_loss_mb_data = experience.give_me_empty_md_td()
        policy.on_new_training_run()
        for _loss_name in loss_instances.keys():
            shared_loss_mb_data[_loss_name] = experience.give_me_empty_md_td()
        record_heartbeat()

        with torch_profiler:
            # ---- ROLLOUT PHASE ----
            with timer("_rollout"):
                raw_infos = []
                total_steps = 0
                experience.reset_for_rollout()
                for _loss_name in list(policy_losses):
                    loss_instances[_loss_name].on_rollout_start()

                buffer_step = experience.buffer[experience.ep_indices, experience.ep_lengths - 1]
                buffer_step = buffer_step.select(*policy_spec.keys())

                while not experience.ready_for_training:
                    # Get observation
                    o, r, d, t, info, training_env_id, _, num_steps = get_observation(vecenv, device, timer)
                    total_steps += num_steps

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

                    # Inference
                    # note that each loss will modify the td, the same one that is passed to other losses. We want this
                    # because this allows other parts of the network to only run what's needed on these obs, efficiently
                    # reusing hiddens within the network. Other losses should clear fields and/or clone as necessary.
                    for _lname in list(policy_losses):
                        loss_obj = loss_instances[_lname]
                        loss_obj.rollout(td, trainer_state)

                    # Store experience
                    experience.store(data_td=td, env_id=training_env_id)

                    # Send observation
                    send_observation(vecenv, td["actions"], dtype_actions, timer)

                    if info:
                        raw_infos.extend(info)

                agent_step += total_steps * world_size
            accumulate_rollout_stats(raw_infos, stats_tracker.rollout_stats)

            # ---- TRAINING PHASE ----
            with timer("_train"):
                loss_tracker.zero()
                shared_loss_mb_data.zero_()

                # Train for multiple epochs
                minibatch_idx = 0
                epochs_trained = 0

                for _update_epoch in range(trainer_cfg.update_epochs):
                    trainer_state.update_epoch = _update_epoch
                    for mb_idx in range(experience.num_minibatches):
                        trainer_state.mb_idx = mb_idx
                        trainer_state.early_stop_update_epoch = False
                        total_loss = torch.tensor(0.0, device=device)
                        for _lname in list(policy_losses):
                            loss_obj = loss_instances[_lname]
                            loss_val, shared_loss_mb_data = loss_obj.train(shared_loss_mb_data, trainer_state)
                            total_loss = total_loss + loss_val

                        # Count this minibatch once for averaging metrics
                        loss_tracker.minibatches_processed += 1

                        if trainer_state.early_stop_update_epoch:
                            break

                        # Optimizer step
                        optimizer.zero_grad()

                        # This also serves as a barrier for all ranks
                        total_loss.backward()

                        if (minibatch_idx + 1) % experience.accumulate_minibatches == 0:
                            torch.nn.utils.clip_grad_norm_(policy.parameters(), policy_cfg.losses.PPO.max_grad_norm)
                            optimizer.step()

                            if device.type == "cuda":
                                torch.cuda.synchronize()

                        for _lname in list(policy_losses):
                            loss_obj = loss_instances[_lname]
                            loss_obj.on_mb_end()

                        minibatch_idx += 1
                    epochs_trained += 1

                for _lname in list(policy_losses):
                    loss_obj = loss_instances[_lname]
                    loss_obj.on_train_phase_end()

            epoch += epochs_trained
            trainer_state.epoch = epoch
            trainer_state.agent_step = agent_step  # update agent_step count state not in between rollout and train

        # Safe to proceed to next rollout phase only once all ranks have completed training
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if not is_master:
            # Only master needs to do bookkeeping
            continue

        torch_profiler.on_epoch_end(epoch)

        with timer("_process_stats"):
            if wandb_run:
                process_stats(
                    agent_cfg=agent_cfg,
                    stats=stats_tracker.rollout_stats,
                    losses=loss_tracker,
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
                    latest_saved_policy_record=latest_saved_policy_record,
                    optimizer=optimizer,
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
        checkpoint_result = maybe_establish_checkpoint(
            checkpoint_manager=checkpoint_manager,
            epoch=epoch,
            policy=policy,
            agent_step=agent_step,
            eval_scores=eval_scores,
            timer=timer,
            initial_policy_record=initial_policy_record,
            optimizer=optimizer,
            run_dir=run_dir,
            wandb_run=wandb_run,
        )
        if checkpoint_result:
            # TODO: wandb_policy_name should come directly from last_saved_policy_record
            latest_saved_policy_record, wandb_policy_name = checkpoint_result

        if should_run(epoch, trainer_cfg.simulation.evaluate_interval):
            if latest_saved_policy_record:
                if stats_client and stats_tracker.stats_run_id:
                    stats_tracker.stats_epoch_id = stats_client.create_epoch(
                        run_id=stats_tracker.stats_run_id,
                        start_training_epoch=stats_tracker.stats_epoch_start,
                        end_training_epoch=epoch,
                    ).id

                # Create extended simulation suite that includes the training task
                # Deep merge trainer env_overrides with sim_suite_config env_overrides
                merged_env_overrides: dict = OmegaConf.to_container(  # type: ignore
                    OmegaConf.merge(sim_suite_config.env_overrides, trainer_cfg.env_overrides)
                )
                extended_suite_config = SimulationSuiteConfig(
                    name=sim_suite_config.name,
                    simulations=dict(sim_suite_config.simulations),
                    env_overrides=merged_env_overrides,
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

                if trainer_cfg.simulation.evaluate_remote:
                    evaluate_policy_remote(
                        policy_record=latest_saved_policy_record,
                        sim_suite_config=extended_suite_config,
                        stats_epoch_id=stats_tracker.stats_epoch_id,
                        wandb_policy_name=wandb_policy_name,
                        stats_client=stats_client,
                        wandb_run=wandb_run,
                        trainer_cfg=trainer_cfg,
                    )
                if trainer_cfg.simulation.evaluate_local:
                    eval_scores = evaluate_policy(
                        policy_record=latest_saved_policy_record,
                        sim_suite_config=extended_suite_config,
                        device=device,
                        vectorization=env_cfg.vectorization,
                        replay_dir=trainer_cfg.simulation.replay_dir,
                        stats_epoch_id=stats_tracker.stats_epoch_id,
                        wandb_policy_name=wandb_policy_name,
                        policy_store=policy_store,
                        stats_client=stats_client,
                        wandb_run=wandb_run,
                        trainer_cfg=trainer_cfg,
                        agent_step=agent_step,
                        epoch=epoch,
                    )

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
                wandb_run.config.update({"trainer.total_timesteps": trainer_cfg.total_timesteps}, allow_val_change=True)
                break

    # All ranks wait until training is complete before closing vecenv
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    vecenv.close()

    if not is_master:
        return

    logger.info("Training complete!")
    timing_summary = timer.get_all_summaries()
    for name, summary in timing_summary.items():
        logger.info(f"  {name}: {timer.format_time(summary['total_elapsed'])}")

    maybe_establish_checkpoint(
        checkpoint_manager=checkpoint_manager,
        epoch=epoch,
        policy=policy,
        agent_step=agent_step,
        eval_scores=eval_scores,
        timer=timer,
        initial_policy_record=initial_policy_record,
        optimizer=optimizer,
        run_dir=run_dir,
        force=True,
        wandb_run=wandb_run,
    )

    cleanup_monitoring(memory_monitor, system_monitor)
