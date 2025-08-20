import logging
import os
from collections import defaultdict
from typing import cast

import numpy as np
import torch
import torch.distributed
from heavyball import ForeachMuon
from torchrl.data import Composite

from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import PolicyAgent
from metta.agent.policy_store import PolicyStore
from metta.agent.world_model import WorldModel
from metta.app_backend.clients.stats_client import StatsClient
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.wandb.wandb_context import WandbRun
from metta.core.distributed import TorchDistributedConfig
from metta.core.monitoring import (
    cleanup_monitoring,
    setup_monitoring,
)
from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_service import evaluate_policy
from metta.mettagrid import MettaGridEnv, dtype_actions
from metta.rl.advantage import compute_advantage
from metta.rl.checkpoint_manager import CheckpointManager, maybe_establish_checkpoint
from metta.rl.dual_policy import _aggregate_dual_policy_stats, setup_dual_policy
from metta.rl.evaluate import evaluate_policy_remote, upload_replay_html
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses, get_loss_experience_spec, process_minibatch_update
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
from metta.rl.system_config import SystemConfig
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import TrainerConfig
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
from metta.utils.batch import calculate_batch_sizes, calculate_prioritized_sampling_params

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
    policy_store: PolicyStore,
    stats_client: StatsClient | None,
    torch_dist_cfg: TorchDistributedConfig,
) -> None:
    """Main training loop for Metta agents."""
    logger.info(f"run_dir = {run_dir}")
    is_master = torch_dist_cfg.is_master
    world_size = torch_dist_cfg.world_size

    # Log recent checkpoints for debugging
    checkpoints_dir = trainer_cfg.checkpoint.checkpoint_dir
    if os.path.exists(checkpoints_dir):
        files = sorted(os.listdir(checkpoints_dir))[-3:]
        if files:
            logger.info(f"Recent checkpoints: {', '.join(files)}")

    # Create timer, Losses, profiler, curriculum
    timer = Stopwatch(logger)
    timer.start()
    losses = Losses()
    torch_profiler = TorchProfiler(is_master, trainer_cfg.profiler, wandb_run, run_dir)
    curriculum = trainer_cfg.curriculum.make()

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

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        policy_store=policy_store,
        checkpoint_config=trainer_cfg.checkpoint,
        device=device,
        is_master=is_master,
        rank=torch_dist_cfg.rank,
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
        system_cfg=system_cfg,
        trainer_cfg=trainer_cfg,
        checkpoint=checkpoint,
        metta_grid_env=metta_grid_env,
    )

    # Don't proceed until all ranks have the policy
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    policy: PolicyAgent = latest_saved_policy_record.policy

    if trainer_cfg.compile:
        logger.info("Compiling policy")
        # torch.compile gives a CallbackFunctionType, but it preserves the interface of the original policy
        policy = cast(PolicyAgent, torch.compile(policy, mode=trainer_cfg.compile_mode))

    # Wrap in DDP if distributed
    if torch.distributed.is_initialized():
        if is_master:
            logger.info("Initializing DistributedDataParallel")
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

    # Create kickstarter
    kickstarter = Kickstarter(
        cfg=trainer_cfg.kickstart,
        device=device,
        policy_store=policy_store,
        metta_grid_env=metta_grid_env,
    )

    # Get the experience buffer specification from the policy
    policy_spec = policy.get_agent_experience_spec()
    act_space = vecenv.single_action_space
    act_dtype = torch.int32 if np.issubdtype(act_space.dtype, np.integer) else torch.float32
    loss_spec = get_loss_experience_spec(act_space.nvec, act_dtype)

    # Create experience buffer
    experience = Experience(
        total_agents=vecenv.num_agents,
        batch_size=trainer_cfg.batch_size,
        bptt_horizon=trainer_cfg.bptt_horizon,
        minibatch_size=trainer_cfg.minibatch_size,
        max_minibatch_size=trainer_cfg.minibatch_size,
        experience_spec=Composite({**dict(policy_spec.items()), **dict(loss_spec.items())}),
        device=device,
        cpu_offload=trainer_cfg.cpu_offload,
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

    # Instantiate world model used during rollout encoding/training
    world_model = WorldModel().to(device)
    world_model_optimizer = torch.optim.Adam(world_model.parameters(), lr=0.001)

    # World model pre-training phase
    world_model_pretraining_steps = 0
    if trainer_cfg.world_model_pretraining.enabled and agent_step == 0:
        logger.info(f"Starting world model pre-training for {trainer_cfg.world_model_pretraining.steps} steps")

    # Main training loop
    try:
        while agent_step < trainer_cfg.total_timesteps:
            steps_before = agent_step
            record_heartbeat()

            with torch_profiler:
                # ---- ROLLOUT PHASE ----
                with timer("_rollout"):
                    raw_infos = []
                    experience.reset_for_rollout()
                    total_steps = 0

                    policy.reset_memory()
                    buffer_step = experience.buffer[experience.ep_indices, experience.ep_lengths - 1]

                    # Dual-policy setup for the new epoch
                    npc_policy, npc_mask_per_env, agents_per_env = setup_dual_policy(
                        trainer_cfg.dual_policy,
                        policy_store,
                        metta_grid_env,
                        device,
                        epoch,
                        stats_tracker.rollout_stats,
                    )
                    if npc_policy:
                        npc_policy.reset_memory()

                    # If dual-policy is disabled, ensure env is not in dual-policy mode
                    if not trainer_cfg.dual_policy.enabled:
                        try:
                            metta_grid_env._dual_policy_enabled = False
                            metta_grid_env._dual_policy_agent_groups = [[], []]
                        except Exception:
                            pass
                        # Reset mask so all agents are treated as students downstream
                        npc_mask_per_env = None

                while not experience.ready_for_training:
                    # Get observation
                    o, r, d, t, info, training_env_id, _, num_steps = get_observation(vecenv, device, timer)
                    total_steps += num_steps

                    # Simple shape asserts for observations and rewards
                    assert o.ndim == 3 and o.shape[1:] == (
                        200,
                        3,
                    ), f"env_obs expected [B,200,3], got {tuple(o.shape)}"
                    assert r.ndim == 1 and r.shape[0] == o.shape[0], (
                        f"rewards expected [B], got {tuple(r.shape)} with B={o.shape[0]}"
                    )

                    # supervised training of the world model
                    # Cast obs to float for world model supervision to avoid dtype mismatch
                    reconstructed_obs = world_model(o)
                    world_model_loss = torch.nn.functional.mse_loss(reconstructed_obs, o.float())
                    world_model_optimizer.zero_grad()
                    world_model_loss.backward()
                    world_model_optimizer.step()

                    # World model pre-training phase logging and control
                    if (
                        trainer_cfg.world_model_pretraining.enabled
                        and world_model_pretraining_steps < trainer_cfg.world_model_pretraining.steps
                    ):
                        world_model_pretraining_steps += num_steps

                        # Log to wandb every 100 steps during pre-training
                        if wandb_run and is_master and world_model_pretraining_steps % 100 == 0:
                            wandb_run.log(
                                {
                                    "world_model/reconstruction_loss": world_model_loss.item(),
                                    "world_model/pretraining_steps": world_model_pretraining_steps,
                                }
                            )

                        # Log progress every 1000 steps during pre-training
                        if (
                            world_model_pretraining_steps % 1000 == 0
                            or world_model_pretraining_steps >= trainer_cfg.world_model_pretraining.steps
                        ):
                            logger.info(
                                f"World model pre-training: {world_model_pretraining_steps}/"
                                f"{trainer_cfg.world_model_pretraining.steps} steps, "
                                f"loss: {world_model_loss.item():.6f}"
                            )

                        # During pre-training, skip agent training and continue rollout for more world model data
                        if world_model_pretraining_steps < trainer_cfg.world_model_pretraining.steps:
                            # Record heartbeat during pre-training to prevent timeouts
                            record_heartbeat()

                            # Create TensorDict for agent to generate valid actions during pre-training
                            td = buffer_step[training_env_id].clone()
                            td["latent_obs"] = world_model.encode(o).detach()  # Use latent obs for action generation
                            # Remove raw observations to force latent path
                            if "env_obs" in td:
                                del td["env_obs"]

                            # Generate actions using the agent (random behavior during pre-training is fine)
                            with torch.no_grad():
                                policy(td)

                            send_observation(vecenv, td["actions"], dtype_actions, timer)
                            continue  # Skip agent inference and training during pre-training

                        else:
                            logger.info("World model pre-training completed, starting agent training")

                    # compute latent encoding (detached); keep raw obs for env_obs
                    with torch.no_grad():
                        o = world_model.encode(o).detach()

                    td = buffer_step[training_env_id].clone()
                    td["latent_obs"] = o  # Use latent_obs key for encoded observations
                    # Remove the original token observations to force latent path
                    if "env_obs" in td:
                        del td["env_obs"]
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
                    with torch.no_grad():
                        # Default: student policy acts for all agents
                        policy(td)

                        # Create student agent mask (inverse of NPC mask)
                        if npc_mask_per_env is not None and agents_per_env > 0:
                            student_mask = ~npc_mask_per_env
                            # Expand mask to match batch size
                            if td["actions"].ndim == 2:
                                # Shape like [B*agents_per_env, action_components]
                                total = td["actions"].shape[0]
                                if agents_per_env > 0 and total % agents_per_env == 0:
                                    repeats = total // agents_per_env
                                    student_mask_flat = student_mask.repeat(repeats)
                                    td["is_student_agent"] = student_mask_flat.float()
                                else:
                                    # Default to all students if can't determine structure
                                    td["is_student_agent"] = torch.ones(total, device=device, dtype=torch.float32)
                            else:
                                # For other shapes, use per-env mask
                                td["is_student_agent"] = student_mask.float()
                        else:
                            # No NPCs, all agents are students
                            td["is_student_agent"] = torch.ones(td.batch_size[0], device=device, dtype=torch.float32)

                        # If dual-policy is enabled and npc_policy is available, overwrite NPC agents' actions
                        if (
                            trainer_cfg.dual_policy.enabled
                            and npc_policy is not None
                            and npc_mask_per_env is not None
                            and npc_mask_per_env.any().item()
                        ):
                            td_npc = td.clone()
                            npc_policy(td_npc)
                            actions = td["actions"].clone()
                            # Merge NPC actions depending on action tensor shape
                            if actions.ndim >= 3:
                                # Shape like [B, num_agents, action_components]
                                actions[..., npc_mask_per_env, :] = td_npc["actions"][..., npc_mask_per_env, :]
                            elif actions.ndim == 2:
                                # Shape like [B*agents_per_env, action_components]
                                total = actions.shape[0]
                                if agents_per_env > 0:
                                    if total % agents_per_env == 0:
                                        repeats = total // agents_per_env
                                        npc_mask_flat = npc_mask_per_env.repeat(repeats)
                                        # Validate shapes before indexing
                                        if npc_mask_flat.shape[0] == total and td_npc["actions"].shape[0] == total:
                                            actions[npc_mask_flat, :] = td_npc["actions"][npc_mask_flat, :]
                                        else:
                                            logger.warning(
                                                "Skipping NPC action assignment due to shape mismatch: "
                                                f"mask={npc_mask_flat.shape[0]}, actions={total}, "
                                                f"npc_actions={td_npc['actions'].shape[0]}"
                                            )
                                    else:
                                        logger.warning(
                                            "Skipping NPC action assignment: total agents "
                                            f"{total} not divisible by agents_per_env {agents_per_env}"
                                        )
                                else:
                                    logger.warning("Skipping NPC action assignment: agents_per_env <= 0")
                            else:
                                # Unsupported shape; skip merge
                                pass
                            td["actions"] = actions

                    # Store experience
                    experience.store(
                        data_td=td,
                        env_id=training_env_id,
                    )

                    # Send observation
                    send_observation(vecenv, td["actions"], dtype_actions, timer)

                    if info:
                        raw_infos.extend(info)

                agent_step += total_steps * world_size
            accumulate_rollout_stats(raw_infos, stats_tracker.rollout_stats)

            # Aggregate dual_policy stats across all distributed nodes
            if trainer_cfg.dual_policy.enabled and torch.distributed.is_initialized():
                _aggregate_dual_policy_stats(stats_tracker.rollout_stats, device)

            # ---- TRAINING PHASE ----
            with timer("_train"):
                # Inline PPO training
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
                advantages = torch.zeros(experience.buffer["values"].shape, device=device)
                initial_importance_sampling_ratio = torch.ones_like(experience.buffer["values"])

                advantages = compute_advantage(
                    experience.buffer["values"],
                    experience.buffer["rewards"],
                    experience.buffer["dones"],
                    initial_importance_sampling_ratio,
                    advantages,
                    trainer_cfg.ppo.gamma,
                    trainer_cfg.ppo.gae_lambda,
                    trainer_cfg.vtrace.vtrace_rho_clip,
                    trainer_cfg.vtrace.vtrace_c_clip,
                    device,
                )

                # Train for multiple epochs
                minibatch_idx = 0
                epochs_trained = 0
                policy_spec = policy.get_agent_experience_spec()

                for _update_epoch in range(trainer_cfg.update_epochs):
                    for _ in range(experience.num_minibatches):
                        policy.reset_memory()
                        # Sample minibatch
                        minibatch, indices, prio_weights = experience.sample_minibatch(
                            advantages=advantages,
                            prio_alpha=trainer_cfg.prioritized_experience_replay.prio_alpha,
                            prio_beta=anneal_beta,
                        )

                        policy_td = minibatch.select(*policy_spec.keys(include_nested=True))

                        # Process minibatch
                        loss = process_minibatch_update(
                            policy=policy,
                            experience=experience,
                            minibatch=minibatch,
                            policy_td=policy_td,
                            indices=indices,
                            prio_weights=prio_weights,
                            trainer_cfg=trainer_cfg,
                            kickstarter=kickstarter,
                            agent_step=agent_step,
                            losses=losses,
                            device=device,
                        )

                        # Optimizer step
                        optimizer.zero_grad()

                        # This also serves as a barrier for all ranks
                        loss.backward()

                        if (minibatch_idx + 1) % experience.accumulate_minibatches == 0:
                            torch.nn.utils.clip_grad_norm_(policy.parameters(), trainer_cfg.ppo.max_grad_norm)
                            optimizer.step()

                            # Optional weight clipping
                            policy.clip_weights()

                            if device.type == "cuda":
                                torch.cuda.synchronize()

                        minibatch_idx += 1
                    epochs_trained += 1

                    # Early exit if KL divergence is too high
                    if trainer_cfg.ppo.target_kl is not None:
                        average_approx_kl = losses.approx_kl_sum / losses.minibatches_processed
                        if average_approx_kl > trainer_cfg.ppo.target_kl:
                            break

                # Calculate explained variance
                y_pred = experience.buffer["values"].flatten()
                y_true = advantages.flatten() + experience.buffer["values"].flatten()
                var_y = y_true.var()
                losses.explained_variance = (1 - (y_true - y_pred).var() / var_y).item() if var_y > 0 else 0.0
            epoch += epochs_trained

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
                        losses=losses,
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
                        kickstarter=kickstarter,
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
                kickstarter=kickstarter,
                wandb_run=wandb_run,
            )
            if checkpoint_result:
                # TODO: wandb_policy_name should come directly from last_saved_policy_record
                latest_saved_policy_record, wandb_policy_name = checkpoint_result

            if trainer_cfg.evaluation and should_run(epoch, trainer_cfg.evaluation.evaluate_interval):
                if latest_saved_policy_record:
                    if stats_client and stats_tracker.stats_run_id:
                        stats_tracker.stats_epoch_id = stats_client.create_epoch(
                            run_id=stats_tracker.stats_run_id,
                            start_training_epoch=stats_tracker.stats_epoch_start,
                            end_training_epoch=epoch,
                        ).id

                    sims = [
                        curriculum.get_task().get_env_cfg().to_sim(f"train_task_{i}")
                        for i in range(trainer_cfg.evaluation.num_training_tasks)
                    ]
                    sims.extend(trainer_cfg.evaluation.simulations)

                    evaluate_local = trainer_cfg.evaluation.evaluate_local
                    if trainer_cfg.evaluation.evaluate_remote:
                        try:
                            evaluate_policy_remote(
                                policy_record=latest_saved_policy_record,
                                simulations=sims,
                                stats_epoch_id=stats_tracker.stats_epoch_id,
                                wandb_policy_name=wandb_policy_name,
                                stats_client=stats_client,
                                wandb_run=wandb_run,
                                trainer_cfg=trainer_cfg,
                            )
                        except Exception as e:
                            logger.error(f"Failed to evaluate policy remotely: {e}", exc_info=True)
                            logger.error("Falling back to local evaluation")
                            evaluate_local = True
                    if evaluate_local:
                        evaluation_results = evaluate_policy(
                            policy_record=latest_saved_policy_record,
                            simulations=sims,
                            device=device,
                            vectorization=system_cfg.vectorization,
                            replay_dir=trainer_cfg.evaluation.replay_dir,
                            stats_epoch_id=stats_tracker.stats_epoch_id,
                            wandb_policy_name=wandb_policy_name,
                            policy_store=policy_store,
                            stats_client=stats_client,
                            logger=logger,
                        )
                        logger.info("Simulation complete")
                        eval_scores = evaluation_results.scores
                        category_scores = list(eval_scores.category_scores.values())
                        if category_scores and latest_saved_policy_record:
                            latest_saved_policy_record.metadata["score"] = float(np.mean(category_scores))
                        if wandb_run is not None and evaluation_results.replay_urls:
                            upload_replay_html(
                                replay_urls=evaluation_results.replay_urls,
                                agent_step=agent_step,
                                epoch=epoch,
                                wandb_run=wandb_run,
                                metric_prefix="training_eval",
                                step_metric_key="metric/epoch",
                                epoch_metric_key="metric/epoch",
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
                    wandb_run.config.update(
                        {"trainer.total_timesteps": trainer_cfg.total_timesteps}, allow_val_change=True
                    )
                    break

        # All ranks wait until training is complete before closing vecenv
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    except Exception as e:
        _update_training_status_on_failure(stats_client, stats_tracker.stats_run_id, logger)
        raise e

    vecenv.close()

    if not is_master:
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
        kickstarter=kickstarter,
        force=True,
        wandb_run=wandb_run,
    )

    cleanup_monitoring(memory_monitor, system_monitor)

    # Return stats info for exception handling at higher levels
    if stats_client and stats_tracker and hasattr(stats_tracker, "stats_run_id"):
        return {"stats_run_id": stats_tracker.stats_run_id, "stats_client": stats_client, "logger": logger}
    return None
