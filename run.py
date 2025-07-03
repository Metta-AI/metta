#!/usr/bin/env -S uv run
"""Example of using Metta as a library without Hydra configuration.

This script demonstrates how to use Metta's training components directly
without going through the Hydra configuration system.

Single GPU training:
    python run.py

Multi-GPU training (on a single node):
    torchrun --nproc_per_node=4 run.py

    # Or with specific GPUs:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 run.py

Multi-node training:
    # Node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<master_ip> --master_port=29500 run.py

    # Node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=<master_ip> --master_port=29500 run.py

The script automatically detects distributed training when launched with torchrun
and handles multi-GPU coordination, including:
- Distributed data parallel (DDP) for the agent
- Checkpoint synchronization across ranks
- Master-only evaluation and replay generation
"""

import logging
import os
import time

import torch
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.api import (
    Agent,
    Environment,
    Optimizer,
    accumulate_rollout_stats,
    calculate_anneal_beta,
    cleanup_distributed,
    compute_advantage,
    create_evaluation_config_suite,
    create_replay_config,
    load_checkpoint,
    perform_rollout_step,
    process_minibatch_update,
    save_checkpoint,
    save_experiment_config,
    setup_device_and_distributed,
    setup_run_directories,
    wrap_agent_distributed,
)
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.fs import wait_for_file
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.eval.eval_stats_db import EvalStatsDB
from metta.mettagrid import mettagrid_c  # noqa: F401
from metta.rl.experience import Experience
from metta.rl.functions import (
    calculate_explained_variance,
    compute_gradient_stats,
    get_lstm_config,
    maybe_update_l2_weights,
    setup_distributed_vars,
    should_run_on_interval,
)
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_config import (
    CheckpointConfig,
    OptimizerConfig,
    PPOConfig,
    SimulationConfig,
    TorchProfilerConfig,
    TrainerConfig,
)
from metta.sim.simulation import Simulation
from metta.sim.simulation_suite import SimulationSuite

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set up directories and device first
dirs = setup_run_directories()
logger.info(f"Run directories set up: {dirs.run_dir}")

device = setup_device_and_distributed("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device setup complete: {device}")

# Get distributed info after initialization
is_master, world_size, rank = setup_distributed_vars()
logger.info(f"Distributed mode: rank {rank}/{world_size}, is_master={is_master}")

# Configuration
trainer_config = TrainerConfig(
    num_workers=4,
    total_timesteps=10_000_000,
    batch_size=16384,
    minibatch_size=512,
    curriculum="/env/mettagrid/curriculum/navigation/bucketed",
    ppo=PPOConfig(
        clip_coef=0.1,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
    ),
    optimizer=OptimizerConfig(
        type="adam",
        learning_rate=3e-4,
    ),
    checkpoint=CheckpointConfig(
        checkpoint_dir=dirs.checkpoint_dir,
        checkpoint_interval=10,
        wandb_checkpoint_interval=0,
    ),
    simulation=SimulationConfig(
        evaluate_interval=100,
        replay_interval=200,
        replay_dir=dirs.replay_dir,
    ),
    profiler=TorchProfilerConfig(
        interval_epochs=0,
        profile_dir=os.path.join(dirs.run_dir, "torch_traces"),
    ),
    grad_mean_variance_interval=50,
)

# Adjust batch sizes for distributed training
if torch.distributed.is_initialized() and trainer_config.scale_batches_by_world_size:
    trainer_config.batch_size = trainer_config.batch_size // world_size

# Save config only on master
if is_master:
    save_experiment_config(dirs, device, trainer_config)

# Create environment
logger.info("Creating environment...")
env = Environment(
    curriculum_path="/env/mettagrid/curriculum/navigation/bucketed",
    num_agents=4,
    width=32,
    height=32,
    device=str(device),
    num_envs=64,
    num_workers=trainer_config.num_workers,
    batch_size=64,
    async_factor=trainer_config.async_factor,
    zero_copy=trainer_config.zero_copy,
    is_training=True,
)
metta_grid_env = env.driver_env  # type: ignore - vecenv attribute

# Create agent
logger.info("Creating agent...")
agent = Agent(env, device=str(device))
hidden_size, num_lstm_layers = get_lstm_config(agent)

# Create policy store
policy_store = PolicyStore(
    DictConfig(
        {
            "device": str(device),
            "policy_cache_size": 10,
            "run": dirs.run_name,
            "run_dir": dirs.run_dir,
            "trainer": trainer_config.model_dump(),
        }
    ),
    wandb_run=None,
)

# Load checkpoint or create initial policy with distributed coordination
checkpoint_path = trainer_config.checkpoint.checkpoint_dir
checkpoint = load_checkpoint(checkpoint_path, None, None, policy_store, device)
agent_step, epoch, loaded_policy_path = checkpoint

# Handle initial policy creation/loading like MettaTrainer
if loaded_policy_path is None:
    # No existing checkpoint - need to coordinate initial policy creation
    logger.info("No existing checkpoint found, coordinating initial policy creation")

    if torch.distributed.is_initialized():
        if is_master:
            # Master creates and saves initial policy
            logger.info("Master: Creating and saving initial policy")
            saved_policy = save_checkpoint(
                epoch=0,
                agent_step=0,
                agent=agent,
                optimizer=None,
                policy_store=policy_store,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=1,  # Force save
                stats={},
                force_save=True,
            )
            logger.info("Master: Initial policy saved")

            # Master waits at barrier after saving
            torch.distributed.barrier()
        else:
            # Non-master ranks wait at barrier first
            logger.info(f"Rank {rank}: Waiting at barrier for master to create policy")
            torch.distributed.barrier()

            # Then load the policy master created
            default_policy_path = os.path.join(checkpoint_path, policy_store.make_model_name(0))
            logger.info(f"Rank {rank}: Loading policy from {default_policy_path}")

            def log_progress(elapsed: float, status: str) -> None:
                if status == "waiting" and int(elapsed) % 10 == 0 and elapsed > 0:
                    logger.info(f"Rank {rank}: Still waiting for policy file... ({elapsed:.0f}s elapsed)")
                elif status == "found":
                    logger.info(f"Rank {rank}: Policy file found, waiting for write to complete...")
                elif status == "stable":
                    logger.info(f"Rank {rank}: Policy file stable after {elapsed:.1f}s")

            if not wait_for_file(default_policy_path, timeout=300, progress_callback=log_progress):
                raise RuntimeError(f"Rank {rank}: Timeout waiting for policy at {default_policy_path}")

            # Load the policy
            policy_pr = policy_store.policy_record(default_policy_path)
            agent.load_state_dict(policy_pr.policy.state_dict())
            logger.info(f"Rank {rank}: Loaded initial policy")
    else:
        # Single GPU mode
        logger.info("Single GPU mode: Creating initial checkpoint")
        save_checkpoint(
            epoch=0,
            agent_step=0,
            agent=agent,
            optimizer=None,
            policy_store=policy_store,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=1,
            stats={},
            force_save=True,
        )
else:
    # Checkpoint exists - load it
    logger.info(f"Loading from existing checkpoint at step {agent_step}")
    # The load_checkpoint function already loaded the policy state into the agent

# Wrap agent in distributed after all ranks have the same policy
logger.info("Wrapping agent for distributed training...")
agent = wrap_agent_distributed(agent, device)

# Ensure all ranks have wrapped their agents before proceeding
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# Create optimizer
optimizer = Optimizer(
    optimizer_type=trainer_config.optimizer.type,
    policy=agent,
    learning_rate=trainer_config.optimizer.learning_rate,
    betas=(trainer_config.optimizer.beta1, trainer_config.optimizer.beta2),
    eps=trainer_config.optimizer.eps,
    weight_decay=trainer_config.optimizer.weight_decay,
    max_grad_norm=trainer_config.ppo.max_grad_norm,
)

# Load optimizer state from checkpoint if it exists
_, _, checkpoint_path_from_load = load_checkpoint(checkpoint_path, None, optimizer, policy_store, device)

# Create experience buffer
logger.info("Creating experience buffer...")
experience = Experience(
    total_agents=env.num_agents,  # type: ignore
    batch_size=trainer_config.batch_size,
    bptt_horizon=trainer_config.bptt_horizon,
    minibatch_size=trainer_config.minibatch_size,
    max_minibatch_size=trainer_config.minibatch_size,
    obs_space=env.single_observation_space,  # type: ignore
    atn_space=env.single_action_space,  # type: ignore
    device=device,
    hidden_size=hidden_size,
    cpu_offload=trainer_config.cpu_offload,
    num_lstm_layers=num_lstm_layers,
    agents_per_batch=getattr(env, "agents_per_batch", None),  # type: ignore
)

# Create kickstarter
logger.info("Creating kickstarter...")
kickstarter = Kickstarter(
    trainer_config.kickstart,
    str(device),
    policy_store,
    metta_grid_env.action_names,
    metta_grid_env.max_action_args,
)

# Create losses tracker
losses = Losses()

# Create timer
timer = Stopwatch(logger)
timer.start()

# Create learning rate scheduler
lr_scheduler = None
if hasattr(trainer_config, "lr_scheduler") and trainer_config.lr_scheduler.enabled:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optimizer, T_max=trainer_config.total_timesteps // trainer_config.batch_size
    )
    logger.info("Created learning rate scheduler")

# Memory and System Monitoring (master only)
if is_master:
    memory_monitor = MemoryMonitor()
    memory_monitor.add(experience, name="Experience", track_attributes=True)
    memory_monitor.add(agent, name="Agent", track_attributes=False)

    system_monitor = SystemMonitor(
        sampling_interval_sec=1.0,
        history_size=100,
        logger=logger,
        auto_start=True,
    )

# Evaluation configuration
evaluation_config = create_evaluation_config_suite()

# Training loop
saved_policy_path = None
logger.info(f"Starting training on {device}")
evaluation_scores = {}
epoch_start_time = time.time()
steps_at_epoch_start = agent_step
stats = {}

while agent_step < trainer_config.total_timesteps:
    steps_before = agent_step

    # ===== ROLLOUT PHASE =====
    rollout_start = time.time()
    raw_infos = []
    experience.reset_for_rollout()

    # Collect experience
    while not experience.ready_for_training:
        num_steps, info = perform_rollout_step(agent, env, experience, device, timer)
        agent_step += num_steps

        if info:
            raw_infos.extend(info)

    # Process rollout statistics
    accumulate_rollout_stats(raw_infos, stats)
    rollout_time = time.time() - rollout_start

    # ===== TRAINING PHASE =====
    train_start = time.time()
    losses.zero()
    experience.reset_importance_sampling_ratios()

    # Calculate prioritized replay parameters
    prio_cfg = trainer_config.prioritized_experience_replay
    anneal_beta = calculate_anneal_beta(
        epoch=epoch,
        total_timesteps=trainer_config.total_timesteps,
        batch_size=trainer_config.batch_size,
        prio_alpha=prio_cfg.prio_alpha,
        prio_beta0=prio_cfg.prio_beta0,
    )

    advantages = torch.zeros(experience.values.shape, device=device)
    initial_importance_sampling_ratio = torch.ones_like(experience.values)

    advantages = compute_advantage(
        experience.values,
        experience.rewards,
        experience.dones,
        initial_importance_sampling_ratio,
        advantages,
        trainer_config.ppo.gamma,
        trainer_config.ppo.gae_lambda,
        trainer_config.vtrace.vtrace_rho_clip,
        trainer_config.vtrace.vtrace_c_clip,
        device,
    )

    # Train for multiple epochs
    total_minibatches = experience.num_minibatches * trainer_config.update_epochs
    minibatch_idx = 0

    for _update_epoch in range(trainer_config.update_epochs):
        for _ in range(experience.num_minibatches):
            # Sample minibatch
            minibatch = experience.sample_minibatch(
                advantages=advantages,
                prio_alpha=prio_cfg.prio_alpha,
                prio_beta=anneal_beta,
                minibatch_idx=minibatch_idx,
                total_minibatches=total_minibatches,
            )

            # Train on minibatch
            loss = process_minibatch_update(
                policy=agent,
                experience=experience,
                minibatch=minibatch,
                advantages=advantages,
                trainer_cfg=trainer_config,
                kickstarter=kickstarter,
                agent_step=agent_step,
                losses=losses,
                device=device,
            )

            optimizer.step(loss, epoch, experience.accumulate_minibatches)
            minibatch_idx += 1
        epoch += 1

        # Early exit if KL divergence is too high
        if trainer_config.ppo.target_kl is not None:
            average_approx_kl = losses.approx_kl_sum / losses.minibatches_processed
            if average_approx_kl > trainer_config.ppo.target_kl:
                break

    if minibatch_idx > 0:
        # CUDA synchronization
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    if lr_scheduler is not None:
        lr_scheduler.step()

    losses.explained_variance = calculate_explained_variance(experience.values, advantages)

    # Calculate performance metrics
    train_time = time.time() - train_start
    steps_calculated = agent_step - steps_before
    total_time = train_time + rollout_time
    steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

    # Account for distributed training
    steps_per_sec *= world_size

    train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
    rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0

    logger.info(f"Epoch {epoch} - {steps_per_sec:.0f} steps/sec ({train_pct:.0f}% train / {rollout_pct:.0f}% rollout)")

    # Record heartbeat periodically (master only)
    if should_run_on_interval(epoch, 10, is_master):
        record_heartbeat()

    # Update L2 weights if configured
    if hasattr(agent, "l2_init_weight_update_interval"):
        maybe_update_l2_weights(
            agent=agent,
            epoch=epoch,
            interval=getattr(agent, "l2_init_weight_update_interval", 0),
            is_master=is_master,
        )

    # Compute gradient statistics (master only)
    if trainer_config.grad_mean_variance_interval > 0 and epoch % trainer_config.grad_mean_variance_interval == 0:
        if is_master:
            grad_stats = compute_gradient_stats(agent)
            logger.info(
                f"Gradient stats - mean: {grad_stats.get('grad/mean', 0):.2e}, "
                f"variance: {grad_stats.get('grad/variance', 0):.2e}, "
                f"norm: {grad_stats.get('grad/norm', 0):.2e}"
            )

    # Log system monitoring stats (master only)
    if is_master and epoch % 10 == 0:
        system_stats = system_monitor.get_summary()
        logger.info(
            f"System stats - CPU: {system_stats.get('cpu_percent', 0):.1f}%, "
            f"Memory: {system_stats.get('memory_percent', 0):.1f}%, "
            f"Process Memory: {system_stats.get('process_memory_mb', 0):.1f}MB"
        )

        # Log GPU stats if available
        if "gpu_utilization_avg" in system_stats:
            logger.info(
                f"GPU stats - Utilization: {system_stats.get('gpu_utilization_avg', 0):.1f}%, "
                f"Memory: {system_stats.get('gpu_memory_percent_avg', 0):.1f}%"
            )

        # Log memory monitor stats
        memory_stats = memory_monitor.stats()
        if memory_stats:
            logger.info(f"Memory usage: {memory_stats}")

    # Save checkpoint periodically with distributed coordination
    if trainer_config.checkpoint.checkpoint_interval > 0 and epoch % trainer_config.checkpoint.checkpoint_interval == 0:
        if torch.distributed.is_initialized():
            # All ranks must participate in checkpoint saving
            if is_master:
                # Master saves the checkpoint
                saved_policy_path = save_checkpoint(
                    epoch=epoch,
                    agent_step=agent_step,
                    agent=agent,
                    optimizer=optimizer,
                    policy_store=policy_store,
                    checkpoint_path=checkpoint_path,
                    checkpoint_interval=trainer_config.checkpoint.checkpoint_interval,
                    stats=stats,
                    force_save=False,
                )
            # All ranks wait at barrier
            torch.distributed.barrier()
        else:
            # Single GPU mode
            saved_policy_path = save_checkpoint(
                epoch=epoch,
                agent_step=agent_step,
                agent=agent,
                optimizer=optimizer,
                policy_store=policy_store,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=trainer_config.checkpoint.checkpoint_interval,
                stats=stats,
                force_save=False,
            )

    # Policy evaluation (master only)
    if (
        is_master
        and trainer_config.simulation.evaluate_interval > 0
        and epoch % trainer_config.simulation.evaluate_interval == 0
        and saved_policy_path
    ):
        logger.info(f"Evaluating policy at epoch {epoch}")

        # Run evaluation suite
        sim_suite = SimulationSuite(
            config=evaluation_config,
            policy_pr=saved_policy_path,
            policy_store=policy_store,
            device=device,
            vectorization="serial",
            stats_dir=dirs.stats_dir,
            stats_client=None,
            stats_epoch_id=None,
            wandb_policy_name=None,
        )

        results = sim_suite.simulate()
        stats_db = EvalStatsDB.from_sim_stats_db(results.stats_db)
        logger.info("Evaluation complete")

        # Build evaluation metrics
        eval_scores = {}
        categories = set()
        for sim_name in evaluation_config.simulations.keys():
            categories.add(sim_name.split("/")[0])

        for category in categories:
            score = stats_db.get_average_metric_by_filter("reward", saved_policy_path, f"sim_name LIKE '%{category}%'")
            logger.info(f"{category} score: {score}")
            record_heartbeat()
            if score is not None:
                eval_scores[f"{category}/score"] = score

        # Get detailed per-simulation scores
        all_scores = stats_db.simulation_scores(saved_policy_path, "reward")
        for (_, sim_name, _), score in all_scores.items():
            category = sim_name.split("/")[0]
            sim_short_name = sim_name.split("/")[-1]
            eval_scores[f"{category}/{sim_short_name}"] = score

        evaluation_scores[epoch] = eval_scores
        stats_db.close()

    # Replay generation (master only)
    if (
        is_master
        and trainer_config.simulation.replay_interval > 0
        and epoch % trainer_config.simulation.replay_interval == 0
        and saved_policy_path
    ):
        logger.info(f"Generating replay at epoch {epoch}")

        # Generate replay on the bucketed curriculum environment
        replay_sim_config = create_replay_config("varied_terrain/balanced_medium")

        replay_simulator = Simulation(
            name=f"replay_{epoch}",
            config=replay_sim_config,
            policy_pr=saved_policy_path,
            policy_store=policy_store,
            device=device,
            vectorization="serial",
            replay_dir=dirs.replay_dir,
        )

        results = replay_simulator.simulate()

        # Get replay URLs from the database
        replay_urls = results.stats_db.get_replay_urls()
        if replay_urls:
            replay_url = replay_urls[0]
            player_url = f"https://metta-ai.github.io/metta/?replayUrl={replay_url}"
            logger.info(f"Replay available at: {player_url}")

        results.stats_db.close()

    # Clear stats for next iteration
    stats.clear()

# Training complete
total_elapsed = time.time() - epoch_start_time
logger.info("Training complete!")
logger.info(f"Total training time: {total_elapsed:.1f}s")
logger.info(f"Final epoch: {epoch}")
logger.info(f"Total steps: {agent_step}")

# Log final stats if available
if hasattr(losses, "stats"):
    losses_stats = losses.stats()
    logger.info(
        f"Final losses - "
        f"Policy: {losses_stats.get('policy_loss', 0):.4f}, "
        f"Value: {losses_stats.get('value_loss', 0):.4f}, "
        f"Entropy: {losses_stats.get('entropy', 0):.4f}, "
        f"Explained Variance: {losses_stats.get('explained_variance', 0):.3f}"
    )

# Log evaluation history
if evaluation_scores:
    logger.info("\nEvaluation History:")
    for eval_epoch, scores in sorted(evaluation_scores.items()):
        logger.info(f"  Epoch {eval_epoch}:")
        for env_name, score in scores.items():
            logger.info(f"    {env_name}: {score:.2f}")

# Log final system stats (master only)
if is_master:
    final_system_stats = system_monitor.get_summary()
    logger.info(f"\nFinal system stats: {final_system_stats}")
    system_monitor.stop()
    memory_monitor.clear()

# Save final checkpoint with distributed coordination
if torch.distributed.is_initialized():
    if is_master:
        # Master saves final checkpoint
        saved_policy_path = save_checkpoint(
            epoch=epoch,
            agent_step=agent_step,
            agent=agent,
            optimizer=optimizer,
            policy_store=policy_store,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=trainer_config.checkpoint.checkpoint_interval,
            stats=stats,
            force_save=True,
        )
    # All ranks wait at barrier
    torch.distributed.barrier()
else:
    # Single GPU mode
    saved_policy_path = save_checkpoint(
        epoch=epoch,
        agent_step=agent_step,
        agent=agent,
        optimizer=optimizer,
        policy_store=policy_store,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=trainer_config.checkpoint.checkpoint_interval,
        stats=stats,
        force_save=True,
    )

# Close environment
env.close()  # type: ignore

logger.info(f"\nTraining run complete! Run saved to: {dirs.run_dir}")

# Clean up distributed training if initialized
cleanup_distributed()
