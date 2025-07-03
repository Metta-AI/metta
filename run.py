#!/usr/bin/env -S uv run
"""Example of using Metta as a library without Hydra configuration."""

import logging
import os
import time

import torch

from metta.api import (
    Agent,
    Environment,
    Optimizer,
    accumulate_rollout_stats,
    calculate_anneal_beta,
    compute_advantage,
    create_policy_store,
    perform_rollout_step,
    process_minibatch_update,
    save_experiment_config,
    setup_run_directories,
)
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.eval.eval_stats_db import EvalStatsDB
from metta.mettagrid import mettagrid_c  # noqa: F401
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.experience import Experience
from metta.rl.functions import (
    calculate_explained_variance,
    cleanup_old_policies,
    compute_gradient_stats,
    get_lstm_config,
)
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import (
    CheckpointConfig,
    OptimizerConfig,
    PPOConfig,
    SimulationConfig,
    TorchProfilerConfig,
    TrainerConfig,
)
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.sim.simulation_suite import SimulationSuite

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dirs = setup_run_directories()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create trainer config with structured Pydantic classes
trainer_config = TrainerConfig(
    num_workers=4,
    total_timesteps=10_000_000,
    batch_size=16384,
    minibatch_size=512,
    # Use structured config classes instead of dictionaries
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
        checkpoint_interval=100,
        wandb_checkpoint_interval=0,  # Disabled for this example
    ),
    simulation=SimulationConfig(
        evaluate_interval=100,  # Evaluate every 100 epochs
        replay_interval=200,  # Generate replay every 200 epochs
        replay_dir=dirs.replay_dir,
    ),
    profiler=TorchProfilerConfig(
        interval_epochs=0,  # 0 disables profiling
        profile_dir="./profiles",
    ),
    grad_mean_variance_interval=50,  # Compute gradient stats every 50 epochs
)

# Create environment
env = Environment(
    num_agents=4,  # Simplified - just specify what we need
    width=32,
    height=32,
    device=str(device),
    num_envs=64,  # Number of parallel environments
    num_workers=trainer_config.num_workers,
    batch_size=64,  # Batch size per worker
    async_factor=trainer_config.async_factor,
    zero_copy=trainer_config.zero_copy,
    is_training=True,
)

# Create agent
agent = Agent(env, device=str(device))  # Uses default config

# Save configuration to run directory (like train.py does)
experiment_config = {
    "run": dirs.run_name,
    "run_dir": dirs.run_dir,
    "data_dir": os.path.dirname(dirs.run_dir),  # Get parent directory
    "device": str(device),
    "trainer": {
        "num_workers": trainer_config.num_workers,
        "total_timesteps": trainer_config.total_timesteps,
        "batch_size": trainer_config.batch_size,
        "minibatch_size": trainer_config.minibatch_size,
        "checkpoint_dir": dirs.checkpoint_dir,
        "optimizer": trainer_config.optimizer.model_dump(),
        "ppo": trainer_config.ppo.model_dump(),
        "checkpoint": trainer_config.checkpoint.model_dump(),
        "simulation": trainer_config.simulation.model_dump(),
        "profiler": trainer_config.profiler.model_dump(),
    },
}
save_experiment_config(dirs.run_dir, experiment_config)

# Create policy store for checkpointing
policy_store = create_policy_store(
    checkpoint_dir=dirs.checkpoint_dir,
    device=str(device),
    wandb_run=None,  # No wandb in this example
    policy_cache_size=10,
)

# Create optimizer wrapper from api.py
optimizer = Optimizer(
    optimizer_type=trainer_config.optimizer.type,
    policy=agent,
    learning_rate=trainer_config.optimizer.learning_rate,
    betas=(trainer_config.optimizer.beta1, trainer_config.optimizer.beta2),
    eps=trainer_config.optimizer.eps,
    weight_decay=trainer_config.optimizer.weight_decay,
    max_grad_norm=trainer_config.ppo.max_grad_norm,
)

# Get environment info from the vecenv (not Environment class)
# Note: env is actually the vecenv returned by Environment.__new__
metta_grid_env = env.driver_env  # type: ignore - vecenv attribute
assert isinstance(metta_grid_env, MettaGridEnv)

# Create experience buffer
logger.info("Creating experience buffer...")
obs_space = env.single_observation_space  # type: ignore - vecenv attribute
atn_space = env.single_action_space  # type: ignore - vecenv attribute
total_agents = env.num_agents  # type: ignore - vecenv attribute
hidden_size, num_lstm_layers = get_lstm_config(agent)

experience = Experience(
    total_agents=total_agents,
    batch_size=trainer_config.batch_size,
    bptt_horizon=trainer_config.bptt_horizon,
    minibatch_size=trainer_config.minibatch_size,
    max_minibatch_size=trainer_config.minibatch_size,
    obs_space=obs_space,
    atn_space=atn_space,
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

# Training state variables
agent_step = 0
epoch = 0
stats = {}

# Load checkpoint if exists
checkpoint_path = trainer_config.checkpoint.checkpoint_dir
checkpoint = TrainerCheckpoint.load(checkpoint_path) if checkpoint_path else None

if checkpoint:
    logger.info(f"Restoring from checkpoint at {checkpoint.agent_step} steps")
    agent_step = checkpoint.agent_step
    epoch = checkpoint.epoch

    # Load optimizer state
    if checkpoint.optimizer_state_dict:
        try:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            logger.info("Successfully loaded optimizer state from checkpoint")
        except ValueError:
            logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

# Memory and System Monitoring
memory_monitor = MemoryMonitor()
memory_monitor.add(experience, name="Experience", track_attributes=True)
memory_monitor.add(agent, name="Agent", track_attributes=False)

system_monitor = SystemMonitor(
    sampling_interval_sec=1.0,
    history_size=100,
    logger=logger,
    auto_start=True,
)

# Evaluation Configuration
evaluation_config = SimulationSuiteConfig(
    name="evaluation",
    simulations={
        "navigation/simple": SingleEnvSimulationConfig(
            env="/env/mettagrid/simple",
            num_episodes=5,
            max_time_s=30,
            env_overrides={},
        ),
        "navigation/medium": SingleEnvSimulationConfig(
            env="/env/mettagrid/medium",
            num_episodes=5,
            max_time_s=30,
            env_overrides={},
        ),
    },
    num_episodes=10,  # Will be overridden by individual configs
    env_overrides={},  # Suite-level overrides
)

# Latest saved policy record for evaluations
latest_saved_policy_record = None

# Starting training
logger.info(f"Starting training on {device}")
evaluation_scores = {}
rollout_time = 0
train_time = 0
epoch_start_time = time.time()
steps_at_epoch_start = agent_step

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

    # Reset training state
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

    # Compute advantages once
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

            # Optimize using wrapper
            optimizer.step(loss, epoch, experience.accumulate_minibatches)

            minibatch_idx += 1

        epoch += 1

        # Early exit if KL divergence is too high
        if trainer_config.ppo.target_kl is not None:
            average_approx_kl = losses.approx_kl_sum / losses.minibatches_processed
            if average_approx_kl > trainer_config.ppo.target_kl:
                break

    # Apply additional training steps
    if minibatch_idx > 0:  # Only if we actually trained
        # CUDA synchronization
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    # Step learning rate scheduler
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Calculate explained variance
    losses.explained_variance = calculate_explained_variance(experience.values, advantages)

    train_time = time.time() - train_start

    # Calculate performance metrics
    steps_calculated = agent_step - steps_before
    total_time = train_time + rollout_time
    steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

    train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
    rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0

    # Log progress similar to trainer.py
    logger.info(f"Epoch {epoch} - {steps_per_sec:.0f} steps/sec ({train_pct:.0f}% train / {rollout_pct:.0f}% rollout)")

    # Heartbeat recording
    if epoch % 10 == 0:
        record_heartbeat()

    # Update L2 weights if configured
    if hasattr(agent, "l2_init_weight_update_interval"):
        l2_interval = getattr(agent, "l2_init_weight_update_interval", 0)
        if isinstance(l2_interval, int) and l2_interval > 0 and epoch % l2_interval == 0:
            if hasattr(agent, "update_l2_init_weight_copy"):
                agent.update_l2_init_weight_copy()
                logger.info(f"Updated L2 init weights at epoch {epoch}")

    # Compute gradient statistics
    if trainer_config.grad_mean_variance_interval > 0 and epoch % trainer_config.grad_mean_variance_interval == 0:
        grad_stats = compute_gradient_stats(agent)
        logger.info(
            f"Gradient stats - mean: {grad_stats.get('grad/mean', 0):.2e}, "
            f"variance: {grad_stats.get('grad/variance', 0):.2e}, "
            f"norm: {grad_stats.get('grad/norm', 0):.2e}"
        )

    # Log system monitoring stats
    if epoch % 10 == 0:
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

    # Save checkpoint periodically
    saved_policy_path = None
    if epoch % trainer_config.checkpoint.checkpoint_interval == 0:
        logger.info(f"Saving policy at epoch {epoch}")

        # Create policy record directly
        name = policy_store.make_model_name(epoch)
        policy_record = policy_store.create_empty_policy_record(name)
        policy_record.metadata = {
            "agent_step": agent_step,
            "epoch": epoch,
            "stats": dict(stats),
        }
        policy_record.policy = agent

        # Save through policy store
        latest_saved_policy_record = policy_store.save(policy_record)
        saved_policy_path = latest_saved_policy_record
        logger.info(f"Successfully saved policy at epoch {epoch}")

        # Save training state
        logger.info("Saving training state...")
        trainer_checkpoint = TrainerCheckpoint(
            agent_step=agent_step,
            epoch=epoch,
            total_agent_step=agent_step,
            optimizer_state_dict=optimizer.state_dict(),
            policy_path=saved_policy_path.uri if hasattr(saved_policy_path, "uri") else None,
            stopwatch_state=None,  # Timer state not implemented in this example
        )
        trainer_checkpoint.save(checkpoint_path)
        logger.info(f"Saved training state at epoch {epoch}")

        # Clean up old policies to prevent disk space issues
        if epoch % 10 == 0:  # Clean up every 10 epochs
            cleanup_old_policies(checkpoint_path, keep_last_n=5)

    # Policy evaluation
    if (
        trainer_config.simulation.evaluate_interval > 0
        and epoch % trainer_config.simulation.evaluate_interval == 0
        and saved_policy_path
    ):
        logger.info(f"Evaluating policy at epoch {epoch}")

        # Run evaluation suite (similar to trainer.py's _evaluate_policy)
        sim_suite = SimulationSuite(
            config=evaluation_config,
            policy_pr=saved_policy_path,
            policy_store=policy_store,
            device=device,
            vectorization="serial",
            stats_dir=dirs.stats_dir,
            stats_client=None,  # No stats client in this example
            stats_epoch_id=None,
            wandb_policy_name=None,  # No wandb in this example
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

    # Replay generation
    if (
        trainer_config.simulation.replay_interval > 0
        and epoch % trainer_config.simulation.replay_interval == 0
        and saved_policy_path
    ):
        logger.info(f"Generating replay at epoch {epoch}")

        # Generate replay (similar to trainer.py's _generate_and_upload_replay)
        replay_sim_config = SingleEnvSimulationConfig(
            env="/env/mettagrid/simple",  # You can customize this or use curriculum
            num_episodes=1,
            max_time_s=60,
            env_overrides={},
        )

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

# Log final system stats
final_system_stats = system_monitor.get_summary()
logger.info(f"\nFinal system stats: {final_system_stats}")
system_monitor.stop()
memory_monitor.clear()

# Final checkpoint (force save)
if epoch % trainer_config.checkpoint.checkpoint_interval != 0:
    logger.info("Saving final checkpoint...")

    # Create final policy record
    name = policy_store.make_model_name(epoch)
    policy_record = policy_store.create_empty_policy_record(name)
    policy_record.metadata = {
        "agent_step": agent_step,
        "epoch": epoch,
        "final": True,
    }
    policy_record.policy = agent

    # Save through policy store
    saved_policy_path = policy_store.save(policy_record)
    logger.info("Successfully saved final policy")

    # Save final training state
    final_checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        total_agent_step=agent_step,
        optimizer_state_dict=optimizer.state_dict(),
        policy_path=saved_policy_path.uri if hasattr(saved_policy_path, "uri") else None,
        stopwatch_state=None,
    )
    final_checkpoint.save(checkpoint_path)
    logger.info("Saved final training state")

# Close environment - env is the vecenv returned by Environment()
env.close()  # type: ignore

logger.info(f"\nTraining run complete! Run saved to: {dirs.run_dir}")
