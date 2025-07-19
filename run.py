#!/usr/bin/env -S uv run
import logging
import os
import time
from collections import defaultdict

import torch
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.api import (
    Agent,
    Environment,
    Optimizer,
    cleanup_distributed,
    cleanup_wandb,
    create_evaluation_config_suite,
    create_replay_config,
    ensure_initial_policy,
    initialize_wandb,
    load_checkpoint,
    save_checkpoint,
    save_experiment_config,
    setup_distributed_training,
    setup_run_directories,
)
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_stats_db import EvalStatsDB
from metta.mettagrid import mettagrid_c  # noqa: F401
from metta.mettagrid.mettagrid_env import dtype_actions
from metta.rl.experience import Experience
from metta.rl.functions.advantage import compute_advantage
from metta.rl.functions.batch_utils import (
    calculate_batch_sizes,
    calculate_prioritized_sampling_params,
)
from metta.rl.functions.losses import process_minibatch_update
from metta.rl.functions.optimization import (
    calculate_explained_variance,
    maybe_update_l2_weights,
)
from metta.rl.functions.policy_management import wrap_agent_distributed
from metta.rl.functions.rollout import (
    get_lstm_config,
    get_observation,
    run_policy_inference,
)
from metta.rl.functions.stats import (
    accumulate_rollout_stats,
    build_wandb_stats,
    compute_timing_stats,
    process_training_stats,
)
from metta.rl.functions.utils import should_run
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

# Set up directories and distributed training
dirs = setup_run_directories()
device, is_master, world_size, rank = setup_distributed_training("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
# Note: batch_size must be >= total_agents * bptt_horizon
# With navigation curriculum: 4 agents per env * many envs = ~2048 total agents
# Required batch_size >= 2048 * 64 (bptt_horizon) = 131072
trainer_config = TrainerConfig(
    num_workers=4,
    total_timesteps=10_000_000,
    batch_size=524288 if torch.cuda.is_available() else 131072,  # 512k for GPU, 128k for CPU (minimum for navigation)
    minibatch_size=16384 if torch.cuda.is_available() else 4096,  # 16k for GPU, 4k for CPU
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
        checkpoint_interval=300,
        wandb_checkpoint_interval=0,
    ),
    simulation=SimulationConfig(
        evaluate_interval=300,
        replay_dir=dirs.replay_dir,
    ),
    profiler=TorchProfilerConfig(
        interval_epochs=0,  # Disabled by default
        profile_dir=os.path.join(dirs.run_dir, "torch_traces"),
    ),
    grad_mean_variance_interval=150,
    forward_pass_minibatch_target_size=4096 if torch.cuda.is_available() else 2048,  # Adjust for CPU
    async_factor=2,  # Add this to match trainer.yaml
)

# Adjust batch sizes for distributed training
if torch.distributed.is_initialized() and trainer_config.scale_batches_by_world_size:
    trainer_config.batch_size = trainer_config.batch_size // world_size

# Save config
save_experiment_config(dirs, device, trainer_config)

# Calculate batch sizes like trainer.py does
# We need to know num_agents first, so let's assume 4 for navigation curriculum
num_agents = 4  # Default for navigation tasks
target_batch_size, batch_size, num_envs = calculate_batch_sizes(
    forward_pass_minibatch_target_size=trainer_config.forward_pass_minibatch_target_size,
    num_agents=num_agents,
    num_workers=trainer_config.num_workers,
    async_factor=trainer_config.async_factor,
)

# Create environment
env = Environment(
    curriculum_path="/env/mettagrid/curriculum/navigation/bucketed",
    num_agents=num_agents,
    width=32,
    height=32,
    device=str(device),
    num_envs=num_envs,
    num_workers=trainer_config.num_workers,
    batch_size=batch_size,
    async_factor=trainer_config.async_factor,
    zero_copy=trainer_config.zero_copy,
    is_training=True,
    vectorization="serial",  # Match the vectorization mode
)
metta_grid_env = env.driver_env  # type: ignore - vecenv attribute

# Create agent
agent = Agent(env, device=str(device))
hidden_size, num_lstm_layers = get_lstm_config(agent)

# Evaluation configuration
evaluation_config = create_evaluation_config_suite()

# Initialize wandb if master
# This uses the helper from api.py that handles all the wandb config setup
wandb_run = None
wandb_ctx = None
if is_master:
    # Build a config similar to what Hydra would create
    full_config = {
        "run": dirs.run_name,
        "run_dir": dirs.run_dir,
        "cmd": "train",
        "device": str(device),
        "seed": 1,  # Default seed
        "trainer": trainer_config.model_dump(),
        "train_job": {"evals": evaluation_config.model_dump() if hasattr(evaluation_config, "model_dump") else {}},
    }

    wandb_run, wandb_ctx = initialize_wandb(
        run_name=dirs.run_name,
        run_dir=dirs.run_dir,
        enabled=os.environ.get("WANDB_DISABLED", "").lower() != "true",
        config=full_config,
    )

# Create policy store with config structure matching what Hydra provides
policy_store_config = {
    "device": str(device),
    "policy_cache_size": 10,
    "run": dirs.run_name,
    "run_dir": dirs.run_dir,
    "vectorization": "serial",  # Set to serial for simplicity in this example
    "trainer": trainer_config.model_dump(),
}

# Add wandb config if available (PolicyStore expects it for wandb:// URIs)
if wandb_run and wandb_ctx:
    # Access the wandb config from the context
    try:
        wandb_cfg = wandb_ctx.cfg
        if isinstance(wandb_cfg, DictConfig):
            wandb_config_dict = OmegaConf.to_container(wandb_cfg, resolve=True)
            if isinstance(wandb_config_dict, dict) and wandb_config_dict.get("enabled"):
                policy_store_config["wandb"] = {
                    "entity": wandb_config_dict.get("entity"),
                    "project": wandb_config_dict.get("project"),
                }
    except AttributeError:
        # wandb_ctx might not have cfg attribute if wandb is disabled
        pass

policy_store = PolicyStore(
    DictConfig(policy_store_config),
    wandb_run=wandb_run,
)

# Load checkpoint or create initial policy with distributed coordination
checkpoint_path = trainer_config.checkpoint.checkpoint_dir
checkpoint = load_checkpoint(checkpoint_path, None, None, policy_store, device)
agent_step, epoch, loaded_policy_path = checkpoint

# Ensure all ranks have the same initial policy
ensure_initial_policy(agent, policy_store, checkpoint_path, loaded_policy_path, device)
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
kickstarter = Kickstarter(
    trainer_config.kickstart,
    str(device),
    policy_store,
    metta_grid_env,  # Pass the full environment object, not individual attributes
)

# Create losses tracker
losses = Losses()

# Create timer
timer = Stopwatch(logger)
timer.start()

# Create learning rate scheduler
lr_scheduler = None
if getattr(trainer_config, "lr_scheduler", None) and trainer_config.lr_scheduler.enabled:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.optimizer, T_max=trainer_config.total_timesteps // trainer_config.batch_size
    )

# Memory and System Monitoring (master only)
system_monitor = None
memory_monitor = None
if is_master:
    memory_monitor = MemoryMonitor()
    memory_monitor.add(experience, name="Experience", track_attributes=True)
    memory_monitor.add(agent, name="Agent", track_attributes=False)

    system_monitor = SystemMonitor(
        sampling_interval_sec=1.0,
        history_size=100,
        logger=logger,
        auto_start=True,
        external_timer=timer,  # Pass timer for persistent elapsed time across restarts
    )

# Training loop
saved_policy_path = None
logger.info(f"Starting training on {device}")
evaluation_scores = {}
epoch_start_time = time.time()
steps_at_epoch_start = agent_step
stats = defaultdict(list)  # Use defaultdict like trainer.py
initial_policy_record = None  # Track initial policy
current_policy_generation = 0  # Track policy generation

while agent_step < trainer_config.total_timesteps:
    steps_before = agent_step

    # ===== ROLLOUT PHASE =====
    rollout_start = time.time()
    raw_infos = []
    experience.reset_for_rollout()

    # Collect experience
    while not experience.ready_for_training:
        # Receive environment data
        o, r, d, t, info, training_env_id, mask, num_steps = get_observation(env, device, timer)
        agent_step += num_steps

        # Run policy inference
        actions, selected_action_log_probs, values, lstm_state_to_store = run_policy_inference(
            agent, o, experience, training_env_id.start, device
        )

        # Store experience
        experience.store(
            obs=o,
            actions=actions,
            logprobs=selected_action_log_probs,
            rewards=r,
            dones=d,
            truncations=t,
            values=values,
            env_id=training_env_id,
            mask=mask,
            lstm_state=lstm_state_to_store,
        )

        # Send actions back to environment
        with timer("_rollout.env"):
            env.send(actions.cpu().numpy().astype(dtype_actions))  # type: ignore - env is vecenv wrapper

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
    anneal_beta = calculate_prioritized_sampling_params(
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

    if minibatch_idx > 0 and str(device).startswith("cuda"):
        torch.cuda.synchronize()

    if lr_scheduler is not None:
        lr_scheduler.step()

    losses.explained_variance = calculate_explained_variance(experience.values, advantages)

    # Calculate performance metrics
    train_time = time.time() - train_start

    # ===== STATS PROCESSING PHASE =====
    stats_start = time.time()

    # Process collected stats (convert lists to means)
    processed_stats = process_training_stats(
        raw_stats=stats,
        losses=losses,
        experience=experience,
        trainer_config=trainer_config,
        kickstarter=kickstarter,
    )

    # Update stats with mean values for consistency
    stats = processed_stats["mean_stats"]

    # Compute timing stats
    timing_info = compute_timing_stats(
        timer=timer,
        agent_step=agent_step,
    )

    # Build complete stats for wandb
    if is_master:
        # Get current learning rate
        current_lr = trainer_config.optimizer.learning_rate
        if hasattr(optimizer, "param_groups"):
            current_lr = optimizer.param_groups[0]["lr"]
        elif hasattr(optimizer, "optimizer"):
            current_lr = optimizer.optimizer.param_groups[0]["lr"]

        # Build parameters dictionary
        parameters = {
            "learning_rate": current_lr,
            "epoch_steps": timing_info["epoch_steps"],
            "num_minibatches": experience.num_minibatches,
            "generation": current_policy_generation,
        }

        # Get system and memory stats
        system_stats = system_monitor.stats() if system_monitor else {}
        memory_stats = memory_monitor.stats() if memory_monitor else {}

        # Build complete stats dictionary
        all_stats = build_wandb_stats(
            processed_stats=processed_stats,
            timing_info=timing_info,
            weight_stats={},  # Weight stats not computed in run.py
            grad_stats={},  # Grad stats not computed in run.py
            system_stats=system_stats,
            memory_stats=memory_stats,
            parameters=parameters,
            hyperparameters={},  # Hyperparameters not used in run.py
            evals=evaluation_scores.get(epoch, EvalRewardSummary()),
            agent_step=agent_step,
            epoch=epoch,
        )

        # Log to wandb if available
        if wandb_run:
            wandb_run.log(all_stats, step=agent_step)

        # Also log key metrics to console
        losses_stats = processed_stats["losses_stats"]
        log_parts = []

        if losses_stats:
            log_parts.append(
                f"Loss[P:{losses_stats.get('policy_loss', 0):.3f} "
                f"V:{losses_stats.get('value_loss', 0):.3f} "
                f"E:{losses_stats.get('entropy', 0):.3f} "
                f"EV:{losses_stats.get('explained_variance', 0):.2f}]"
            )

        if "reward" in processed_stats["overview"]:
            log_parts.append(f"Reward:{processed_stats['overview']['reward']:.2f}")

        log_parts.append(f"LR:{current_lr:.1e}")

        if log_parts:
            logger.info(" | ".join(log_parts))

    # Clear stats for next iteration
    stats.clear()

    stats_time = time.time() - stats_start

    # Calculate total time and percentages
    steps_calculated = agent_step - steps_before
    total_time = train_time + rollout_time + stats_time
    steps_per_sec = steps_calculated / total_time if total_time > 0 else 0
    steps_per_sec *= world_size

    train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
    rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0
    stats_pct = (stats_time / total_time) * 100 if total_time > 0 else 0

    logger.info(
        f"Epoch {epoch} - {steps_per_sec:.0f} steps/sec "
        f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout / {stats_pct:.0f}% stats)"
    )

    # Record heartbeat periodically (master only)
    if should_run(epoch, 10, is_master):
        record_heartbeat()

    # Update L2 weights if configured
    if hasattr(agent, "l2_init_weight_update_interval"):
        maybe_update_l2_weights(
            agent=agent,
            epoch=epoch,
            interval=getattr(agent, "l2_init_weight_update_interval", 0),
            is_master=is_master,
        )

    # Save checkpoint periodically
    if should_run(epoch, trainer_config.checkpoint.checkpoint_interval, True):  # All ranks participate
        saved_policy_path = save_checkpoint(
            epoch=epoch,
            agent_step=agent_step,
            agent=agent,
            optimizer=optimizer,
            policy_store=policy_store,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=trainer_config.checkpoint.checkpoint_interval,
            stats=processed_stats["mean_stats"],
            force_save=False,
        )
        # Ensure all ranks synchronize after checkpoint saving
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

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
        category_scores: dict[str, float] = {}
        categories = set()
        for sim_name in evaluation_config.simulations.keys():
            categories.add(sim_name.split("/")[0])

        for category in categories:
            score = stats_db.get_average_metric_by_filter("reward", saved_policy_path, f"sim_name LIKE '%{category}%'")
            logger.info(f"{category} score: {score}")
            record_heartbeat()
            if score is not None:
                category_scores[category] = score

        # Get detailed per-simulation scores
        per_sim_scores: dict[tuple[str, str], float] = {}
        all_scores = stats_db.simulation_scores(saved_policy_path, "reward")
        for (_, sim_name, _), score in all_scores.items():
            category = sim_name.split("/")[0]
            sim_short_name = sim_name.split("/")[-1]
            per_sim_scores[(category, sim_short_name)] = score

        evaluation_scores[epoch] = EvalRewardSummary(
            category_scores=category_scores,
            simulation_scores=per_sim_scores,
        )
        stats_db.close()

        # Replay generation (master only)
        if is_master and saved_policy_path:
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

# Training complete
total_elapsed = time.time() - epoch_start_time
logger.info("Training complete!")
logger.info(f"Total training time: {total_elapsed:.1f}s")
logger.info(f"Final epoch: {epoch}, Total steps: {agent_step}")

# Log evaluation history
if evaluation_scores:
    logger.info("\nEvaluation History:")
    for eval_epoch, scores in sorted(evaluation_scores.items()):
        logger.info(f"  Epoch {eval_epoch}:")
        for env_name, score in scores.items():
            logger.info(f"    {env_name}: {score:.2f}")

# Stop monitoring if master
if is_master:
    if system_monitor:
        system_monitor.stop()
    if memory_monitor:
        memory_monitor.clear()

# Save final checkpoint
saved_policy_path = save_checkpoint(
    epoch=epoch,
    agent_step=agent_step,
    agent=agent,
    optimizer=optimizer,
    policy_store=policy_store,
    checkpoint_path=checkpoint_path,
    checkpoint_interval=trainer_config.checkpoint.checkpoint_interval,
    stats={},  # Empty dict since processed_stats is out of scope
    force_save=True,
)

# Ensure all ranks synchronize after final checkpoint
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# Close environment
env.close()  # type: ignore

logger.info(f"\nTraining run complete! Run saved to: {dirs.run_dir}")

# Clean up distributed training if initialized
cleanup_distributed()

# Clean up wandb if initialized
if is_master:
    cleanup_wandb(wandb_ctx)
