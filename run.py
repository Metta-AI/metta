#!/usr/bin/env -S uv run
import logging
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.distributed
from heavyball import ForeachMuon
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.common.wandb.wandb_context import WandbContext
from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_stats_db import EvalStatsDB
from metta.interface import (
    Environment,
    create_evaluation_config_suite,
    setup_run_directories,
)
from metta.interface.agent import create_or_load_agent
from metta.interface.directories import save_experiment_config
from metta.mettagrid import mettagrid_c  # noqa: F401
from metta.mettagrid.mettagrid_env import dtype_actions
from metta.rl.experience import Experience
from metta.rl.hyperparameter_scheduler import HyperparameterScheduler
from metta.rl.kickstarter import Kickstarter
from metta.rl.kickstarter_config import KickstartConfig
from metta.rl.losses import Losses
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.trainer_config import (
    CheckpointConfig,
    InitialPolicyConfig,
    OptimizerConfig,
    PPOConfig,
    PrioritizedExperienceReplayConfig,
    SimulationConfig,
    TorchProfilerConfig,
    TrainerConfig,
    VTraceConfig,
)
from metta.rl.util.advantage import compute_advantage
from metta.rl.util.batch_utils import (
    calculate_batch_sizes,
    calculate_prioritized_sampling_params,
)
from metta.rl.util.distributed import setup_device_and_distributed
from metta.rl.util.losses import process_minibatch_update
from metta.rl.util.optimization import (
    calculate_explained_variance,
    compute_gradient_stats,
    maybe_update_l2_weights,
)
from metta.rl.util.policy_management import (
    cleanup_old_policies,
    save_policy_with_metadata,
    validate_policy_environment_match,
    wrap_agent_distributed,
)
from metta.rl.util.rollout import (
    get_lstm_config,
    get_observation,
    run_policy_inference,
)
from metta.rl.util.stats import (
    accumulate_rollout_stats,
    build_wandb_stats,
    compute_timing_stats,
    process_training_stats,
)
from metta.rl.util.utils import should_run
from metta.sim.simulation_config import SimulationSuiteConfig, SingleEnvSimulationConfig
from metta.sim.simulation_suite import SimulationSuite

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set up directories
dirs = setup_run_directories()

# Set up device and distributed training
device, is_master, world_size, rank = setup_device_and_distributed("cuda" if torch.cuda.is_available() else "cpu")

# Configuration using individual component configs
# Note: batch_size must be >= total_agents * bptt_horizon
# With navigation curriculum: 4 agents per env * many envs = ~2048 total agents
# Required batch_size >= 2048 * 64 (bptt_horizon) = 131072

# Core training parameters
num_workers = 4
total_timesteps = 10_000_000
batch_size = 524288 if torch.cuda.is_available() else 131072  # 512k for GPU, 128k for CPU
minibatch_size = 16384 if torch.cuda.is_available() else 4096  # 16k for GPU, 4k for CPU
curriculum = "/env/mettagrid/curriculum/navigation/bucketed"
bptt_horizon = 64
update_epochs = 1
forward_pass_minibatch_target_size = 4096 if torch.cuda.is_available() else 2048
async_factor = 2
grad_mean_variance_interval = 150
scale_batches_by_world_size = False
cpu_offload = False
zero_copy = True

# Individual component configs with explicit values
ppo_config = PPOConfig(
    clip_coef=0.1,
    ent_coef=0.01,
    gamma=0.99,
    gae_lambda=0.95,
)

optimizer_config = OptimizerConfig(
    type="adam",
    learning_rate=3e-4,
)

checkpoint_config = CheckpointConfig(
    checkpoint_dir=dirs.checkpoint_dir,
    checkpoint_interval=300,
    wandb_checkpoint_interval=0,
)

simulation_config = SimulationConfig(
    evaluate_interval=300,
    replay_dir=dirs.replay_dir,
)

profiler_config = TorchProfilerConfig(
    interval_epochs=0,  # Disabled by default
    profile_dir=os.path.join(dirs.run_dir, "torch_traces"),
)

# Use defaults for these
prioritized_replay_config = PrioritizedExperienceReplayConfig()
vtrace_config = VTraceConfig()
kickstart_config = KickstartConfig()

# Check for initial policy URI from environment variable
initial_policy_uri = os.environ.get("INITIAL_POLICY_URI", None)
initial_policy_config = InitialPolicyConfig(uri=initial_policy_uri)

# Create a trainer config for compatibility with functions that expect it
# This is just for backward compatibility - we use the individual configs directly
trainer_config = TrainerConfig(
    num_workers=num_workers,
    total_timesteps=total_timesteps,
    batch_size=batch_size,
    minibatch_size=minibatch_size,
    curriculum=curriculum,
    bptt_horizon=bptt_horizon,
    update_epochs=update_epochs,
    forward_pass_minibatch_target_size=forward_pass_minibatch_target_size,
    async_factor=async_factor,
    grad_mean_variance_interval=grad_mean_variance_interval,
    scale_batches_by_world_size=scale_batches_by_world_size,
    cpu_offload=cpu_offload,
    zero_copy=zero_copy,
    ppo=ppo_config,
    optimizer=optimizer_config,
    checkpoint=checkpoint_config,
    simulation=simulation_config,
    profiler=profiler_config,
    prioritized_experience_replay=prioritized_replay_config,
    vtrace=vtrace_config,
    kickstart=kickstart_config,
    initial_policy=initial_policy_config,
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

# Remove the curriculum object creation - Environment will handle it
# The Environment class can create the curriculum internally from the path

# Agent will be created later after checking for checkpoint
# Evaluation configuration
evaluation_config = create_evaluation_config_suite()

# WandB initialization
wandb_run = None
wandb_ctx = None
if is_master:
    # Build a config similar to what Hydra would create
    wandb_enabled = os.environ.get("WANDB_DISABLED", "").lower() != "true"

    if wandb_enabled:
        wandb_config = DictConfig(
            {
                "enabled": True,
                "project": os.environ.get("WANDB_PROJECT", "metta"),
                "entity": os.environ.get("WANDB_ENTITY", "metta-research"),
                "group": dirs.run_name,
                "name": dirs.run_name,
                "run_id": dirs.run_name,
                "data_dir": dirs.run_dir,
                "job_type": "train",
                "tags": [],
                "notes": "",
            }
        )
    else:
        wandb_config = DictConfig({"enabled": False})

    global_config = DictConfig(
        {
            "run": dirs.run_name,
            "run_dir": dirs.run_dir,
            "cmd": "train",
            "device": str(device),
            "seed": 1,  # Default seed
            "trainer": trainer_config.model_dump(),
            "train_job": {"evals": evaluation_config.model_dump() if hasattr(evaluation_config, "model_dump") else {}},
            "wandb": wandb_config,
        }
    )

    wandb_ctx = WandbContext(wandb_config, global_config)
    wandb_run = wandb_ctx.__enter__()

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

# Create or load agent with a single function call
agent, policy_record, agent_step, epoch, checkpoint = create_or_load_agent(
    env=env,  # type: ignore - interface.Environment works like api.Environment
    run_dir=dirs.run_dir,
    policy_store=policy_store,
    trainer_config=trainer_config,
    device=device,
    is_master=is_master,
    rank=rank,
)

# Get LSTM config from the agent
hidden_size, num_lstm_layers = get_lstm_config(agent)

# Validate that policy matches environment
validate_policy_environment_match(agent, metta_grid_env)  # type: ignore

# Store initial policy record for later use
initial_policy_record = policy_record

# Create optimizer like trainer.py does

optimizer_type = trainer_config.optimizer.type
opt_cls = torch.optim.Adam if optimizer_type == "adam" else ForeachMuon
# ForeachMuon expects int for weight_decay, Adam expects float
weight_decay = trainer_config.optimizer.weight_decay
if optimizer_type != "adam":
    # Ensure weight_decay is int for ForeachMuon
    weight_decay = int(weight_decay)

optimizer = opt_cls(
    agent.parameters(),  # type: ignore - agent is MettaAgent from factory
    lr=trainer_config.optimizer.learning_rate,
    betas=(trainer_config.optimizer.beta1, trainer_config.optimizer.beta2),
    eps=trainer_config.optimizer.eps,
    weight_decay=weight_decay,  # type: ignore - ForeachMuon type annotation issue
)

if checkpoint and checkpoint.optimizer_state_dict:
    try:
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        logger.info("Successfully loaded optimizer state from checkpoint")
    except ValueError:
        logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

# Wrap agent for distributed training
agent = wrap_agent_distributed(agent, device)

# Ensure all ranks have wrapped their agents before proceeding
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# Log model parameters to wandb
if is_master and wandb_run:
    num_params = sum(p.numel() for p in agent.parameters())  # type: ignore
    if wandb_run.summary:
        wandb_run.summary["model/total_parameters"] = num_params
    logger.info(f"Model has {num_params:,} parameters")

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
        optimizer, T_max=trainer_config.total_timesteps // trainer_config.batch_size
    )

# Create hyperparameter scheduler (handles dynamic learning rate and other hyperparameter adjustments)
hyperparameter_scheduler = HyperparameterScheduler(
    DictConfig(trainer_config), optimizer, trainer_config.total_timesteps, logging
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

# Track policy records for consistency with trainer.py
initial_policy_record = policy_record
latest_saved_policy_record = policy_record

# Training loop
logger.info(f"Starting training on {device}")
evaluation_scores = {}
epoch_start_time = time.time()
steps_at_epoch_start = agent_step
stats = defaultdict(list)  # Use defaultdict like trainer.py
grad_stats = {}
current_policy_generation = initial_policy_record.metadata.get("generation", 0) + 1 if initial_policy_record else 0

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
    anneal_beta = calculate_prioritized_sampling_params(
        epoch=epoch,
        total_timesteps=trainer_config.total_timesteps,
        batch_size=trainer_config.batch_size,
        prio_alpha=trainer_config.prioritized_experience_replay.prio_alpha,
        prio_beta0=trainer_config.prioritized_experience_replay.prio_beta0,
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
                prio_alpha=trainer_config.prioritized_experience_replay.prio_alpha,
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

            # Optimizer step like trainer.py
            optimizer.zero_grad()
            loss.backward()

            if (minibatch_idx + 1) % experience.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), trainer_config.ppo.max_grad_norm)
                optimizer.step()

                # Optional weight clipping
                if hasattr(agent, "clip_weights"):
                    agent.clip_weights()

                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()

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

    # Update hyperparameter scheduler
    hyperparameter_scheduler.step(agent_step)

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
        current_lr = optimizer.param_groups[0]["lr"]

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

        # Current hyperparameter values
        hyperparameters = {
            "learning_rate": current_lr,
            "ppo_clip_coef": trainer_config.ppo.clip_coef,
            "ppo_vf_clip_coef": trainer_config.ppo.vf_clip_coef,
            "ppo_ent_coef": trainer_config.ppo.ent_coef,
            "ppo_l2_reg_loss_coef": trainer_config.ppo.l2_reg_loss_coef,
            "ppo_l2_init_loss_coef": trainer_config.ppo.l2_init_loss_coef,
        }

        # Build complete stats dictionary
        all_stats = build_wandb_stats(
            processed_stats=processed_stats,
            timing_info=timing_info,
            weight_stats={},  # Weight stats not computed in run.py
            grad_stats=grad_stats,
            system_stats=system_stats,
            memory_stats=memory_stats,
            parameters=parameters,
            hyperparameters=hyperparameters,
            evals=evaluation_scores.get(epoch, EvalRewardSummary()),
            agent_step=agent_step,
            epoch=epoch,
        )

        # Log to wandb if available
        if wandb_run:
            wandb_run.log(all_stats, step=agent_step)

    # Clear stats for next iteration
    stats.clear()
    grad_stats.clear()

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

    # Compute gradient statistics (master only)
    if should_run(epoch, trainer_config.grad_mean_variance_interval, is_master):
        grad_stats = compute_gradient_stats(agent)

    # Save checkpoint periodically
    if should_run(epoch, trainer_config.checkpoint.checkpoint_interval, True):  # All ranks participate
        # Save policy with metadata (master only)
        if is_master:
            saved_record = save_policy_with_metadata(
                policy=agent,
                policy_store=policy_store,
                epoch=epoch,
                agent_step=agent_step,
                evals=evaluation_scores.get(epoch, {}),
                timer=timer,
                initial_policy_record=initial_policy_record,
                run_name=dirs.run_name,
                is_master=is_master,
            )

            if saved_record:
                latest_saved_policy_record = saved_record

                # Clean up old policies periodically
                if epoch % 10 == 0:
                    cleanup_old_policies(trainer_config.checkpoint.checkpoint_dir, keep_last_n=5)

        # Save training state (master only)
        if is_master:
            extra_args = {}
            if kickstarter.enabled and kickstarter.teacher_uri is not None:
                extra_args["teacher_pr_uri"] = kickstarter.teacher_uri

            latest_uri = latest_saved_policy_record.uri if latest_saved_policy_record else None
            checkpoint = TrainerCheckpoint(
                agent_step=agent_step,
                epoch=epoch,
                optimizer_state_dict=optimizer.state_dict(),
                stopwatch_state=timer.save_state(),
                policy_path=latest_uri,
                extra_args=extra_args,
            )
            checkpoint.save(dirs.run_dir)
            logger.info(f"Saved training state at epoch {epoch}")

        # Ensure all ranks synchronize after checkpoint saving
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Policy evaluation (master only)
    if (
        is_master
        and trainer_config.simulation.evaluate_interval > 0
        and epoch % trainer_config.simulation.evaluate_interval == 0
        and latest_saved_policy_record
    ):
        logger.info(f"Evaluating policy at epoch {epoch}")

        # Create extended evaluation config with training task
        extended_eval_config = SimulationSuiteConfig(
            name=evaluation_config.name,
            simulations=dict(evaluation_config.simulations),
            env_overrides=evaluation_config.env_overrides,
            num_episodes=evaluation_config.num_episodes,
        )

        # Add training task to the suite
        # Use the environment's current task configuration
        # Note: This is a simplified approach - in practice, you might want to
        # save the actual task configuration during training
        training_task_config = SingleEnvSimulationConfig(
            env=trainer_config.curriculum,  # Use the same curriculum path
            num_episodes=1,
            env_overrides={},  # Empty overrides - use curriculum defaults
        )
        extended_eval_config.simulations["eval/training_task"] = training_task_config

        # Run evaluation suite
        sim_suite = SimulationSuite(
            config=extended_eval_config,
            policy_pr=latest_saved_policy_record,
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
        for sim_name in extended_eval_config.simulations.keys():
            categories.add(sim_name.split("/")[0])

        for category in categories:
            score = stats_db.get_average_metric_by_filter(
                "reward", latest_saved_policy_record, f"sim_name LIKE '%{category}%'"
            )
            logger.info(f"{category} score: {score}")
            record_heartbeat()
            if score is not None:
                category_scores[category] = score

        # Get detailed per-simulation scores
        per_sim_scores: dict[tuple[str, str], float] = {}
        all_scores = stats_db.simulation_scores(latest_saved_policy_record, "reward")
        for (_, sim_name, _), score in all_scores.items():
            category = sim_name.split("/")[0]
            sim_short_name = sim_name.split("/")[-1]
            per_sim_scores[(category, sim_short_name)] = score

        evaluation_scores[epoch] = EvalRewardSummary(
            category_scores=category_scores,
            simulation_scores=per_sim_scores,
        )

        # Set policy metadata score for sweep_eval.py
        category_score_values = list(category_scores.values())
        if category_score_values and latest_saved_policy_record:
            latest_saved_policy_record.metadata["score"] = float(np.mean(category_score_values))
            logger.info(f"Set policy metadata score to {latest_saved_policy_record.metadata['score']}")

        stats_db.close()

        # Replay generation (master only)
        if is_master and latest_saved_policy_record:
            logger.info(f"Generating replay at epoch {epoch}")

            # Generate replay using the same function as trainer.py
            # For now, skip replay generation as it requires a curriculum object
            # In a production setup, you'd create a curriculum object or use an alternative approach
            logger.info("Skipping replay generation in run.py - requires curriculum object")

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
# Save policy with metadata (master only)
if is_master:
    saved_record = save_policy_with_metadata(
        policy=agent,
        policy_store=policy_store,
        epoch=epoch,
        agent_step=agent_step,
        evals=evaluation_scores.get(epoch, {}),
        timer=timer,
        initial_policy_record=initial_policy_record,
        run_name=dirs.run_name,
        is_master=is_master,
    )

    if saved_record:
        latest_saved_policy_record = saved_record

# Save training state (master only)
if is_master:
    extra_args = {}
    if kickstarter.enabled and kickstarter.teacher_uri is not None:
        extra_args["teacher_pr_uri"] = kickstarter.teacher_uri

    latest_uri = latest_saved_policy_record.uri if latest_saved_policy_record else None
    checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        optimizer_state_dict=optimizer.state_dict(),
        stopwatch_state=timer.save_state(),
        policy_path=latest_uri,
        extra_args=extra_args,
    )
    checkpoint.save(dirs.run_dir)
    logger.info("Saved final training state")

# Ensure all ranks synchronize after final checkpoint
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# Close environment
env.close()  # type: ignore

logger.info(f"\nTraining run complete! Run saved to: {dirs.run_dir}")

# Clean up distributed training if initialized
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

# Clean up wandb if initialized
if is_master:
    wandb_ctx.__exit__(None, None, None)
