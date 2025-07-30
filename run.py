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
from metta.common.util.constants import METTASCOPE_REPLAY_URL
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.common.wandb.wandb_context import WandbContext
from metta.eval.eval_request_config import EvalRewardSummary
from metta.eval.eval_stats_db import EvalStatsDB
from metta.interface.agent import create_or_load_agent
from metta.interface.directories import save_experiment_config, setup_run_directories
from metta.interface.environment import Environment
from metta.interface.evaluation import create_evaluation_config_suite
from metta.mettagrid import mettagrid_c  # noqa: F401
from metta.mettagrid.mettagrid_env import dtype_actions
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_config import (
    CheckpointConfig,
    InitialPolicyConfig,
    KickstartConfig,
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
    validate_policy_environment_match,
    wrap_agent_distributed,
)
from metta.rl.util.rollout import (
    get_lstm_config,
    get_observation,
    run_policy_inference,
)
from metta.rl.util.stats import (
    StatsTracker,
    accumulate_rollout_stats,
    build_wandb_stats,
    compute_timing_stats,
    process_training_stats,
)
from metta.rl.util.training_loop import should_run
from metta.rl.wandb import (
    abort_requested,
    log_model_parameters,
    setup_wandb_metrics,
    upload_env_configs,
    upload_replay_html,
)
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
batch_size = 524288 if torch.cuda.is_available() else 16384  # 512k for GPU, 16k for CPU
minibatch_size = 16384 if torch.cuda.is_available() else 1024  # 16k for GPU, 1k for CPU
curriculum = "/env/mettagrid/curriculum/navigation/bucketed"
bptt_horizon = 64
update_epochs = 1
forward_pass_minibatch_target_size = 4096 if torch.cuda.is_available() else 256

# Adjust defaults based on vectorization mode
vectorization_mode = "serial"  # Use serial for macOS compatibility
if vectorization_mode == "serial":
    async_factor = 1
    zero_copy = False
else:
    async_factor = 2
    zero_copy = True

grad_mean_variance_interval = 150
scale_batches_by_world_size = False
cpu_offload = False

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

# Create a trainer config
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

# Create optimizer like bad_run.py does
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

# Set up wandb metrics and log model parameters
if wandb_run and is_master:
    setup_wandb_metrics(wandb_run)
    log_model_parameters(agent, wandb_run)

# Log to console
if is_master:
    num_params = sum(p.numel() for p in agent.parameters())  # type: ignore
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

# Initialize specialized state containers
stats_tracker = StatsTracker(rollout_stats=defaultdict(list))
eval_scores = EvalRewardSummary()  # Initialize eval_scores

# Track policy records individually
latest_saved_policy_record = policy_record
initial_policy_uri = policy_record.uri if policy_record else None
initial_generation = policy_record.metadata.get("generation", 0) if policy_record else 0
last_evaluation_epoch = epoch - 1  # Track last epoch when evaluation was performed

# Create checkpoint manager
checkpoint_manager = CheckpointManager(
    trainer_cfg=trainer_config,
    policy_store=policy_store,
    checkpoint_dir=trainer_config.checkpoint.checkpoint_dir,
    run_name=dirs.run_name,
    is_master=is_master,
)

# Training loop
logger.info(f"Starting training on {device}")
current_policy_generation = initial_generation + 1 if policy_record else 0

# After environment is initialized but before training loop
if is_master and wandb_run:
    curr_obj = getattr(metta_grid_env, "_curriculum", None)
    if curr_obj is not None and hasattr(curr_obj, "get_env_cfg_by_bucket"):
        env_configs = curr_obj.get_env_cfg_by_bucket()
        upload_env_configs(env_configs=env_configs, wandb_run=wandb_run)

# Create torch profiler (matches bad_run.py)
torch_profiler = TorchProfiler(is_master, trainer_config.profiler, wandb_run, dirs.run_dir)

training_start_time = time.time()

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
    accumulate_rollout_stats(raw_infos, stats_tracker.rollout_stats)
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

            if optimizer_type == "adam":
                optimizer.step()
            else:
                # ForeachMuon has custom step signature
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

    losses.explained_variance = calculate_explained_variance(experience.values, advantages)

    # Calculate performance metrics
    train_time = time.time() - train_start

    torch_profiler.on_epoch_end(epoch)

    # ===== STATS PROCESSING PHASE =====
    stats_start = time.time()

    # Process collected stats (convert lists to means)
    processed_stats = process_training_stats(
        raw_stats=stats_tracker.rollout_stats,
        losses=losses,
        experience=experience,
        trainer_config=trainer_config,
        kickstarter=kickstarter,
    )

    # Update stats with mean values for consistency
    stats_tracker.rollout_stats = processed_stats["mean_stats"]

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

        # Compute weight stats if configured
        weight_stats = {}
        if hasattr(trainer_config, "agent") and hasattr(trainer_config.agent, "analyze_weights_interval"):
            if (
                trainer_config.agent.analyze_weights_interval != 0
                and epoch % trainer_config.agent.analyze_weights_interval == 0
            ):
                for metrics in agent.compute_weight_metrics():
                    name = metrics.get("name", "unknown")
                    for key, value in metrics.items():
                        if key != "name":
                            weight_stats[f"weights/{key}/{name}"] = value

        # Build complete stats dictionary
        all_stats = build_wandb_stats(
            processed_stats=processed_stats,
            timing_info=timing_info,
            weight_stats=weight_stats,
            grad_stats=stats_tracker.grad_stats,
            system_stats=system_stats,
            memory_stats=memory_stats,
            parameters=parameters,
            hyperparameters=hyperparameters,
            evals=eval_scores,
            agent_step=agent_step,
            epoch=epoch,
        )

        # Log to wandb if available
        if wandb_run:
            wandb_run.log(all_stats, step=agent_step)

    # Clear stats for next iteration
    stats_tracker.clear_rollout_stats()
    stats_tracker.clear_grad_stats()

    stats_time = time.time() - stats_start

    # Calculate total time and percentages
    steps_calculated = agent_step - steps_before
    total_time = train_time + rollout_time + stats_time
    steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

    train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
    rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0
    stats_pct = (stats_time / total_time) * 100

    # Format total timesteps with scientific notation for large numbers
    total_timesteps_for_log = trainer_config.total_timesteps
    if total_timesteps_for_log >= 1e9:
        total_steps_str = f"{total_timesteps_for_log:.0e}"
    else:
        total_steps_str = f"{total_timesteps_for_log:,}"

    logger.info(
        f"Epoch {epoch}- "
        f"{steps_per_sec:.0f} SPS- "
        f"step {agent_step}/{total_steps_str}- "
        f"({train_pct:.0f}% train- {rollout_pct:.0f}% rollout- {stats_pct:.0f}% stats)"
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
        stats_tracker.grad_stats = compute_gradient_stats(agent)

    # Save checkpoint periodically - all ranks must participate in checkpoint decision
    if checkpoint_manager.should_checkpoint(epoch):
        saved_record = checkpoint_manager.save_policy(
            policy=agent,
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
                    run_dir=dirs.run_dir,
                    kickstarter=kickstarter,
                )

        # All ranks must synchronize after checkpoint operations
        # This barrier must be outside the if saved_record block so all ranks hit it
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Upload latest policy to wandb (master only)
    if (
        is_master
        and wandb_run
        and latest_saved_policy_record
        and should_run(epoch, trainer_config.checkpoint.wandb_checkpoint_interval, True)
    ):
        try:
            policy_store.add_to_wandb_run(wandb_run.id, latest_saved_policy_record)
            logger.info(f"Uploaded policy to wandb at epoch {epoch}")
        except Exception as e:
            logger.warning(f"Failed to upload policy to wandb: {e}")

    # Abort check via wandb tag (master only)
    if is_master and wandb_run and should_run(epoch, 5, True):
        if abort_requested(wandb_run, min_interval_sec=60):
            break

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
        # Get the actual task configuration from the curriculum
        curr_obj = getattr(metta_grid_env, "_curriculum", None)
        if curr_obj:
            # Pass the config as _pre_built_env_config to avoid Hydra loading
            task_cfg = curr_obj.get_task().env_cfg()
            training_task_config = SingleEnvSimulationConfig(
                env="eval/training_task",  # Just a descriptive name
                num_episodes=1,
                env_overrides={"_pre_built_env_config": task_cfg},
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

        eval_scores = EvalRewardSummary(
            category_scores=category_scores,
            simulation_scores=per_sim_scores,
        )

        # Set policy metadata score for sweep_eval.py
        category_score_values = list(category_scores.values())
        if category_score_values and latest_saved_policy_record:
            latest_saved_policy_record.metadata["score"] = float(np.mean(category_score_values))
            logger.info(f"Set policy metadata score to {latest_saved_policy_record.metadata['score']}")

        stats_db.close()

        # Track that we evaluated at this epoch
        last_evaluation_epoch = epoch

        # Upload replay HTML if we have replay URLs
        if is_master and wandb_run and hasattr(results, "replay_urls") and results.replay_urls:
            upload_replay_html(
                replay_urls=results.replay_urls,
                agent_step=agent_step,
                epoch=epoch,
                wandb_run=wandb_run,
            )

        # Replay generation (master only)
        if is_master and latest_saved_policy_record:
            logger.info(f"Generating replay at epoch {epoch}")

            # Get replay URLs from the database
            replay_urls = results.stats_db.get_replay_urls()
            if replay_urls:
                replay_url = replay_urls[0]
                player_url = f"{METTASCOPE_REPLAY_URL}/?replayUrl={replay_url}"
                logger.info(f"Replay available at: {player_url}")

            results.stats_db.close()

# Training complete
total_elapsed = time.time() - training_start_time
logger.info("Training complete!")
logger.info(f"Total training time: {total_elapsed:.1f}s")
logger.info(f"Final epoch: {epoch}, Total steps: {agent_step}")

# Stop monitoring if master
if is_master:
    if system_monitor:
        system_monitor.stop()
    if memory_monitor:
        memory_monitor.clear()

# Always evaluate policy at training end if we haven't just evaluated
if is_master and last_evaluation_epoch < epoch and latest_saved_policy_record:
    logger.info(f"Performing final evaluation at epoch {epoch}")

    # Create extended evaluation config with training task
    extended_eval_config = SimulationSuiteConfig(
        name=evaluation_config.name,
        simulations=dict(evaluation_config.simulations),
        env_overrides=evaluation_config.env_overrides,
        num_episodes=evaluation_config.num_episodes,
    )

    # Add training task to the suite
    # Get the actual task configuration from the curriculum
    curr_obj = getattr(metta_grid_env, "_curriculum", None)
    if curr_obj:
        # Pass the config as _pre_built_env_config to avoid Hydra loading
        task_cfg = curr_obj.get_task().env_cfg()
        training_task_config = SingleEnvSimulationConfig(
            env="eval/training_task",  # Just a descriptive name
            num_episodes=1,
            env_overrides={"_pre_built_env_config": task_cfg},
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
    logger.info("Final evaluation complete")

    # Build evaluation metrics
    categories = set()
    for sim_name in extended_eval_config.simulations.keys():
        categories.add(sim_name.split("/")[0])

    category_scores = {}
    for category in categories:
        score = stats_db.get_average_metric_by_filter(
            "reward", latest_saved_policy_record, f"sim_name LIKE '%{category}%'"
        )
        logger.info(f"{category} score: {score}")
        record_heartbeat()
        if score is not None:
            category_scores[category] = score

    # Get detailed per-simulation scores
    simulation_scores = {}
    all_scores = stats_db.simulation_scores(latest_saved_policy_record, "reward")
    for (_, sim_name, _), score in all_scores.items():
        simulation_scores[sim_name] = score

    # Create EvalRewardSummary
    category_score_values = list(category_scores.values())
    simulation_score_values = list(simulation_scores.values())
    eval_scores = EvalRewardSummary(
        category_scores=category_scores,
        simulation_scores=simulation_scores,
        avg_category_score=np.mean(category_score_values) if category_score_values else 0.0,
        avg_simulation_score=np.mean(simulation_score_values) if simulation_score_values else 0.0,
    )

    # Update policy metadata with score
    if category_score_values:
        latest_saved_policy_record.metadata["score"] = float(np.mean(category_score_values))
        logger.info(f"Set policy metadata score to {latest_saved_policy_record.metadata['score']}")

    stats_db.close()

    # Upload replay HTML if we have replay URLs from final evaluation
    if wandb_run and hasattr(results, "replay_urls") and results.replay_urls:
        upload_replay_html(
            replay_urls=results.replay_urls,
            agent_step=agent_step,
            epoch=epoch,
            wandb_run=wandb_run,
        )

# Force final saves - all ranks must participate
if is_master:
    saved_record = checkpoint_manager.save_policy(
        policy=agent,
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
            run_dir=dirs.run_dir,
            kickstarter=kickstarter,
            force=True,
        )

# All ranks must synchronize after final save operations
if torch.distributed.is_initialized():
    torch.distributed.barrier()

# Close environment
env.close()  # type: ignore

logger.info(f"\nTraining run complete! Run saved to: {dirs.run_dir}")

# Clean up distributed training if initialized
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

# Clean up wandb if initialized
if is_master and wandb_ctx:
    wandb_ctx.__exit__(None, None, None)
