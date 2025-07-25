#!/usr/bin/env -S uv run
"""Refactored run.py using component-based architecture."""

import logging
import os
import time

import torch
import torch.distributed
from omegaconf import DictConfig, OmegaConf

from metta.agent.policy_store import PolicyStore
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.common.wandb.wandb_context import WandbContext
from metta.interface.agent import create_or_load_agent
from metta.interface.directories import save_experiment_config, setup_run_directories
from metta.mettagrid import mettagrid_c  # noqa: F401
from metta.rl.components import (
    EnvironmentManager,
    EvaluationManager,
    OptimizerManager,
    RolloutManager,
    StatsManager,
    TrainingManager,
)
from metta.rl.experience import Experience
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.torch_profiler import TorchProfiler
from metta.rl.trainer_checkpoint import TrainerCheckpoint
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
from metta.rl.util.distributed import setup_device_and_distributed
from metta.rl.util.optimization import maybe_update_l2_weights
from metta.rl.util.policy_management import (
    cleanup_old_policies,
    save_policy_with_metadata,
    validate_policy_environment_match,
    wrap_agent_distributed,
)
from metta.rl.util.rollout import get_lstm_config
from metta.rl.util.stats import compute_timing_stats
from metta.rl.util.utils import check_abort, should_run
from metta.rl.wandb import log_model_parameters, setup_wandb_metrics, upload_env_configs

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set up directories
dirs = setup_run_directories()

# Set up device and distributed training
device, is_master, world_size, rank = setup_device_and_distributed("cuda" if torch.cuda.is_available() else "cpu")

# Configuration using individual component configs
# Core training parameters
num_workers = 4
total_timesteps = 10_000_000
batch_size = 65536 if torch.cuda.is_available() else 16384
minibatch_size = 16384 if torch.cuda.is_available() else 4096
curriculum = "/env/mettagrid/curriculum/navigation/bucketed"
bptt_horizon = 64
update_epochs = 1
forward_pass_minibatch_target_size = 4096 if torch.cuda.is_available() else 2048

# Adjust defaults based on vectorization mode
vectorization_mode = "multiprocessing"
if vectorization_mode == "serial":
    async_factor = 1
    zero_copy = False
else:
    async_factor = 2
    zero_copy = True

grad_mean_variance_interval = 150
scale_batches_by_world_size = False
cpu_offload = False

# Individual component configs
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
    interval_epochs=0,
    profile_dir=os.path.join(dirs.run_dir, "torch_traces"),
)

prioritized_replay_config = PrioritizedExperienceReplayConfig()
vtrace_config = VTraceConfig()
kickstart_config = KickstartConfig()

# Check for initial policy URI from environment variable
initial_policy_uri = os.environ.get("INITIAL_POLICY_URI", None)
initial_policy_config = InitialPolicyConfig(uri=initial_policy_uri)

# Create trainer config
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

# Create environment manager
env_manager = EnvironmentManager(trainer_config, device)
env = env_manager.create_environment(vectorization=vectorization_mode)
metta_grid_env = env_manager.driver_env

# WandB initialization
wandb_run = None
wandb_ctx = None
if is_master:
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
            "seed": 1,
            "trainer": trainer_config.model_dump(),
            "train_job": {"evals": {}},
            "wandb": wandb_config,
        }
    )

    wandb_ctx = WandbContext(wandb_config, global_config)
    wandb_run = wandb_ctx.__enter__()

# Create policy store
policy_store_config = {
    "device": str(device),
    "policy_cache_size": 10,
    "run": dirs.run_name,
    "run_dir": dirs.run_dir,
    "vectorization": vectorization_mode,
    "trainer": trainer_config.model_dump(),
}

if wandb_run and wandb_ctx:
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
        pass

policy_store = PolicyStore(
    DictConfig(policy_store_config),
    wandb_run=wandb_run,
)

# Create or load agent
agent, policy_record, agent_step, epoch, checkpoint = create_or_load_agent(
    env=env,
    run_dir=dirs.run_dir,
    policy_store=policy_store,
    trainer_config=trainer_config,
    device=device,
    is_master=is_master,
    rank=rank,
)

# Get LSTM config
hidden_size, num_lstm_layers = get_lstm_config(agent)

# Validate policy matches environment
validate_policy_environment_match(agent, metta_grid_env)

# Store initial policy record
initial_policy_record = policy_record
initial_policy_uri = policy_record.uri if policy_record else None
initial_generation = policy_record.metadata.get("generation", 0) if policy_record else 0

# Create optimizer manager and optimizer
optimizer_manager = OptimizerManager(trainer_config.optimizer, device)
optimizer = optimizer_manager.create_optimizer(agent)
optimizer_manager.load_state_from_checkpoint(optimizer, checkpoint)

# Wrap agent for distributed training
agent = wrap_agent_distributed(agent, device)

if torch.distributed.is_initialized():
    torch.distributed.barrier()

# Set up wandb metrics and log model parameters
if wandb_run and is_master:
    setup_wandb_metrics(wandb_run)
    log_model_parameters(agent, wandb_run)

# Log to console
if is_master:
    num_params = sum(p.numel() for p in agent.parameters())
    logger.info(f"Model has {num_params:,} parameters")

# Create experience buffer
experience = Experience(
    total_agents=env_manager.num_agents,
    batch_size=trainer_config.batch_size,
    bptt_horizon=trainer_config.bptt_horizon,
    minibatch_size=trainer_config.minibatch_size,
    max_minibatch_size=trainer_config.minibatch_size,
    obs_space=env.single_observation_space,
    atn_space=env.single_action_space,
    device=device,
    hidden_size=hidden_size,
    cpu_offload=trainer_config.cpu_offload,
    num_lstm_layers=num_lstm_layers,
    agents_per_batch=getattr(env, "agents_per_batch", None),
)

# Create kickstarter
kickstarter = Kickstarter(
    trainer_config.kickstart,
    str(device),
    policy_store,
    metta_grid_env,
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
        external_timer=timer,
    )

# Create component managers
rollout_manager = RolloutManager(env, device, timer)
training_manager = TrainingManager(trainer_config, device, kickstarter)
stats_manager = StatsManager(trainer_config, timer, is_master, system_monitor, memory_monitor)
evaluation_manager = EvaluationManager(trainer_config, policy_store, device, dirs.stats_dir, is_master)

# Track policy records
latest_saved_policy_record = policy_record
current_policy_generation = initial_generation + 1 if policy_record else 0

# Upload environment configs
if is_master and wandb_run:
    curr_obj = env_manager.get_curriculum()
    if curr_obj is not None and hasattr(curr_obj, "get_env_cfg_by_bucket"):
        env_configs = curr_obj.get_env_cfg_by_bucket()
        upload_env_configs(env_configs=env_configs, wandb_run=wandb_run)

# Create torch profiler
torch_profiler = TorchProfiler(is_master, trainer_config.profiler, wandb_run, dirs.run_dir)

# Training loop
logger.info(f"Starting training on {device}")
training_start_time = time.time()

while agent_step < trainer_config.total_timesteps:
    steps_before = agent_step

    # ===== ROLLOUT PHASE =====
    rollout_start = time.time()
    raw_infos, agent_step = rollout_manager.collect_rollouts(agent, experience, agent_step)
    rollout_time = time.time() - rollout_start

    # Process rollout statistics
    stats_manager.process_rollout_stats(raw_infos)

    # ===== TRAINING PHASE =====
    train_start = time.time()
    training_manager.train_on_experience(agent, optimizer, experience, losses, epoch, agent_step)
    train_time = time.time() - train_start
    epoch += 1

    torch_profiler.on_epoch_end(epoch)

    # ===== STATS PROCESSING PHASE =====
    stats_start = time.time()

    # Compute timing stats
    timing_info = compute_timing_stats(timer=timer, agent_step=agent_step)

    # Build and log stats
    if is_master:
        current_lr = optimizer_manager.get_current_lr(optimizer)

        # Compute weight stats if configured
        weight_stats = stats_manager.compute_weight_stats(agent, epoch)

        # Build complete stats
        all_stats = stats_manager.build_training_stats(
            losses=losses,
            experience=experience,
            kickstarter=kickstarter,
            agent_step=agent_step,
            epoch=epoch,
            current_lr=current_lr,
            current_policy_generation=current_policy_generation,
            timing_info=timing_info,
        )

        # Add weight stats
        all_stats.update(weight_stats)

        # Log to wandb
        if wandb_run:
            wandb_run.log(all_stats, step=agent_step)

    # Clear stats for next iteration
    stats_manager.clear_stats()
    stats_time = time.time() - stats_start

    # Log progress
    steps_calculated = agent_step - steps_before
    total_time = train_time + rollout_time + stats_time
    steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

    train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
    rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0
    stats_pct = (stats_time / total_time) * 100

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

    # Record heartbeat
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

    # Compute gradient statistics
    stats_manager.compute_gradient_stats(agent, epoch)

    # Save checkpoint periodically
    if should_run(epoch, trainer_config.checkpoint.checkpoint_interval, True):
        if is_master:
            # Create temporary initial_policy_record for save_policy_with_metadata
            temp_initial_policy_record = None
            if initial_policy_uri:
                temp_initial_policy_record = type(
                    "obj",
                    (object,),
                    {
                        "uri": initial_policy_uri,
                        "metadata": {"generation": initial_generation},
                    },
                )()

            saved_record = save_policy_with_metadata(
                policy=agent,
                policy_store=policy_store,
                epoch=epoch,
                agent_step=agent_step,
                evals=stats_manager.eval_scores,
                timer=timer,
                initial_policy_record=temp_initial_policy_record,
                run_name=dirs.run_name,
                is_master=is_master,
            )

            if saved_record:
                latest_saved_policy_record = saved_record

                # Clean up old policies periodically
                if epoch % 10 == 0:
                    cleanup_old_policies(trainer_config.checkpoint.checkpoint_dir, keep_last_n=5)

        # Save training state
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

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    # Upload policy to wandb
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

    # Abort check
    if is_master and wandb_run and should_run(epoch, 5, True):
        if check_abort(wandb_run, trainer_config, agent_step):
            break

    # Policy evaluation
    if evaluation_manager.should_evaluate(epoch) and latest_saved_policy_record:
        eval_scores = evaluation_manager.evaluate_policy(
            latest_saved_policy_record,
            epoch,
            env_manager.get_curriculum(),
            wandb_run,
        )
        stats_manager.update_eval_scores(eval_scores)

        # Generate replay
        evaluation_manager.generate_replay(
            latest_saved_policy_record,
            epoch,
            env_manager.get_curriculum(),
            wandb_run,
        )

# Training complete
total_elapsed = time.time() - training_start_time
logger.info("Training complete!")
logger.info(f"Total training time: {total_elapsed:.1f}s")
logger.info(f"Final epoch: {epoch}, Total steps: {agent_step}")

# Stop monitoring
if is_master:
    if system_monitor:
        system_monitor.stop()
    if memory_monitor:
        memory_monitor.clear()

# Final evaluation if needed
if evaluation_manager.final_evaluation_needed(epoch) and latest_saved_policy_record:
    eval_scores = evaluation_manager.evaluate_policy(
        latest_saved_policy_record,
        epoch,
        env_manager.get_curriculum(),
        wandb_run,
    )
    stats_manager.update_eval_scores(eval_scores)

# Save final checkpoint
if is_master:
    temp_initial_policy_record = None
    if initial_policy_uri:
        temp_initial_policy_record = type(
            "obj",
            (object,),
            {"uri": initial_policy_uri, "metadata": {"generation": initial_generation}},
        )()

    saved_record = save_policy_with_metadata(
        policy=agent,
        policy_store=policy_store,
        epoch=epoch,
        agent_step=agent_step,
        evals=stats_manager.eval_scores,
        timer=timer,
        initial_policy_record=temp_initial_policy_record,
        run_name=dirs.run_name,
        is_master=is_master,
    )

    if saved_record:
        latest_saved_policy_record = saved_record

# Save final training state
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

if torch.distributed.is_initialized():
    torch.distributed.barrier()

# Close environment
env_manager.close()

logger.info(f"\nTraining run complete! Run saved to: {dirs.run_dir}")

# Clean up distributed training
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

# Clean up wandb
if is_master and wandb_ctx:
    wandb_ctx.__exit__(None, None, None)
