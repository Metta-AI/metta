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
    evaluate_policy_suite,
    generate_replay_simple,
    initialize_wandb,
    load_checkpoint,
    save_experiment_config,
    setup_distributed_training,
    setup_run_directories,
    wrap_agent_distributed,
)
from metta.common.profiling.memory_monitor import MemoryMonitor
from metta.common.profiling.stopwatch import Stopwatch
from metta.common.util.heartbeat import record_heartbeat
from metta.common.util.system_monitor import SystemMonitor
from metta.mettagrid import mettagrid_c  # noqa: F401
from metta.rl.experience import Experience
from metta.rl.functions import (
    accumulate_rollout_stats,
    build_wandb_stats,
    calculate_batch_sizes,
    cleanup_old_policies,
    compute_gradient_stats,
    compute_timing_stats,
    get_lstm_config,
    maybe_update_l2_weights,
    process_stats,
    process_training_stats,
    save_policy_with_metadata,
    save_training_state,
    should_run_on_interval,
)
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer import rollout, train_epoch
from metta.rl.trainer_config import (
    CheckpointConfig,
    OptimizerConfig,
    PPOConfig,
    SimulationConfig,
    TorchProfilerConfig,
    TrainerConfig,
)

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
        replay_interval=300,
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

# Optional WandB integration - set to True to enable
USE_WANDB = False
wandb_run = None

if USE_WANDB and is_master:
    import wandb

    wandb_run = wandb.init(
        project="metta-run-example",
        name=dirs.run_name,
        config=trainer_config.model_dump(),
    )

# Adjust batch sizes for distributed training
if torch.distributed.is_initialized() and trainer_config.scale_batches_by_world_size:
    trainer_config.batch_size = trainer_config.batch_size // world_size
    trainer_config.forward_pass_minibatch_target_size = trainer_config.forward_pass_minibatch_target_size // world_size

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

# Track policy records for consistency with trainer.py
initial_policy_record = None
latest_saved_policy_record = None

# Create initial policy record if we have a loaded policy
if loaded_policy_path:
    initial_policy_record = policy_store.policy_record(loaded_policy_path)
    latest_saved_policy_record = initial_policy_record

# Ensure all ranks have the same initial policy
ensure_initial_policy(agent, policy_store, checkpoint_path, loaded_policy_path, device)
agent = wrap_agent_distributed(agent, device)

# If no initial policy record yet (i.e., new policy was created), create it now
if initial_policy_record is None:
    # The policy was just saved by ensure_initial_policy at epoch 0
    initial_policy_path = os.path.join(checkpoint_path, policy_store.make_model_name(0))
    if os.path.exists(initial_policy_path):
        initial_policy_record = policy_store.policy_record(initial_policy_path)
        latest_saved_policy_record = initial_policy_record

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
    )

# Training loop
logger.info(f"Starting training on {device}")
evaluation_scores = {}
epoch_start_time = time.time()
steps_at_epoch_start = agent_step
stats = defaultdict(list)  # Use defaultdict like trainer.py
grad_stats = {}
initial_policy_record = initial_policy_record  # Already set above
current_policy_generation = initial_policy_record.metadata.get("generation", 0) + 1 if initial_policy_record else 0

while agent_step < trainer_config.total_timesteps:
    steps_before = agent_step

    # ===== ROLLOUT PHASE =====
    with timer("_rollout"):
        num_steps, raw_infos = rollout(
            vecenv=env,
            policy=agent,
            experience=experience,
            device=device,
            timer=timer,
        )
        agent_step += num_steps

        # Process rollout stats
        accumulate_rollout_stats(raw_infos, stats)

    # ===== TRAINING PHASE =====
    with timer("_train"):
        epochs_trained = train_epoch(
            policy=agent,
            optimizer=optimizer.optimizer,  # Pass the actual PyTorch optimizer
            experience=experience,
            kickstarter=kickstarter,
            losses=losses,
            trainer_cfg=trainer_config,
            agent_step=agent_step,
            epoch=epoch,
            device=device,
        )
        epoch += epochs_trained

        # Update learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

    # ===== STATS PROCESSING PHASE =====
    with timer("_process_stats"):
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
            world_size=world_size,
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
                evals=evaluation_scores.get(epoch, {}),
                agent_step=agent_step,
                epoch=epoch,
                world_size=world_size,
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

    # Calculate performance metrics using timer
    rollout_time = timer.get_last_elapsed("_rollout")
    train_time = timer.get_last_elapsed("_train")
    stats_time = timer.get_last_elapsed("_process_stats")
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

    # Process and log stats (console only, no WandB)
    process_stats(
        stats=stats,
        losses=losses,
        evals=evaluation_scores.get(epoch, {}),
        grad_stats=grad_stats,
        experience=experience,
        policy=agent,
        timer=timer,
        trainer_cfg=trainer_config,
        agent_step=agent_step,
        epoch=epoch,
        world_size=world_size,
        wandb_run=wandb_run,  # Optional WandB integration
        memory_monitor=memory_monitor if is_master else None,
        system_monitor=system_monitor if is_master else None,
        latest_saved_policy_record=latest_saved_policy_record,
        initial_policy_record=initial_policy_record,
        optimizer=optimizer,
        kickstarter=kickstarter,
    )

    # Clear stats for next iteration (similar to trainer.py)
    stats.clear()
    grad_stats.clear()

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
    if should_run_on_interval(epoch, trainer_config.grad_mean_variance_interval, is_master):
        grad_stats = compute_gradient_stats(agent)

    # Log system monitoring stats (master only) - removed as process_stats handles this
    # Log memory monitor stats - removed as process_stats handles this

    # Save checkpoint periodically
    if should_run_on_interval(epoch, trainer_config.checkpoint.checkpoint_interval, True):  # All ranks participate
        # Save policy with metadata (master only)
        if is_master:
            saved_record = save_policy_with_metadata(
                policy=agent,
                policy_store=policy_store,
                epoch=epoch,
                agent_step=agent_step,
                evals=evaluation_scores.get(epoch, {}),
                timer=timer,
                vecenv=env,
                initial_policy_record=initial_policy_record,
                run_name=dirs.run_name,
                is_master=is_master,
            )

            if saved_record:
                latest_saved_policy_record = saved_record

                # Clean up old policies periodically
                if epoch % 10 == 0:
                    cleanup_old_policies(checkpoint_path, keep_last_n=5)

        # Save training state (master only)
        latest_uri = latest_saved_policy_record.uri if latest_saved_policy_record else None
        save_training_state(
            checkpoint_dir=dirs.run_dir,
            agent_step=agent_step,
            epoch=epoch,
            optimizer=optimizer,
            timer=timer,
            latest_saved_policy_uri=latest_uri,
            kickstarter=kickstarter,
            world_size=world_size,
            is_master=is_master,
        )

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
        eval_scores = evaluate_policy_suite(
            policy_record=latest_saved_policy_record,
            policy_store=policy_store,
            evaluation_config=evaluation_config,
            device=device,
            vectorization="serial",
            stats_dir=dirs.stats_dir,
            logger=logger,
        )
        evaluation_scores[epoch] = eval_scores

    # Replay generation (master only)
    if (
        is_master
        and trainer_config.simulation.replay_interval > 0
        and epoch % trainer_config.simulation.replay_interval == 0
        and latest_saved_policy_record
    ):
        # Generate replay on the bucketed curriculum environment
        replay_sim_config = create_replay_config("varied_terrain/balanced_medium")
        player_url = generate_replay_simple(
            policy_record=latest_saved_policy_record,
            policy_store=policy_store,
            replay_config=replay_sim_config,
            device=device,
            vectorization="serial",
            replay_dir=dirs.replay_dir,
            epoch=epoch,
            logger=logger,
        )

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
        vecenv=env,
        initial_policy_record=initial_policy_record,
        run_name=dirs.run_name,
        is_master=is_master,
    )

    if saved_record:
        latest_saved_policy_record = saved_record

# Save training state (master only)
latest_uri = latest_saved_policy_record.uri if latest_saved_policy_record else None
save_training_state(
    checkpoint_dir=dirs.run_dir,
    agent_step=agent_step,
    epoch=epoch,
    optimizer=optimizer,
    timer=timer,
    latest_saved_policy_uri=latest_uri,
    kickstarter=kickstarter,
    world_size=world_size,
    is_master=is_master,
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
