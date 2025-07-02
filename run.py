"""Example of using Metta as a library without Hydra configuration.

This example shows how to use the Metta API to train an agent with full
control over the training loop, using the same components as the main trainer.
"""

import logging
import sys
import time

import torch
from omegaconf import DictConfig

from metta.agent.policy_store import PolicyStore
from metta.api import (
    Agent,
    Environment,
    TrainingComponents,
    calculate_anneal_beta,
    create_default_trainer_config,
    save_checkpoint,
)
from metta.common.util.heartbeat import record_heartbeat
from metta.rl.functions import calculate_explained_variance
from metta.rl.trainer_checkpoint import TrainerCheckpoint


def _cleanup_old_policies(checkpoint_dir: str, keep_last_n: int = 5):
    """Clean up old saved policies to prevent memory accumulation."""
    try:
        from pathlib import Path

        # Get checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return

        # List all policy files
        policy_files = sorted(checkpoint_path.glob("policy_*.pt"))

        # Keep only the most recent ones
        if len(policy_files) > keep_last_n:
            files_to_remove = policy_files[:-keep_last_n]
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    logger.info(f"Removed old policy file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove old policy file {file_path}: {e}")

    except Exception as e:
        logger.warning(f"Error during policy cleanup: {e}")


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if mettagrid is available
try:
    from metta.mettagrid import mettagrid_c  # noqa: F401
except ImportError:
    logger.error(
        "MetaGrid C++ module not available. Please install the package:\n"
        "  uv sync --inexact\n"
        "or if you don't have uv:\n"
        "  pip install -e ."
    )
    sys.exit(1)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create trainer config with Pydantic
trainer_config = create_default_trainer_config(
    num_workers=4,
    total_timesteps=10_000_000,
    batch_size=16384,  # Increased to accommodate all agents
    minibatch_size=512,
    checkpoint_dir="./checkpoints",
    # Override specific nested configs
    ppo={
        "clip_coef": 0.1,
        "ent_coef": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    },
    optimizer={
        "type": "adam",
        "learning_rate": 3e-4,
    },
)

# Create environment
logger.info("Creating environment...")
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

# Create agent using DictConfig (as expected by make_policy)
logger.info("Creating agent...")
# Create agent configuration based on the fast.yaml example
agent_config = DictConfig(
    {
        "device": str(device),
        "agent": {
            "clip_range": 0,
            "analyze_weights_interval": 300,
            "l2_init_weight_update_interval": 0,
            "observations": {"obs_key": "grid_obs"},
            "components": {
                "_obs_": {
                    "_target_": "metta.agent.lib.obs_token_to_box_shaper.ObsTokenToBoxShaper",
                    "sources": None,
                },
                "obs_normalizer": {
                    "_target_": "metta.agent.lib.observation_normalizer.ObservationNormalizer",
                    "sources": [{"name": "_obs_"}],
                },
                "cnn1": {
                    "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                    "sources": [{"name": "obs_normalizer"}],
                    "nn_params": {
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                },
                "cnn2": {
                    "_target_": "metta.agent.lib.nn_layer_library.Conv2d",
                    "sources": [{"name": "cnn1"}],
                    "nn_params": {
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                },
                "obs_flattener": {
                    "_target_": "metta.agent.lib.nn_layer_library.Flatten",
                    "sources": [{"name": "cnn2"}],
                },
                "encoded_obs": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "obs_flattener"}],
                    "nn_params": {"out_features": 512},
                },
                "_core_": {
                    "_target_": "metta.agent.lib.lstm.LSTM",
                    "sources": [{"name": "encoded_obs"}],
                    "output_size": 512,
                    "nn_params": {
                        "num_layers": 1,
                    },
                },
                "_value_": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "_core_"}],
                    "nn_params": {"out_features": 1},
                    "nonlinearity": None,
                },
                "actor_1": {
                    "_target_": "metta.agent.lib.nn_layer_library.Linear",
                    "sources": [{"name": "_core_"}],
                    "nn_params": {"out_features": 512},
                },
                "_action_embeds_": {
                    "_target_": "metta.agent.lib.action.ActionEmbedding",
                    "sources": None,
                    "nn_params": {
                        "num_embeddings": 100,
                        "embedding_dim": 16,
                    },
                },
                "_action_": {
                    "_target_": "metta.agent.lib.actor.MettaActorSingleHead",
                    "sources": [
                        {"name": "actor_1"},
                        {"name": "_action_embeds_"},
                    ],
                },
            },
        },
    }
)
agent = Agent(env, agent_config, str(device))

# Create policy store for checkpointing
# Create a minimal config for PolicyStore - it needs cfg and wandb_run
policy_store_cfg = DictConfig(
    {
        "device": str(device),
        "policy_cache_size": 10,
        "trainer": {
            "checkpoint": {
                "checkpoint_dir": trainer_config.checkpoint.checkpoint_dir,
            }
        },
    }
)
policy_store = PolicyStore(cfg=policy_store_cfg, wandb_run=None)

# Create training components
logger.info("Initializing training components...")
training = TrainingComponents.create(
    vecenv=env,
    policy=agent,
    trainer_config=trainer_config,
    device=str(device),
    policy_store=policy_store,
)

# Create learning rate scheduler
lr_scheduler = None
if hasattr(trainer_config, "lr_scheduler") and trainer_config.lr_scheduler.enabled:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        training.optimizer, T_max=trainer_config.total_timesteps // trainer_config.batch_size
    )
    logger.info("Created learning rate scheduler")

# Load checkpoint if exists
checkpoint_path = trainer_config.checkpoint.checkpoint_dir
checkpoint = TrainerCheckpoint.load(checkpoint_path) if checkpoint_path else None

if checkpoint:
    logger.info(f"Restoring from checkpoint at {checkpoint.agent_step} steps")
    training.agent_step = checkpoint.agent_step
    training.epoch = checkpoint.epoch

    # Load optimizer state
    if checkpoint.optimizer_state_dict:
        try:
            training.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            logger.info("Successfully loaded optimizer state from checkpoint")
        except ValueError:
            logger.warning("Optimizer state dict doesn't match. Starting with fresh optimizer state.")

# Training loop
logger.info("Starting training")
logger.info(f"Training on {device}")

# Track timing for performance metrics
rollout_time = 0
train_time = 0
epoch_start_time = time.time()
steps_at_epoch_start = training.agent_step

while training.agent_step < trainer_config.total_timesteps:
    steps_before = training.agent_step

    # ===== ROLLOUT PHASE =====
    rollout_start = time.time()
    raw_infos = []
    training.reset_for_rollout()

    # Collect experience
    while not training.is_ready_for_training():
        num_steps, info = training.rollout_step()
        training.agent_step += num_steps

        if info:
            raw_infos.extend(info)

    # Process rollout statistics
    training.accumulate_stats(raw_infos)
    rollout_time = time.time() - rollout_start

    # ===== TRAINING PHASE =====
    train_start = time.time()
    training.reset_training_state()

    # Calculate prioritized replay parameters
    prio_cfg = trainer_config.prioritized_experience_replay
    anneal_beta = calculate_anneal_beta(
        epoch=training.epoch,
        total_timesteps=trainer_config.total_timesteps,
        batch_size=trainer_config.batch_size,
        prio_alpha=prio_cfg.prio_alpha,
        prio_beta0=prio_cfg.prio_beta0,
    )

    # Compute advantages once
    advantages = training.compute_advantages()

    # Train for multiple epochs
    total_minibatches = training.experience.num_minibatches * trainer_config.update_epochs
    minibatch_idx = 0

    for _update_epoch in range(trainer_config.update_epochs):
        for _ in range(training.experience.num_minibatches):
            # Sample minibatch
            minibatch = training.sample_minibatch(
                advantages=advantages,
                minibatch_idx=minibatch_idx,
                total_minibatches=total_minibatches,
                anneal_beta=anneal_beta,
            )

            # Train on minibatch
            loss = training.train_minibatch(minibatch, advantages)

            # Optimize
            training.optimize_step(loss, training.experience.accumulate_minibatches)

            minibatch_idx += 1

        training.epoch += 1

        # Early exit if KL divergence is too high
        if trainer_config.ppo.target_kl is not None:
            average_approx_kl = training.losses.approx_kl_sum / training.losses.minibatches_processed
            if average_approx_kl > trainer_config.ppo.target_kl:
                break

    # Apply additional training steps
    if minibatch_idx > 0:  # Only if we actually trained
        # Weight clipping if enabled
        if hasattr(agent_config.agent, "clip_range") and agent_config.agent.clip_range > 0:
            if hasattr(agent, "clip_weights"):
                agent.clip_weights()

        # CUDA synchronization
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    # Step learning rate scheduler
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Calculate explained variance
    training.losses.explained_variance = calculate_explained_variance(training.experience.values, advantages)

    train_time = time.time() - train_start

    # Calculate performance metrics
    steps_calculated = training.agent_step - steps_before
    total_time = train_time + rollout_time
    steps_per_sec = steps_calculated / total_time if total_time > 0 else 0

    train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
    rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0

    # Log progress similar to trainer.py
    logger.info(
        f"Epoch {training.epoch} - {steps_per_sec:.0f} steps/sec ({train_pct:.0f}% train / {rollout_pct:.0f}% rollout)"
    )

    # Heartbeat recording
    if training.epoch % 10 == 0:
        record_heartbeat()

    # Update L2 weights if configured
    if hasattr(agent_config.agent, "l2_init_weight_update_interval"):
        l2_interval = agent_config.agent.l2_init_weight_update_interval
        if l2_interval > 0 and training.epoch % l2_interval == 0:
            if hasattr(agent, "update_l2_init_weight_copy"):
                agent.update_l2_init_weight_copy()
                logger.info(f"Updated L2 init weights at epoch {training.epoch}")

    # Compute gradient statistics
    if hasattr(trainer_config, "grad_mean_variance_interval"):
        grad_interval = trainer_config.grad_mean_variance_interval
        if grad_interval > 0 and training.epoch % grad_interval == 0:
            all_gradients = []
            for param in agent.parameters():
                if param.grad is not None:
                    all_gradients.append(param.grad.view(-1))

            if all_gradients:
                all_gradients_tensor = torch.cat(all_gradients).to(torch.float32)
                grad_mean = all_gradients_tensor.mean()
                grad_variance = all_gradients_tensor.var()
                grad_norm = all_gradients_tensor.norm(2)

                logger.info(
                    f"Gradient stats at epoch {training.epoch}: "
                    f"mean={grad_mean.item():.2e}, "
                    f"var={grad_variance.item():.2e}, "
                    f"norm={grad_norm.item():.2e}"
                )

    # Save checkpoint periodically
    if training.epoch % trainer_config.checkpoint.checkpoint_interval == 0:
        logger.info(f"Saving policy at epoch {training.epoch}")
        saved_policy_path = save_checkpoint(
            policy=agent,
            policy_store=policy_store,
            epoch=training.epoch,
            metadata={
                "agent_step": training.agent_step,
                "epoch": training.epoch,
                "stats": dict(training.stats),
            },
        )
        logger.info(f"Successfully saved policy at epoch {training.epoch}")

        # Save training state
        logger.info("Saving training state...")
        trainer_checkpoint = TrainerCheckpoint(
            agent_step=training.agent_step,
            epoch=training.epoch,
            total_agent_step=training.agent_step,
            optimizer_state_dict=training.optimizer.state_dict(),
            policy_path=saved_policy_path.uri if hasattr(saved_policy_path, "uri") else None,
            stopwatch_state=None,  # Timer state not implemented in this example
        )
        trainer_checkpoint.save(checkpoint_path)
        logger.info(f"Saved training state at epoch {training.epoch}")

        # Clean up old policies to prevent disk space issues
        if training.epoch % 10 == 0:  # Clean up every 10 epochs
            _cleanup_old_policies(checkpoint_path, keep_last_n=5)

    # Clear stats for next iteration
    training.stats.clear()

# Training complete
total_elapsed = time.time() - epoch_start_time
logger.info("Training complete!")
logger.info(f"Total training time: {total_elapsed:.1f}s")
logger.info(f"Final epoch: {training.epoch}")
logger.info(f"Total steps: {training.agent_step}")

# Log final stats if available
if hasattr(training.losses, "stats"):
    losses_stats = training.losses.stats()
    logger.info(
        f"Final losses - "
        f"Policy: {losses_stats.get('policy_loss', 0):.4f}, "
        f"Value: {losses_stats.get('value_loss', 0):.4f}, "
        f"Entropy: {losses_stats.get('entropy', 0):.4f}, "
        f"Explained Variance: {losses_stats.get('explained_variance', 0):.3f}"
    )

# Log timing breakdown if we tracked it
if hasattr(training, "timer") and hasattr(training.timer, "get_all_summaries"):
    timing_summary = training.timer.get_all_summaries()
    logger.info("Timing breakdown:")
    for name, summary in timing_summary.items():
        logger.info(f"  {name}: {summary['total_elapsed']:.1f}s")
else:
    # Simple timing summary
    avg_steps_per_sec = training.agent_step / total_elapsed if total_elapsed > 0 else 0
    logger.info(f"Average speed: {avg_steps_per_sec:.1f} steps/sec")

# Final checkpoint (force save)
if training.epoch % trainer_config.checkpoint.checkpoint_interval != 0:
    logger.info("Saving final checkpoint...")
    saved_policy_path = save_checkpoint(
        policy=agent,
        policy_store=policy_store,
        epoch=training.epoch,
        metadata={
            "agent_step": training.agent_step,
            "epoch": training.epoch,
            "final": True,
        },
    )
    logger.info("Successfully saved final policy")

    # Save final training state
    final_checkpoint = TrainerCheckpoint(
        agent_step=training.agent_step,
        epoch=training.epoch,
        total_agent_step=training.agent_step,
        optimizer_state_dict=training.optimizer.state_dict(),
        policy_path=saved_policy_path.uri if hasattr(saved_policy_path, "uri") else None,
        stopwatch_state=None,
    )
    final_checkpoint.save(checkpoint_path)
    logger.info("Saved final training state")

# Close environment - env is the vecenv returned by Environment()
env.close()  # type: ignore
