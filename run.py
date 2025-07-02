"""Example of using Metta as a library without Hydra configuration.

This example shows how to use the Metta API to train an agent with full
control over the training loop, using the same components as the main trainer.

Note: If you encounter "Unable to cast Python instance to C++ type" errors,
the mettagrid C++ module may need to be rebuilt:
  cd mettagrid && make clean && make build

This can happen after pulling changes or switching branches.
"""

import logging
import os
import subprocess
import sys

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


def ensure_mettagrid_built():
    """Ensure the mettagrid C++ module is built."""
    mettagrid_dir = os.path.join(os.path.dirname(__file__), "mettagrid")
    build_dir = os.path.join(mettagrid_dir, "build-debug")

    # Check if the build directory exists
    if not os.path.exists(build_dir):
        logger.info("MetaGrid C++ module not found, building...")

        # Check if cmake is available
        try:
            subprocess.run(["cmake", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(
                "cmake is not available. Please install cmake to build the MetaGrid C++ module.\n"
                "On macOS: brew install cmake\n"
                "On Ubuntu: sudo apt-get install cmake"
            )
            sys.exit(1)

        # Build the module
        try:
            logger.info("Configuring MetaGrid build...")
            subprocess.run(["cmake", "--preset", "debug"], cwd=mettagrid_dir, check=True)
            logger.info("Building MetaGrid C++ module...")
            subprocess.run(["cmake", "--build", "build-debug"], cwd=mettagrid_dir, check=True)
            logger.info("MetaGrid C++ module built successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to build MetaGrid C++ module: {e}")
            sys.exit(1)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure mettagrid is built before importing anything that depends on it
ensure_mettagrid_built()

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Create trainer config with Pydantic
trainer_config = create_default_trainer_config(
    num_workers=4,
    total_timesteps=10_000_000,
    batch_size=8192,
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
    num_envs=trainer_config.batch_size // trainer_config.num_workers,
    num_workers=trainer_config.num_workers,
    batch_size=trainer_config.batch_size // trainer_config.num_workers,
    async_factor=trainer_config.async_factor,
    zero_copy=trainer_config.zero_copy,
    is_training=True,
)

# Create agent using DictConfig (as expected by make_policy)
logger.info("Creating agent...")
agent_config = DictConfig(
    {
        "device": str(device),
        "agent": {
            "hidden_dim": 512,
            "lstm_layers": 1,
            "bptt_horizon": 8,
            "use_prev_action": True,
            "use_prev_reward": True,
            "mlp_layers": 2,
            "forward_lstm": True,
            "backbone": "cnn",
            "clip_range": 0,
            "analyze_weights_interval": 300,
            "l2_init_weight_update_interval": 0,
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

# Training loop
logger.info(f"Starting training for {trainer_config.total_timesteps} steps...")

while training.agent_step < trainer_config.total_timesteps:
    # ===== ROLLOUT PHASE =====
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

    # ===== TRAINING PHASE =====
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

    for update_epoch in range(trainer_config.update_epochs):
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
                logger.info(f"Early stopping at epoch {update_epoch} due to KL divergence")
                break

    # Log progress
    if training.epoch % 10 == 0:
        losses_stats = training.losses.stats()
        logger.info(
            f"Epoch {training.epoch} - "
            f"Steps: {training.agent_step}/{trainer_config.total_timesteps} - "
            f"Policy Loss: {losses_stats.get('policy_loss', 0):.4f} - "
            f"Value Loss: {losses_stats.get('value_loss', 0):.4f} - "
            f"Entropy: {losses_stats.get('entropy', 0):.4f}"
        )

    # Save checkpoint periodically
    if training.epoch % trainer_config.checkpoint.checkpoint_interval == 0:
        logger.info(f"Saving checkpoint at epoch {training.epoch}")
        save_checkpoint(
            policy=agent,
            policy_store=policy_store,
            epoch=training.epoch,
            metadata={
                "agent_step": training.agent_step,
                "epoch": training.epoch,
                "stats": dict(training.stats),
            },
        )

    # Clear stats for next iteration
    training.stats.clear()

logger.info("Training complete!")

# Final checkpoint
logger.info("Saving final checkpoint...")
save_checkpoint(
    policy=agent,
    policy_store=policy_store,
    epoch=training.epoch,
    metadata={
        "agent_step": training.agent_step,
        "epoch": training.epoch,
        "final": True,
    },
)

# Close environment - env is the vecenv returned by Environment()
env.close()  # type: ignore
