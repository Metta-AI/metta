#!/usr/bin/env python3
"""
Functional Training Demo for Metta - No Hydra, Pure Python

This demonstrates how to train a Metta agent using a functional approach,
creating all objects directly in Python without any framework magic.
"""

# Add metta to path for demo purposes
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

from metta.agent.lib.action import ActionEmbedding
from metta.agent.lib.actor import MettaActorSingleHead
from metta.agent.lib.lstm import LSTM
from metta.agent.lib.nn_layer_library import Conv2d, Flatten, Linear, ReLU
from metta.agent.lib.obs_token_to_box_shaper import ObsTokenToBoxShaper
from metta.agent.lib.observation_normalizer import ObservationNormalizer
from metta.agent.metta_agent import MettaAgent
from metta.agent.policy_store import PolicyStore
from metta.common.stopwatch import Stopwatch
from metta.mettagrid.curriculum.util import SamplingCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.rl.experience import Experience
from metta.rl.functional_trainer import rollout, train_ppo
from metta.rl.kickstarter import Kickstarter
from metta.rl.losses import Losses
from metta.rl.trainer_checkpoint import TrainerCheckpoint
from metta.rl.vecenv import make_vecenv
from metta.util.config import config_from_path
from metta.util.logging import setup_mettagrid_logger

# Ensure pufferlib is available
try:
    from pufferlib import _C  # noqa: F401
except ImportError:
    raise ImportError("Failed to import C/CUDA advantage kernel. Please install pufferlib.") from None

torch.set_float32_matmul_precision("high")


# Pydantic models for configuration
class PPOConfig(BaseModel):
    """PPO algorithm configuration."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    norm_adv: bool = True
    clip_vloss: bool = True
    vf_clip_coef: float = 0.1
    update_epochs: int = 4
    target_kl: float | None = None
    l2_reg_loss_coef: float = 0.0
    l2_init_loss_coef: float = 0.0
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0


class TrainingConfig(BaseModel):
    """Training configuration."""

    total_timesteps: int = 100000
    batch_size: int = 2048
    minibatch_size: int = 256
    bptt_horizon: int = 64
    learning_rate: float = 0.0003
    checkpoint_interval: int = 10
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EnvironmentConfig(BaseModel):
    """Environment configuration."""

    num_envs: int = 8
    num_workers: int = 1
    env_config_path: str = "configs/env/mettagrid/curriculum/simple.yaml"
    desync_episodes: bool = True


def create_agent(device: torch.device) -> MettaAgent:
    """Create a MettaAgent directly with all components."""
    # Create component instances directly
    components = {
        "_obs_": ObsTokenToBoxShaper(name="_obs_"),
        "obs_normalizer": ObservationNormalizer(name="obs_normalizer", sources=[{"name": "_obs_"}]),
        "cnn1": Conv2d(
            name="cnn1",
            sources=[{"name": "obs_normalizer"}],
            nn_params={"out_channels": 32, "kernel_size": 3, "stride": 1},
        ),
        "obs_flattener": Flatten(name="obs_flattener", sources=[{"name": "cnn1"}]),
        "fc1": Linear(name="fc1", sources=[{"name": "obs_flattener"}], nn_params={"out_features": 128}),
        "_core_": LSTM(name="_core_", sources=[{"name": "fc1"}], output_size=128, nn_params={"num_layers": 2}),
        "core_relu": ReLU(name="core_relu", sources=[{"name": "_core_"}]),
        "critic_1": Linear(
            name="critic_1", sources=[{"name": "core_relu"}], nn_params={"out_features": 256}, nonlinearity="nn.Tanh"
        ),
        "_value_": Linear(
            name="_value_", sources=[{"name": "critic_1"}], nn_params={"out_features": 1}, nonlinearity=None
        ),
        "actor_1": Linear(name="actor_1", sources=[{"name": "core_relu"}], nn_params={"out_features": 256}),
        "_action_embeds_": ActionEmbedding(
            name="_action_embeds_", nn_params={"num_embeddings": 100, "embedding_dim": 16}
        ),
        "_action_": MettaActorSingleHead(name="_action_", sources=[{"name": "actor_1"}, {"name": "_action_embeds_"}]),
    }

    # Create agent with components
    agent = MettaAgent(
        obs_space=None,  # Will be set later
        obs_width=11,
        obs_height=11,
        action_space=None,  # Will be set later
        feature_normalizations={},
        device=str(device),
        analyze_weights_interval=0,
        clip_range=0,
        l2_init_weight_update_interval=0,
        observations={"obs_key": "grid_obs"},
        components=components,
    )

    return agent


def create_environment(env_config: EnvironmentConfig):
    """Create the vectorized environment directly."""
    # Load environment configuration
    env_cfg = config_from_path(env_config.env_config_path)

    # Create curriculum
    env_overrides = {"desync_episodes": env_config.desync_episodes}
    curriculum = SamplingCurriculum(env_config.env_config_path, env_overrides)

    # Get environment parameters
    task_cfg = curriculum.get_task().env_cfg()
    num_agents = task_cfg.game.num_agents
    batch_size = env_config.num_envs // num_agents

    # Create vectorized environment
    vecenv = make_vecenv(
        curriculum,
        "multiprocessing",
        num_envs=env_config.num_envs,
        batch_size=batch_size,
        num_workers=env_config.num_workers,
        zero_copy=True,
    )

    return vecenv, curriculum


def functional_train_loop(
    agent: MettaAgent,
    vecenv: Any,
    ppo_config: PPOConfig,
    training_config: TrainingConfig,
    checkpoint_dir: Path,
    logger: Any,
):
    """Pure functional training loop."""
    device = torch.device(training_config.device)

    # Reset environment with seed
    vecenv.async_reset(training_config.seed)

    # Get driver environment
    metta_grid_env = vecenv.driver_env
    assert isinstance(metta_grid_env, MettaGridEnv)

    # Now we can properly initialize the agent with environment info
    import gymnasium as gym

    obs_space = gym.spaces.Dict(
        {
            "grid_obs": metta_grid_env.single_observation_space,
            "global_vars": gym.spaces.Box(low=-np.inf, high=np.inf, shape=[0], dtype=np.int32),
        }
    )

    # Reinitialize agent with proper spaces
    agent.obs_space = obs_space
    agent.action_space = metta_grid_env.single_action_space
    agent.feature_normalizations = metta_grid_env.feature_normalizations

    # Setup components (this finalizes the agent initialization)
    agent._setup_components(agent.components["_value_"])
    agent._setup_components(agent.components["_action_"])

    # Move to device and activate actions
    agent = agent.to(device)
    agent.activate_actions(metta_grid_env.action_names, metta_grid_env.max_action_args, device)

    # Create experience buffer
    experience = Experience(
        total_agents=vecenv.num_agents,
        batch_size=training_config.batch_size,
        bptt_horizon=training_config.bptt_horizon,
        minibatch_size=training_config.minibatch_size,
        max_minibatch_size=training_config.minibatch_size,
        obs_space=vecenv.single_observation_space,
        atn_space=vecenv.single_action_space,
        device=device,
        hidden_size=agent.hidden_size,
        cpu_offload=False,
        num_lstm_layers=2,
        agents_per_batch=getattr(vecenv, "agents_per_batch", None),
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        agent.parameters(),
        lr=training_config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Load checkpoint if exists
    checkpoint = TrainerCheckpoint.load(str(checkpoint_dir.parent))
    agent_step = checkpoint.agent_step
    epoch = checkpoint.epoch

    if checkpoint.agent_step > 0:
        try:
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
            logger.info("Loaded optimizer state from checkpoint")
        except Exception as e:
            logger.warning(f"Could not load optimizer state: {e}")

    # Create losses tracker and timer
    losses = Losses()
    timer = Stopwatch(logger)
    timer.start()

    # Minimal policy store and kickstarter (required by train_ppo)
    minimal_cfg = {"run_dir": str(checkpoint_dir.parent), "device": training_config.device}
    policy_store = PolicyStore(minimal_cfg, wandb_run=None)
    kickstarter = Kickstarter(minimal_cfg, policy_store, metta_grid_env.action_names, metta_grid_env.max_action_args)

    logger.info("Starting pure functional training loop...")

    # Main training loop - pure functional
    while agent_step < training_config.total_timesteps:
        epoch_start_time = time.time()
        steps_before = agent_step

        # ========== ROLLOUT ==========
        with timer("rollout"):
            agent_step, stats = rollout(
                policy=agent,
                vecenv=vecenv,
                experience=experience,
                device=device,
                agent_step=agent_step,
                timer=timer,
            )

        # ========== TRAIN ==========
        with timer("train"):
            epoch = train_ppo(
                policy=agent,
                optimizer=optimizer,
                experience=experience,
                device=device,
                losses=losses,
                epoch=epoch,
                cfg=minimal_cfg,  # Minimal config for clip_range
                lr_scheduler=None,
                timer=timer,
                kickstarter=kickstarter,
                agent_step=agent_step,
                # PPO parameters
                gamma=ppo_config.gamma,
                gae_lambda=ppo_config.gae_lambda,
                clip_coef=ppo_config.clip_coef,
                ent_coef=ppo_config.ent_coef,
                vf_coef=ppo_config.vf_coef,
                max_grad_norm=ppo_config.max_grad_norm,
                norm_adv=ppo_config.norm_adv,
                clip_vloss=ppo_config.clip_vloss,
                vf_clip_coef=ppo_config.vf_clip_coef,
                update_epochs=ppo_config.update_epochs,
                target_kl=ppo_config.target_kl,
                l2_reg_loss_coef=ppo_config.l2_reg_loss_coef,
                l2_init_loss_coef=ppo_config.l2_init_loss_coef,
                clip_range=0.0,
                vtrace_rho_clip=ppo_config.vtrace_rho_clip,
                vtrace_c_clip=ppo_config.vtrace_c_clip,
                # Training parameters
                total_timesteps=training_config.total_timesteps,
                batch_size=training_config.batch_size,
                prio_alpha=0.0,
                prio_beta0=0.6,
            )

        # Calculate and log metrics
        steps_collected = agent_step - steps_before
        epoch_time = time.time() - epoch_start_time
        steps_per_sec = steps_collected / epoch_time if epoch_time > 0 else 0

        rollout_time = timer.get_last_elapsed("rollout")
        train_time = timer.get_last_elapsed("train")
        total_time = rollout_time + train_time

        train_pct = (train_time / total_time) * 100 if total_time > 0 else 0
        rollout_pct = (rollout_time / total_time) * 100 if total_time > 0 else 0

        # Log stats
        loss_stats = losses.stats()
        logger.info(
            f"Step {agent_step}/{training_config.total_timesteps} | "
            f"Epoch {epoch} | "
            f"{steps_per_sec:.0f} steps/sec "
            f"({train_pct:.0f}% train / {rollout_pct:.0f}% rollout)"
        )

        if stats and "episode/reward" in stats:
            mean_reward = np.mean(stats["episode/reward"])
            logger.info(f"  Mean reward: {mean_reward:.2f}")

        logger.info(
            f"  Losses - Policy: {loss_stats.get('policy_loss', 0):.4f}, "
            f"Value: {loss_stats.get('value_loss', 0):.4f}, "
            f"Entropy: {loss_stats.get('entropy', 0):.4f}"
        )

        # Save checkpoint
        if epoch % training_config.checkpoint_interval == 0:
            save_checkpoint(agent, optimizer, epoch, agent_step, checkpoint_dir, policy_store, logger)

    logger.info("Training complete!")
    vecenv.close()


def save_checkpoint(agent, optimizer, epoch, agent_step, checkpoint_dir, policy_store, logger):
    """Save a checkpoint."""
    checkpoint_dir.mkdir(exist_ok=True)

    # Save policy
    name = f"model_{epoch:04d}.pt"
    path = str(checkpoint_dir / name)

    pr = policy_store.save(name=name, path=path, policy=agent, metadata={"epoch": epoch, "agent_step": agent_step})

    # Save trainer checkpoint
    checkpoint = TrainerCheckpoint(
        agent_step=agent_step,
        epoch=epoch,
        total_agent_step=agent_step,
        optimizer_state_dict=optimizer.state_dict(),
        policy_path=pr.uri if pr else None,
        extra_args={},
    )
    checkpoint.save(str(checkpoint_dir.parent))

    logger.info(f"Saved checkpoint at epoch {epoch}")


def main():
    """Main entry point - no hydra, pure Python."""
    print("=" * 70)
    print("METTA PURE FUNCTIONAL TRAINING DEMO")
    print("=" * 70)
    print()
    print("This demo creates all objects directly in Python without Hydra.")
    print("Training is a simple while loop calling functional rollout/train_ppo.")
    print()

    # Setup logging
    logger = setup_mettagrid_logger("functional_demo")

    # Load configurations (you could also just create these directly)
    ppo_config = PPOConfig()
    training_config = TrainingConfig()
    env_config = EnvironmentConfig()

    # If you want to load from YAML:
    # with open("configs/demo_ppo.yaml") as f:
    #     ppo_config = PPOConfig(**yaml.safe_load(f))

    device = torch.device(training_config.device)
    logger.info(f"Using device: {device}")

    # Create environment
    logger.info("Creating environment...")
    vecenv, curriculum = create_environment(env_config)

    # Create agent
    logger.info("Creating agent...")
    agent = create_agent(device)

    # Setup checkpoint directory
    checkpoint_dir = Path("./demo_checkpoints")

    # Run functional training loop
    functional_train_loop(
        agent=agent,
        vecenv=vecenv,
        ppo_config=ppo_config,
        training_config=training_config,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
    )

    print()
    print("Demo complete! Check ./demo_checkpoints for saved models.")


if __name__ == "__main__":
    main()
