#!/usr/bin/env -S uv run

import configparser
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pufferlib
import pufferlib.vector
import torch
import torch.optim as optim
from gymnasium.spaces import Discrete, MultiDiscrete
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

# Add the parent directory to path so we can import from metta
sys.path.append(str(Path(__file__).parent.parent))

from dominiks_training.agent import ActorCriticAgent
from dominiks_training.env_config import create_simple_arena_config
from metta.mettagrid import MettaGridEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum


def load_config(config_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def collect_rollout(
    vecenv, agent: ActorCriticAgent, num_steps: int, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect rollout experience from vectorized environments for GAE actor-critic."""
    total_agents = vecenv.num_agents  # This is num_envs * agents_per_env
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    values: list[np.ndarray] = []
    next_values: list[np.ndarray] = []
    log_probs: list[np.ndarray] = []

    # Get initial observations
    obs, _ = vecenv.reset()

    for _ in range(num_steps):
        # obs shape: (total_agents, *obs_shape)
        # Flatten each agent's observation independently
        flat_obs = obs.reshape(total_agents, -1)  # Shape: (total_agents, obs_dim)
        observations.append(flat_obs.copy())

        # Get actions and values from actor-critic agent
        obs_tensor = torch.from_numpy(flat_obs).float().to(device)
        with torch.no_grad():
            action_logits, value = agent(obs_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).squeeze(-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)).squeeze(-1) + 1e-8)

        actions.append(action.cpu().numpy())
        values.append(value.squeeze().cpu().numpy())
        log_probs.append(log_prob.cpu().numpy())

        # PufferLib expects actions in format: (total_agents, action_dims)
        action_array = action.cpu().numpy()
        if hasattr(vecenv.single_action_space, "nvec"):
            # MultiDiscrete action space
            num_action_dims = len(vecenv.single_action_space.nvec)
            actions_2d = np.zeros((total_agents, num_action_dims), dtype=np.int32)
            actions_2d[:, 0] = action_array  # Use first action dimension
        else:
            # Discrete action space
            actions_2d = action_array.reshape(-1, 1).astype(np.int32)

        # Step environments
        obs, reward, terminated, truncated, info = vecenv.step(actions_2d)

        # reward shape: (total_agents,)
        rewards.append(reward.copy())

        # Get next state values for GAE
        next_flat_obs = obs.reshape(total_agents, -1)
        next_obs_tensor = torch.from_numpy(next_flat_obs).float().to(device)
        with torch.no_grad():
            _, next_value = agent(next_obs_tensor)
            next_values.append(next_value.squeeze().cpu().numpy())

    # Convert lists to arrays and reshape for processing
    obs_array = np.array(observations).swapaxes(0, 1)  # (total_agents, num_steps, obs_dim)
    actions_array = np.array(actions).swapaxes(0, 1)  # (total_agents, num_steps)
    rewards_array = np.array(rewards).swapaxes(0, 1)  # (total_agents, num_steps)
    values_array = np.array(values).swapaxes(0, 1)  # (total_agents, num_steps)
    next_values_array = np.array(next_values).swapaxes(0, 1)  # (total_agents, num_steps)
    log_probs_array = np.array(log_probs).swapaxes(0, 1)  # (total_agents, num_steps)

    # Flatten all arrays for batch processing
    obs_flat = obs_array.reshape(-1, obs_array.shape[-1])
    actions_flat = actions_array.flatten()
    rewards_flat = rewards_array.flatten()
    values_flat = values_array.flatten()
    next_values_flat = next_values_array.flatten()
    log_probs_flat = log_probs_array.flatten()

    return obs_flat, actions_flat, rewards_flat, values_flat, next_values_flat, log_probs_flat


def create_agent(obs_dim: int, action_dim: int, device: str = "cpu") -> ActorCriticAgent:
    """Create and initialize an actor-critic agent."""
    agent = ActorCriticAgent(obs_dim, action_dim)
    return agent.to(device)


def make_env_func(curriculum, **kwargs):
    """Simple environment creation function for PufferLib vectorization."""
    return MettaGridEnv(curriculum=curriculum)


def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted returns."""
    returns = np.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, next_values: np.ndarray, gamma: float, gae_lambda: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards for each timestep [T]
        values: Value estimates for each state [T]
        next_values: Value estimates for next states [T]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (bias-variance tradeoff)

    Returns:
        advantages: GAE advantages [T]
        returns: Value targets (advantages + values) [T]
    """
    advantages = np.zeros_like(rewards)
    gae = 0.0

    # Compute GAE backwards through time
    for t in reversed(range(len(rewards))):
        # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_values[t] - values[t]

        # GAE: A_t = δ_t + γλA_{t+1}
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    # Value targets: V_target = A_t + V(s_t)
    returns = advantages + values

    # Normalize advantages for stability
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def compute_actor_critic_loss(
    agent: ActorCriticAgent,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute actor-critic loss."""
    # Forward pass
    action_logits, values = agent(obs)
    action_probs = torch.softmax(action_logits, dim=-1)

    # Actor loss (policy gradient with advantages)
    new_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1) + 1e-8)
    actor_loss = -(new_log_probs * advantages).mean()

    # Critic loss (value function)
    critic_loss = torch.nn.functional.mse_loss(values.squeeze(), returns)

    return actor_loss, critic_loss


def init_wandb(config: configparser.ConfigParser) -> bool:
    """Initialize wandb if enabled and available."""
    if not config.getboolean("wandb", "enabled", fallback=False):
        return False

    if wandb is None:
        print("Warning: wandb is not installed. Logging disabled.")
        return False

    # Generate run name with dff_ prefix and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"dff_{timestamp}"

    wandb.init(
        project=config.get("wandb", "project"),
        entity=config.get("wandb", "entity"),
        name=run_name,
        config={
            "total_timesteps": config.getint("training", "total_timesteps"),
            "learning_rate": config.getfloat("training", "learning_rate"),
            "gamma": config.getfloat("training", "gamma"),
            "gae_lambda": config.getfloat("training", "gae_lambda"),
        },
    )
    return True


def log_wandb_metrics(
    episode: int,
    timestep: int,
    episode_reward: float,
    episode_length: float,
    actor_loss: float,
    critic_loss: float,
    total_loss: float,
    sps: float,
) -> None:
    """Log metrics to wandb if initialized."""
    if wandb is None or wandb.run is None:
        return

    wandb.log(
        {
            "episode": episode,
            "timestep": timestep,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "total_loss": total_loss,
            "sps": sps,
        },
        step=timestep,
    )


def finish_wandb() -> None:
    """Finish wandb run if active."""
    if wandb is not None and wandb.run is not None:
        wandb.finish()


def train() -> None:
    # Load configuration
    config_path = Path(__file__).parent / "config.ini"
    config = load_config(str(config_path))

    # Training parameters
    total_timesteps = config.getint("training", "total_timesteps")
    learning_rate = config.getfloat("training", "learning_rate")
    gamma = config.getfloat("training", "gamma")
    gae_lambda = config.getfloat("training", "gae_lambda")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Environment parameters
    num_envs = config.getint("environment", "num_envs")
    num_workers = config.getint("environment", "num_workers")
    vectorization = config.get("environment", "vectorization")
    rollout_steps = 128  # Number of steps to collect per rollout

    log_interval = config.getint("logging", "log_interval")
    checkpoint_interval = config.getint("logging", "checkpoint_interval")

    print(f"Checkpoint interval: {checkpoint_interval} timesteps")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create vectorized environment
    task_cfg: DictConfig = cast(DictConfig, OmegaConf.create(create_simple_arena_config()))
    curriculum = SingleTaskCurriculum("arena_simple", task_cfg)

    # Determine vectorization backend
    if vectorization == "serial" or num_workers == 1:
        backend = pufferlib.vector.Serial
    elif vectorization == "multiprocessing":
        backend = pufferlib.vector.Multiprocessing
    else:
        backend = pufferlib.vector.Serial  # Default to serial

    # Create vectorized environment
    vecenv = pufferlib.vector.make(
        make_env_func,
        env_kwargs={"curriculum": curriculum},
        backend=backend,
        num_envs=num_envs,
        num_workers=num_workers,
        batch_size=num_envs,
    )

    # Get observation and action dimensions from a single environment
    sample_obs, _ = vecenv.reset()
    print(f"Sample obs shape: {sample_obs.shape}")
    # For vectorized multi-agent environments, PufferLib gives us (total_agents, *obs_shape)
    # where total_agents = num_envs * agents_per_env
    # Each agent gets its own observation, so obs_dim is the flattened single agent obs
    single_agent_obs = sample_obs[0]  # Take first agent
    obs_dim = single_agent_obs.flatten().shape[0]  # Flatten single agent observation

    # Handle MultiDiscrete action space
    space = vecenv.single_action_space
    if isinstance(space, MultiDiscrete):
        action_dim = int(space.nvec[0])  # Use first action dimension
    elif isinstance(space, Discrete):
        action_dim = int(space.n)
    else:
        raise TypeError("Unsupported action space type")

    print(f"Number of environments: {num_envs}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Vectorization: {vectorization}")
    print(f"Action space: {vecenv.action_space}")
    print(f"Single action space: {vecenv.single_action_space}")

    # Create agent and optimizer
    agent = create_agent(obs_dim, action_dim, device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    # Training storage
    episode_rewards = []
    episode_lengths = []

    # Initialize wandb
    wandb_enabled = init_wandb(config)
    if wandb_enabled:
        print("WandB logging enabled")
    else:
        print("WandB logging disabled")

    print("Starting training...")
    print(f"Using device: {device}")

    rollout = 0
    timestep = 0
    last_checkpoint_timestep = 0
    start_time = time.time()

    while timestep < total_timesteps:
        # Collect rollout with vectorized environments
        observations, actions, rewards, values, next_values, log_probs = collect_rollout(
            vecenv, agent, rollout_steps, device
        )

        rollout_timesteps = len(observations)
        timestep += rollout_timesteps
        rollout += 1

        # Compute GAE advantages and returns
        advantages, returns = compute_gae(rewards, values, next_values, gamma, gae_lambda)

        # Convert to tensors
        obs_tensor = torch.from_numpy(observations).float().to(device)
        actions_tensor = torch.from_numpy(actions).long().to(device)
        advantages_tensor = torch.from_numpy(advantages).float().to(device)
        returns_tensor = torch.from_numpy(returns).float().to(device)
        old_log_probs_tensor = torch.from_numpy(log_probs).float().to(device)

        # Compute actor-critic loss and update
        optimizer.zero_grad()
        actor_loss, critic_loss = compute_actor_critic_loss(
            agent, obs_tensor, actions_tensor, advantages_tensor, returns_tensor, old_log_probs_tensor
        )
        total_loss = actor_loss + 0.5 * critic_loss  # Combine losses
        total_loss.backward()
        optimizer.step()

        # Store metrics for logging
        rollout_reward = rewards.mean()
        episode_rewards.append(rollout_reward)
        episode_lengths.append(rollout_steps)

        # Checkpoint - only check at rollout boundaries
        if timestep - last_checkpoint_timestep >= checkpoint_interval:
            checkpoint_path = Path(__file__).parent / f"checkpoint_{timestep}.pt"
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "rollout": rollout,
                    "timestep": timestep,
                },
                checkpoint_path,
            )
            steps_since_last = timestep - last_checkpoint_timestep
            print(f"Saved checkpoint: {checkpoint_path} (after {steps_since_last} steps)")
            last_checkpoint_timestep = timestep

        # Logging
        if rollout % log_interval == 0:
            elapsed_time = time.time() - start_time
            sps = timestep / elapsed_time if elapsed_time > 0 else 0
            avg_reward = float(np.mean(episode_rewards[-log_interval:]))
            avg_length = float(np.mean(episode_lengths[-log_interval:]))
            print(f"Rollout {rollout}, Timestep {timestep}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Actor Loss: {actor_loss.item():.6f}")
            print(f"  Critic Loss: {critic_loss.item():.6f}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  SPS (Steps/sec): {sps:.1f}")
            print(f"  Elapsed: {elapsed_time:.1f}s")

            # Log to wandb
            log_wandb_metrics(
                episode=rollout,
                timestep=timestep,
                episode_reward=avg_reward,
                episode_length=avg_length,
                actor_loss=actor_loss.item(),
                critic_loss=critic_loss.item(),
                total_loss=total_loss.item(),
                sps=sps,
            )

    print("Training completed!")

    # Save final model
    final_path = Path(__file__).parent / "final_model.pt"
    torch.save(agent.state_dict(), final_path)
    print(f"Saved final model: {final_path}")

    # Finish wandb
    finish_wandb()


if __name__ == "__main__":
    train()
