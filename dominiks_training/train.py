#!/usr/bin/env -S uv run

import configparser
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.optim as optim
from gymnasium.spaces import Discrete, MultiDiscrete
from omegaconf import DictConfig, OmegaConf

try:
    import wandb
except ImportError:
    wandb = None

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


def collect_episode(
    env: MettaGridEnv, agent: ActorCriticAgent, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect a single episode of experience for GAE actor-critic."""
    observations = []
    actions = []
    rewards = []
    values = []
    next_values = []
    log_probs = []

    obs, _ = env.reset()
    done = False

    while not done:
        # Flatten observation for the agent
        flat_obs = obs.flatten()
        observations.append(flat_obs)

        # Get action and value from actor-critic agent
        obs_tensor = torch.from_numpy(flat_obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_logits, value = agent(obs_tensor)
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).squeeze()
            log_prob = torch.log(action_probs[0, action])

        single_action = action.item()
        actions.append(single_action)
        values.append(value.item())
        log_probs.append(log_prob.item())

        # Create proper 2D action array for multi-agent environment
        if hasattr(env.single_action_space, "nvec"):
            # MultiDiscrete action space - shape should be (num_agents, num_action_dims)
            num_action_dims = len(env.single_action_space.nvec)
            actions_2d = np.zeros((env.num_agents, num_action_dims), dtype=np.int32)
            for i in range(env.num_agents):
                actions_2d[i, 0] = single_action  # Same action for all agents, first action dimension
        else:
            # Regular discrete action space - reshape to 2D
            actions_2d = np.full((env.num_agents, 1), single_action, dtype=np.int32)

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(actions_2d)
        rewards.append(reward.mean())  # Average reward across agents

        # Get next state value for GAE (unless episode is done)
        if not (terminated.any() or truncated.any()):
            next_flat_obs = obs.flatten()
            next_obs_tensor = torch.from_numpy(next_flat_obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                _, next_value = agent(next_obs_tensor)
                next_values.append(next_value.item())
        else:
            # Terminal state has zero value
            next_values.append(0.0)
            done = True

    return (
        np.array(observations),
        np.array(actions),
        np.array(rewards),
        np.array(values),
        np.array(next_values),
        np.array(log_probs),
    )


def create_agent(obs_dim: int, action_dim: int, device: str = "cpu") -> ActorCriticAgent:
    """Create and initialize an actor-critic agent."""
    agent = ActorCriticAgent(obs_dim, action_dim)
    return agent.to(device)


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

    log_interval = config.getint("logging", "log_interval")
    checkpoint_interval = config.getint("logging", "checkpoint_interval")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create environment
    task_cfg: DictConfig = cast(DictConfig, OmegaConf.create(create_simple_arena_config()))
    curriculum = SingleTaskCurriculum("arena_simple", task_cfg)
    env = MettaGridEnv(curriculum=curriculum)

    # Get observation and action dimensions
    obs, _ = env.reset()
    obs_dim = obs.flatten().shape[0]

    # Handle MultiDiscrete action space
    space = env.single_action_space
    if isinstance(space, MultiDiscrete):
        action_dim = int(space.nvec[0])  # Use first action dimension
    elif isinstance(space, Discrete):
        action_dim = int(space.n)
    else:
        raise TypeError("Unsupported action space type")

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action space: {env.action_space}")
    print(f"Single action space: {env.single_action_space}")

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

    episode = 0
    timestep = 0
    start_time = time.time()

    while timestep < total_timesteps:
        # Collect episode with next state values for GAE
        observations, actions, rewards, values, next_values, log_probs = collect_episode(env, agent, device)

        episode_length = len(observations)
        episode_reward = rewards.sum()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        timestep += episode_length
        episode += 1

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

        # Logging
        if episode % log_interval == 0:
            elapsed_time = time.time() - start_time
            sps = timestep / elapsed_time if elapsed_time > 0 else 0
            avg_reward = float(np.mean(episode_rewards[-log_interval:]))
            avg_length = float(np.mean(episode_lengths[-log_interval:]))
            print(f"Episode {episode}, Timestep {timestep}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Actor Loss: {actor_loss.item():.6f}")
            print(f"  Critic Loss: {critic_loss.item():.6f}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  SPS (Steps/sec): {sps:.1f}")
            print(f"  Elapsed: {elapsed_time:.1f}s")

            # Log to wandb
            log_wandb_metrics(
                episode=episode,
                timestep=timestep,
                episode_reward=avg_reward,
                episode_length=avg_length,
                actor_loss=actor_loss.item(),
                critic_loss=critic_loss.item(),
                total_loss=total_loss.item(),
                sps=sps,
            )

        # Checkpoint
        if episode % checkpoint_interval == 0:
            checkpoint_path = Path(__file__).parent / f"checkpoint_{episode}.pt"
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "episode": episode,
                    "timestep": timestep,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training completed!")

    # Save final model
    final_path = Path(__file__).parent / "final_model.pt"
    torch.save(agent.state_dict(), final_path)
    print(f"Saved final model: {final_path}")

    # Finish wandb
    finish_wandb()


if __name__ == "__main__":
    train()
