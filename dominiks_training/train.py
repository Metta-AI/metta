#!/usr/bin/env -S uv run

import configparser
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from omegaconf import OmegaConf

# Add the parent directory to path so we can import from metta
sys.path.append(str(Path(__file__).parent.parent))

from env_config import create_simple_arena_config

from agent import compute_policy_loss, create_agent
from metta.mettagrid import MettaGridEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum


def load_config(config_path: str) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def collect_episode(env: MettaGridEnv, agent, device: str):
    """Collect a single episode of experience."""
    observations = []
    actions = []
    rewards = []

    obs, _ = env.reset()
    done = False

    while not done:
        # Flatten observation for the agent
        flat_obs = obs.flatten()
        observations.append(flat_obs)

        # Get action from agent
        single_action = agent.get_action(flat_obs)[0]  # Get single action

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

        actions.append(single_action)  # Store single action for training

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(actions_2d)
        rewards.append(reward.mean())  # Average reward across agents

        done = terminated.any() or truncated.any()

    return (np.array(observations), np.array(actions), np.array(rewards))


def compute_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute discounted returns."""
    returns = np.zeros_like(rewards)
    G = 0
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G

    # Normalize returns
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def train():
    # Load configuration
    config_path = Path(__file__).parent / "config.ini"
    config = load_config(str(config_path))

    # Training parameters
    total_timesteps = config.getint("training", "total_timesteps")
    learning_rate = config.getfloat("training", "learning_rate")
    gamma = config.getfloat("training", "gamma")
    device = config.get("training", "device")

    log_interval = config.getint("logging", "log_interval")
    checkpoint_interval = config.getint("logging", "checkpoint_interval")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create environment
    env_config = OmegaConf.create(create_simple_arena_config())
    curriculum = SingleTaskCurriculum("arena_simple", env_config)
    env = MettaGridEnv(curriculum=curriculum)

    # Get observation and action dimensions
    obs, _ = env.reset()
    obs_dim = obs.flatten().shape[0]

    # Handle MultiDiscrete action space
    if hasattr(env.single_action_space, "nvec"):
        action_dim = env.single_action_space.nvec[0]  # Use first action dimension
    else:
        action_dim = env.single_action_space.n

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

    print("Starting training...")

    episode = 0
    timestep = 0
    start_time = time.time()

    while timestep < total_timesteps:
        # Collect episode
        observations, actions, rewards = collect_episode(env, agent, device)

        episode_length = len(observations)
        episode_reward = rewards.sum()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        timestep += episode_length
        episode += 1

        # Compute returns
        returns = compute_returns(rewards, gamma)

        # Convert to tensors
        obs_tensor = torch.from_numpy(observations).float().to(device)
        actions_tensor = torch.from_numpy(actions).long().to(device)
        returns_tensor = torch.from_numpy(returns).float().to(device)

        # Compute loss and update
        optimizer.zero_grad()
        loss = compute_policy_loss(agent, obs_tensor, actions_tensor, returns_tensor)
        loss.backward()
        optimizer.step()

        # Logging
        if episode % log_interval == 0:
            elapsed_time = time.time() - start_time
            sps = timestep / elapsed_time if elapsed_time > 0 else 0
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_length = np.mean(episode_lengths[-log_interval:])
            print(f"Episode {episode}, Timestep {timestep}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  SPS (Steps/sec): {sps:.1f}")
            print(f"  Elapsed: {elapsed_time:.1f}s")

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


if __name__ == "__main__":
    train()
