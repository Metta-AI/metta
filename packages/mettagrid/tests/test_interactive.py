#!/usr/bin/env python3
"""
Interactive test script for the new MettaGrid environment hierarchy.

This script creates a simple interactive demo where you can see the environments
working and manually test different frameworks.
"""

import time

import numpy as np

from mettagrid import dtype_actions
from mettagrid.builder.envs import make_arena
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    GameConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.envs.gym_env import MettaGridGymEnv
from mettagrid.envs.mettagrid_env import MettaGridEnv
from mettagrid.envs.pettingzoo_env import MettaGridPettingZooEnv
from mettagrid.map_builder.ascii import AsciiMapBuilder


def test_puffer_env():
    """Test MettaGridEnv (PufferLib-based) interactively."""
    print("\n" + "=" * 50)
    print("TESTING METTAGRID ENVIRONMENT (PufferLib-based)")
    print("=" * 50)

    env = MettaGridEnv(
        make_arena(num_agents=24),
        render_mode="human",
    )

    print("Environment created!")
    print(f"- Agents: {env.num_agents}")
    print(f"- Observation space: {env.single_observation_space}")
    print(f"- Action space: {env.single_action_space}")
    print(f"- Max steps: {env.max_steps}")

    # Run a quick episode
    obs, _ = env.reset(seed=42)
    print(f"\nStarting episode with observation shape: {obs.shape}")

    total_reward = 0
    for step in range(20):
        # Random actions - in a real game these would be from policies
        actions = np.random.randint(0, env.single_action_space.n, size=env.num_agents).astype(dtype_actions, copy=False)

        obs, rewards, terminals, truncations, _ = env.step(actions)
        total_reward += rewards.sum()

        print(f"Step {step + 1:2d}: Reward = {rewards.sum():6.2f}, Total = {total_reward:6.2f}")

        # Print any rendered output
        render_output = env.render()
        if render_output and step < 3:  # Only show first few renders
            print(f"Render output: {render_output[:100]}...")

        if terminals.any() or truncations.any():
            print("Episode ended!")
            break

        time.sleep(0.1)  # Brief pause for readability

    env.close()
    print("MettaGridEnv (PufferLib-compatible) test completed!")


def test_gym_env():
    """Test Gymnasium environment interactively."""
    print("\n" + "=" * 50)
    print("🏃 TESTING GYMNASIUM ENVIRONMENT")
    print("=" * 50)

    # Create environment with a simple map
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=1,
            actions=ActionsConfig(
                move=ActionConfig(),
                noop=ActionConfig(),
                rotate=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=1)},
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", ".", "@", ".", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ],
            ),
        )
    )

    env = MettaGridGymEnv(cfg, render_mode="human")

    print("Environment created!")
    print(f"- Agents: {env.num_agents}")
    print(f"- Observation space: {env.observation_space}")
    print(f"- Action space: {env.action_space}")
    print(f"- Max steps: {env.max_steps}")

    # Run a quick episode
    obs, _ = env.reset(seed=42)
    print(f"\nStarting episode with observation shape: {obs.shape}")

    total_reward = 0.0
    for step in range(20):
        # Random action - in a real game this would be from a policy
        action = int(np.random.randint(0, env.action_space.n))

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        print(f"Step {step + 1:2d}: Reward = {reward:6.2f}, Total = {total_reward:6.2f}")

        if terminated or truncated:
            print("Episode ended!")
            break

        time.sleep(0.1)  # Brief pause for readability

    env.close()
    print("✅ Gymnasium test completed!")


def test_pettingzoo_env():
    """Test PettingZoo environment interactively."""
    print("\n" + "=" * 50)
    print("🐧 TESTING PETTINGZOO ENVIRONMENT")
    print("=" * 50)

    # Create environment with a simple map for 3 agents
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=3,
            actions=ActionsConfig(
                move=ActionConfig(),
                noop=ActionConfig(),
                rotate=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=1)},
            agents=[
                AgentConfig(team_id=1),
                AgentConfig(team_id=2),
                AgentConfig(team_id=3),
            ],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#", "#", "#"],
                    ["#", ".", ".", ".", ".", ".", "#"],
                    ["#", ".", "1", ".", "2", ".", "#"],
                    ["#", ".", ".", "3", ".", ".", "#"],
                    ["#", ".", ".", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#", "#", "#"],
                ],
            ),
        )
    )

    env = MettaGridPettingZooEnv(cfg, render_mode="human")

    print("Environment created!")
    print(f"- Max agents: {env.max_num_agents}")
    print(f"- Observation space: {env.observation_space(env.possible_agents[0])}")
    print(f"- Action space: {env.action_space(env.possible_agents[0])}")
    print(f"- Max steps: {env.max_steps}")

    # Run a quick episode
    observations, _ = env.reset(seed=42)
    print(f"\nStarting episode with {len(observations)} agents")

    total_rewards = {agent: 0.0 for agent in env.possible_agents}
    for step in range(20):
        # Random actions for all active agents
        actions = {}
        for agent in env.agents:
            action_space = env.action_space(agent)
            actions[agent] = int(np.random.randint(0, action_space.n))

        observations, rewards, _, _, _ = env.step(actions)

        # Update total rewards
        for agent, reward in rewards.items():
            total_rewards[agent] += reward

        active_total = sum(rewards.values())
        print(f"Step {step + 1:2d}: Active agents = {len(env.agents)}, Total reward = {active_total:6.2f}")

        if not env.agents:  # All agents done
            print("All agents finished!")
            break

        time.sleep(0.1)  # Brief pause for readability

    print(f"Final rewards: {total_rewards}")
    env.close()
    print("PettingZoo test completed!")


def main():
    """Run interactive tests for all environment types."""
    print("🎮 MettaGrid Environment Hierarchy Interactive Test")
    print("=" * 60)
    print("This will test all three environment types with visual output!")
    print("Each test runs a short episode with random actions.")
    print("\nPress Ctrl+C at any time to stop.")

    try:
        # Test each environment type
        test_puffer_env()
        test_gym_env()
        test_pettingzoo_env()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("PufferLib environment working")
        print("Gymnasium environment working")
        print("PettingZoo environment working")
        print("\nYour MettaGrid environment hierarchy is ready for:")
        print("- Training with PufferLib")
        print("- Integration with Gymnasium/stable-baselines3")
        print("- Multi-agent training with PettingZoo")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n Test interrupted by user")
    except Exception as e:
        print(f"\n\n Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
