#!/usr/bin/env python3
"""
Interactive test script for the new MettaGrid environment hierarchy.

This script creates a simple interactive demo where you can see the environments
working and manually test different frameworks.
"""

import time

import numpy as np
from omegaconf import DictConfig

from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.gym_env import SingleAgentMettaGridGymEnv
from metta.mettagrid.pettingzoo_env import MettaGridPettingZooEnv
from metta.mettagrid.puffer_env import MettaGridPufferEnv


def create_game_config():
    """Create a simple game configuration for interactive testing."""
    return DictConfig(
        {
            "game": {
                "max_steps": 200,
                "num_agents": 2,
                "obs_width": 7,
                "obs_height": 7,
                "num_observation_tokens": 49,
                "inventory_item_names": ["ore_red", "ore_blue", "battery_red", "battery_blue", "heart"],
                "groups": {"agent": {"id": 0, "sprite": 0}},
                "agent": {
                    "default_resource_limit": 20,
                    "resource_limits": {"heart": 255},
                    "freeze_duration": 10,
                    "rewards": {"heart": 5.0, "ore_red": 0.1, "battery_red": 0.2},
                    "action_failure_penalty": 0.1,
                },
                "actions": {
                    "noop": {"enabled": True},
                    "move": {"enabled": True},
                    "rotate": {"enabled": True},
                    "put_items": {"enabled": True},
                    "get_items": {"enabled": True},
                    "attack": {"enabled": True},
                    "swap": {"enabled": True},
                    "change_color": {"enabled": False},
                    "change_glyph": {"enabled": False, "number_of_glyphs": 0},
                },
                "objects": {
                    "wall": {"type_id": 1, "swappable": False},
                    "mine_red": {
                        "type_id": 2,
                        "output_resources": {"ore_red": 1},
                        "max_output": -1,
                        "conversion_ticks": 1,
                        "cooldown": 0,
                        "initial_resource_count": 0,
                        "color": 0,
                    },
                    "generator_red": {
                        "type_id": 5,
                        "input_resources": {"ore_red": 1},
                        "output_resources": {"battery_red": 1},
                        "max_output": -1,
                        "conversion_ticks": 1,
                        "cooldown": 0,
                        "initial_resource_count": 0,
                        "color": 0,
                    },
                    "altar": {
                        "type_id": 8,
                        "input_resources": {"battery_red": 3},
                        "output_resources": {"heart": 1},
                        "max_output": 5,
                        "conversion_ticks": 1,
                        "cooldown": 100,
                        "initial_resource_count": 1,
                        "color": 1,
                    },
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "agents": 2,
                    "width": 12,
                    "height": 12,
                    "border_width": 1,
                    "objects": {
                        "mine_red": 3,
                        "generator_red": 2,
                        "altar": 1,
                    },
                },
            }
        }
    )


def test_puffer_env():
    """Test PufferLib environment interactively."""
    print("\n" + "=" * 50)
    print("🔥 TESTING PUFFERLIB ENVIRONMENT")
    print("=" * 50)

    config = create_game_config()
    curriculum = SingleTaskCurriculum("puffer_interactive", config)

    env = MettaGridPufferEnv(
        curriculum=curriculum,
        render_mode="human",
        is_training=False,
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
        actions = np.random.randint(
            0, min(3, env.single_action_space.nvec.max()), size=(env.num_agents, 2), dtype=np.int32
        )

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
    print("✅ PufferLib test completed!")


def test_gym_env():
    """Test Gymnasium environment interactively."""
    print("\n" + "=" * 50)
    print("🏃 TESTING GYMNASIUM ENVIRONMENT")
    print("=" * 50)

    config = create_game_config()
    config.game.num_agents = 1  # Single agent for Gym
    config.game.map_builder.agents = 1
    curriculum = SingleTaskCurriculum("gym_interactive", config)

    env = SingleAgentMettaGridGymEnv(
        curriculum=curriculum,
        render_mode="human",
        is_training=False,
    )

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
        action = np.random.randint(0, min(3, env.action_space.nvec.max()), size=2, dtype=np.int32)

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

    config = create_game_config()
    config.game.num_agents = 3  # Multi-agent for PettingZoo
    config.game.map_builder.agents = 3
    curriculum = SingleTaskCurriculum("pettingzoo_interactive", config)

    env = MettaGridPettingZooEnv(
        curriculum=curriculum,
        render_mode="human",
        is_training=False,
    )

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
            actions[agent] = np.random.randint(0, min(3, env.action_space(agent).nvec.max()), size=2, dtype=np.int32)

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
    print("✅ PettingZoo test completed!")


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
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("✅ PufferLib environment working")
        print("✅ Gymnasium environment working")
        print("✅ PettingZoo environment working")
        print("\nYour MettaGrid environment hierarchy is ready for:")
        print("- Training with PufferLib")
        print("- Integration with Gymnasium/stable-baselines3")
        print("- Multi-agent training with PettingZoo")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
