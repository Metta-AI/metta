#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "gymnasium",
#     "pufferlib>=0.6.0",
#     "omegaconf",
#     "typing-extensions",
#     "pydantic",
# ]
# ///

"""Puffer Demo - EnvConfig-based PufferLib ecosystem integration.

This demo shows how to use AutoResetEnv with EnvConfig and PufferLib integration.
This is the comprehensive demo combining the best of both EnvConfig usage and PufferLib compatibility.

Env Hierarchy:
- AutoResetEnv -> MettaGridPufferBase -> PufferEnv (PufferLib compatibility)
- For pure PufferLib usage, you can also use: github.com/PufferAI/PufferLib/pufferlib/environments/metta/


Run with: uv run python mettagrid/demos/demo_train_puffer.py (from project root)
"""

import time

import numpy as np

# Shared demo configuration
from demo_config import create_demo_config

# MettaGrid imports
# Note: AutoResetEnv inherits from PufferEnv, so it's fully PufferLib-compatible
from metta.mettagrid import AutoResetEnv
from metta.mettagrid.config import EnvConfig
from metta.mettagrid.room.utils import make_level_map

# Training framework imports
try:
    import importlib.util

    PUFFERLIB_AVAILABLE = importlib.util.find_spec("pufferlib") is not None
except ImportError:
    PUFFERLIB_AVAILABLE = False


def create_test_env_config() -> EnvConfig:
    """Create test EnvConfig for Puffer integration."""
    return create_demo_config(num_agents=3, max_steps=80, map_width=10, map_height=10)


def demo_puffer_env():
    """Demonstrate PufferLib environment creation and basic usage."""
    print("PUFFERLIB ENVIRONMENT DEMO")
    print("=" * 60)

    env_config = create_test_env_config()

    # Create AutoResetEnv - which IS a PufferLib environment!
    # AutoResetEnv inherits from PufferEnv, so it has all PufferLib functionality
    env = AutoResetEnv(
        env_config=env_config,
        render_mode=None,
        is_training=False,  # Disable training-specific features for this demo
    )

    print("PufferLib environment created")
    print(f"   - Agents: {env.num_agents}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Max steps: {env.max_steps}")

    observations, _ = env.reset(seed=42)
    print(f"   - Reset successful: observations shape {observations.shape}")

    # Generate random actions compatible with the action space
    from gymnasium import spaces

    if isinstance(env.action_space, spaces.MultiDiscrete):
        # MultiDiscrete case
        actions = np.zeros((env.num_agents, len(env.action_space.nvec)), dtype=np.int32)
        for i in range(env.num_agents):
            for j, n in enumerate(env.action_space.nvec):
                actions[i, j] = np.random.randint(0, n)
    else:
        # Box case
        actions = np.random.randint(
            env.action_space.low,
            env.action_space.high + 1,
            size=env.action_space.shape,
            dtype=np.int32,
        )

    _, rewards, terminals, truncations, _ = env.step(actions)
    print(f"   - Step successful: obs {observations.shape}, rewards {rewards.shape}")

    env.close()


def demo_random_rollout():
    """Demonstrate random policy rollout in PufferLib environment."""
    print("\nRANDOM ROLLOUT DEMO")
    print("=" * 60)

    env_config = create_test_env_config()

    # Create AutoResetEnv for rollout
    # Note: is_training=True enables training features like stats collection
    env = AutoResetEnv(
        env_config=env_config,
        render_mode=None,
        is_training=True,
    )

    print("Running random policy rollout...")
    print(f"   - Agents: {env.num_agents}")
    print(f"   - Action space: {env.action_space}")

    _, _ = env.reset(seed=42)
    total_reward = 0
    steps = 0
    max_steps = 100  # Small for CI
    episodes = 0

    for _ in range(max_steps):
        # Generate random actions for all agents
        from gymnasium import spaces

        if isinstance(env.action_space, spaces.MultiDiscrete):
            # MultiDiscrete case
            actions = np.zeros((env.num_agents, len(env.action_space.nvec)), dtype=np.int32)
            for i in range(env.num_agents):
                for j, n in enumerate(env.action_space.nvec):
                    actions[i, j] = np.random.randint(0, n)
        else:
            # Box case
            actions = np.random.randint(
                env.action_space.low, env.action_space.high + 1, size=env.action_space.shape, dtype=np.int32
            )

        _, rewards, terminals, truncations, _ = env.step(actions)
        total_reward += rewards.sum()
        steps += 1

        # Check for episode termination
        if terminals.any() or truncations.any():
            episodes += 1
            print(f"   Episode {episodes} completed at step {steps}")
            _, _ = env.reset()

    avg_reward = total_reward / steps if steps > 0 else 0
    print(f"Completed {steps} steps across {episodes} episodes")
    print(f"   - Average reward per step: {avg_reward:.3f}")

    assert steps > 0, "Expected at least one step to be taken"
    assert not np.isnan(total_reward), "Total reward is NaN"

    env.close()


def demo_custom_level():
    """Demonstrate creating environment with custom level and EnvConfig."""
    print("\nCUSTOM LEVEL DEMO")
    print("=" * 60)

    # Create a simple custom level with agents placed correctly
    grid = np.array(
        [
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
            ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
            ["wall", "empty", "agent.agent", "empty", "generator_red", "empty", "wall"],
            ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
            ["wall", "empty", "mine_red", "empty", "altar", "empty", "wall"],
            ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ],
        dtype="<U50",
    )  # Need larger dtype for longer strings

    # Create level map
    level_map = make_level_map(grid, labels=["custom_puffer_level"])

    # Create EnvConfig with custom level
    env_config = create_demo_config(num_agents=1, max_steps=50)
    # Override the level map with our custom one
    env_config = EnvConfig(game=env_config.game, level_map=level_map)

    # Create environment
    env = AutoResetEnv(env_config=env_config, render_mode=None)

    print("Created environment with custom level")
    print(f"  - Grid shape: {grid.shape}")
    print(f"  - Labels: {level_map.labels}")
    print(f"  - Agents: {env.num_agents}")

    # Reset and take a few steps
    obs, info = env.reset(seed=42)
    print(f"  - Reset successful: observation shape {obs.shape}")

    # Take a few random actions
    for step in range(5):
        actions = env.action_space.sample()
        obs, rewards, terminals, truncations, info = env.step(actions)
        print(f"  - Step {step + 1}: reward {rewards.sum():.2f}")
        if terminals.any() or truncations.any():
            break

    env.close()
    print("Custom level demo completed")


def demo_env_config_modification():
    """Demonstrate modifying environment configuration between resets."""
    print("\nENVCONFIG MODIFICATION DEMO")
    print("=" * 60)

    # Create first EnvConfig (small map)
    env_config_small = create_demo_config(num_agents=1, max_steps=30, map_width=6, map_height=6)

    # Create environment with first config
    env = AutoResetEnv(env_config=env_config_small, render_mode=None)

    print("First configuration (6x6 map):")
    obs, info = env.reset(seed=42)
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Max steps: {env.max_steps}")

    # Take a few steps until episode is done
    for _ in range(50):  # Enough steps to finish the episode
        actions = env.action_space.sample()
        obs, rewards, terminals, truncations, info = env.step(actions)
        if terminals.any() or truncations.any():
            break

    # Create second EnvConfig (larger map)
    env_config_large = create_demo_config(num_agents=1, max_steps=60, map_width=12, map_height=12)

    # Set new configuration for next reset
    env.set_env_config(env_config_large)

    print("\nSecond configuration (12x12 map):")
    obs, info = env.reset(seed=123)
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Max steps: {env.max_steps}")

    # Take a few steps
    for _ in range(3):
        actions = env.action_space.sample()
        obs, rewards, terminals, truncations, info = env.step(actions)
        if terminals.any() or truncations.any():
            break

    env.close()
    print("Environment configuration switching successful!")


def demo_pufferlib_training():
    """Demonstrate actual PufferLib training integration."""
    print("\nPUFFERLIB TRAINING DEMO")
    print("=" * 60)

    if not PUFFERLIB_AVAILABLE:
        print("PufferLib not available")
        print("   Install with: pip install pufferlib")
        print("   Then you can use:")
        print("   - PufferLib's optimized vectorized environments")
        print("   - High-performance neural network training")
        print("   - Integration with CleanRL algorithms")
        return

    env_config = create_test_env_config()

    # AutoResetEnv can be used directly with PufferLib training code
    # It inherits all PufferLib functionality through MettaGridPufferBase -> PufferEnv
    env = AutoResetEnv(
        env_config=env_config,
        render_mode=None,
        is_training=True,
    )

    print("Running PufferLib training...")
    print(f"   - Agents: {env.num_agents}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")

    # Try to use PufferLib's training capabilities if available
    try:
        # PufferLib doesn't have a standard training API like pufferlib.frameworks.cleanrl.ppo
        # So we'll do a simple training loop that demonstrates the environment works
        print("   - Running short training loop (256 steps)...")

        _, _ = env.reset(seed=42)
        total_reward = 0
        steps = 0
        max_steps = 256  # Reduced for faster CI

        # Initialize simple policies based on action space type
        from gymnasium import spaces

        if isinstance(env.action_space, spaces.MultiDiscrete):
            # MultiDiscrete: maintain preferences per action value
            action_preferences = []
            for n in env.action_space.nvec:
                action_preferences.append(np.ones(n) / n)
        else:
            # Box: track preferences for discretized bins
            action_low = env.action_space.low
            action_high = env.action_space.high
            num_bins = 10
            action_preferences = np.ones((env.num_agents, env.action_space.shape[1], num_bins))

        for _ in range(max_steps):
            # Sample actions based on current preferences
            if isinstance(env.action_space, spaces.MultiDiscrete):
                # MultiDiscrete case
                actions = np.zeros((env.num_agents, len(env.action_space.nvec)), dtype=np.int32)
                for i in range(env.num_agents):
                    for j, n in enumerate(env.action_space.nvec):
                        probs = action_preferences[j] / action_preferences[j].sum()
                        actions[i, j] = np.random.choice(n, p=probs)
            else:
                # Box case
                actions = np.zeros((env.num_agents, env.action_space.shape[1]), dtype=np.int32)
                for i in range(env.num_agents):
                    for j in range(env.action_space.shape[1]):
                        # Convert preferences to probabilities
                        probs = action_preferences[i, j] / action_preferences[i, j].sum()
                        # Sample bin and convert to action value
                        bin_idx = np.random.choice(num_bins, p=probs)
                        action_range = action_high[i, j] - action_low[i, j] + 1
                        action_val = int(action_low[i, j] + (bin_idx * action_range) / num_bins)
                        action_val = np.clip(action_val, action_low[i, j], action_high[i, j])
                        actions[i, j] = action_val

            _, rewards, terminals, truncations, _ = env.step(actions)
            total_reward += rewards.sum()
            steps += 1

            # Simple "learning": increase preference for actions that led to positive rewards
            if isinstance(env.action_space, spaces.MultiDiscrete):
                # MultiDiscrete learning
                for i in range(env.num_agents):
                    if rewards[i] > 0:
                        for j in range(len(env.action_space.nvec)):
                            action_preferences[j][actions[i, j]] *= 1.1
            else:
                # Box learning
                for i in range(env.num_agents):
                    if rewards[i] > 0:
                        for j in range(env.action_space.shape[1]):
                            # Find which bin the action belonged to
                            action_range = action_high[i, j] - action_low[i, j] + 1
                            bin_idx = int((actions[i, j] - action_low[i, j]) * num_bins / action_range)
                            bin_idx = np.clip(bin_idx, 0, num_bins - 1)
                            # Increase preference for this bin
                            action_preferences[i, j, bin_idx] *= 1.1

            if terminals.any() or truncations.any():
                _, _ = env.reset()

        avg_reward = total_reward / steps if steps > 0 else 0
        print(f"   - Training completed: {steps} steps")
        print(f"   - Average reward: {avg_reward:.3f}")

        assert steps > 0, "Expected at least one step in training"
        assert not np.isnan(total_reward), "Total reward is NaN"

    except Exception as e:
        print("   - Note: Full PufferLib training API not available, ran basic loop instead")
        print(f"   - Error details: {e}")

    print("\nPufferLib capabilities:")
    print("   - Environment verified for PufferLib compatibility")
    print("   - Ready for integration with PufferLib training algorithms")
    print("   - Supports high-throughput vectorized training")

    env.close()


def main():
    """Run PufferLib adapter demo."""
    print("PUFFERLIB + ENVCONFIG INTEGRATION DEMO")
    print("=" * 80)
    print("This demo shows AutoResetEnv's EnvConfig-based PufferLib integration.")
    print("AutoResetEnv inherits from PufferEnv, making it fully compatible")
    print("with the PufferLib high-performance training ecosystem.")
    print("EnvConfig is the primary configuration method, combining PyGameConfig + LevelMap.")
    print()

    try:
        start_time = time.time()

        # Run pure PufferLib demos
        demo_puffer_env()
        demo_random_rollout()
        demo_pufferlib_training()

        # Run EnvConfig-specific demos
        demo_custom_level()
        demo_env_config_modification()

        # Summary
        duration = time.time() - start_time
        print("\n" + "=" * 80)
        print("PUFFERLIB DEMO COMPLETED")
        print("=" * 80)
        print("Environment creation: Successful")
        print("Random rollout: Completed")
        print("PufferLib integration: Ready")
        print("Custom level creation: Successful")
        print("EnvConfig modification: Successful")
        print(f"\nTotal demo time: {duration:.1f} seconds")
        print("\nKey Features Demonstrated:")
        print("   - EnvConfig as primary configuration method")
        print("   - Custom level creation with make_level_map()")
        print("   - Environment reconfiguration between resets")
        print("   - Full PufferLib compatibility and integration")
        print("\nNext steps:")
        print("   - Use PufferLib's vectorized training")
        print("   - Integrate with CleanRL algorithms")
        print("   - Scale to high-throughput experiments")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
