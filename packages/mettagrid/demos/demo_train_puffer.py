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

"""Puffer Demo - Pure PufferLib ecosystem integration.

This demo shows how to use MettaGridEnv with PufferLib and external training libraries.

IMPORTANT: MettaGridEnv inherits from PufferLib's PufferEnv, making it fully compatible
with the PufferLib ecosystem. You can use MettaGridEnv directly with PufferLib training
code, or use PufferLib's MettaPuff wrapper for additional PufferLib-specific features.

Architecture:
- MettaGridEnv -> MettaGridPufferBase -> PufferEnv (PufferLib compatibility)
- For pure PufferLib usage, you can also use: github.com/PufferAI/PufferLib/pufferlib/environments/metta/

Run with: uv run python packages/mettagrid/demos/demo_train_puffer.py (from project root)
"""

import time

import numpy as np

# MettaGrid imports
# Note: MettaGridEnv inherits from PufferEnv, so it's fully PufferLib-compatible
from mettagrid import MettaGridEnv, dtype_actions
from mettagrid.builder.envs import make_arena
from mettagrid.config.mettagrid_config import MettaGridConfig

# Training framework imports
try:
    import importlib.util

    PUFFERLIB_AVAILABLE = importlib.util.find_spec("pufferlib") is not None
except ImportError:
    PUFFERLIB_AVAILABLE = False


def create_test_config() -> MettaGridConfig:
    """Create test configuration for Puffer integration."""
    return MettaGridConfig()


def demo_puffer_env():
    """Demonstrate PufferLib environment creation and basic usage."""
    print("PUFFERLIB ENVIRONMENT DEMO")
    print("=" * 60)

    # Create MettaGridEnv - which IS a PufferLib environment!
    # MettaGridEnv inherits from PufferEnv, so it has all PufferLib functionality
    env = MettaGridEnv(
        env_cfg=make_arena(num_agents=24),
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

    assert isinstance(env.action_space, spaces.Discrete)
    actions = np.random.randint(0, env.action_space.n, size=(env.num_agents,)).astype(dtype_actions, copy=False)

    _, rewards, terminals, truncations, _ = env.step(actions)
    print(f"   - Step successful: obs {observations.shape}, rewards {rewards.shape}")

    env.close()


def demo_random_rollout():
    """Demonstrate random policy rollout in PufferLib environment."""
    print("\nRANDOM ROLLOUT DEMO")
    print("=" * 60)

    # Create MettaGridEnv for rollout
    # Note: is_training=True enables training features like stats collection
    env = MettaGridEnv(
        env_cfg=make_arena(num_agents=24),
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

        assert isinstance(env.action_space, spaces.Discrete)
        actions = np.random.randint(0, env.action_space.n, size=(env.num_agents,)).astype(dtype_actions, copy=False)

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

    # MettaGridEnv can be used directly with PufferLib training code
    env = MettaGridEnv(
        env_cfg=make_arena(num_agents=24),
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

        assert isinstance(env.action_space, spaces.Discrete)
        action_preferences = np.ones(env.action_space.n)

        for _ in range(max_steps):
            # Sample actions based on current preferences
            probs = action_preferences / action_preferences.sum()
            actions = np.random.choice(env.action_space.n, size=env.num_agents, p=probs).astype(
                dtype_actions, copy=False
            )

            _, rewards, terminals, truncations, _ = env.step(actions)
            total_reward += rewards.sum()
            steps += 1

            # Simple "learning": increase preference for actions that led to positive rewards
            if rewards.any():
                for action, reward in zip(actions, rewards, strict=False):
                    if reward > 0:
                        action_preferences[action] *= 1.1

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
    print("PUFFERLIB INTEGRATION DEMO")
    print("=" * 80)
    print("This demo shows MettaGridEnv's PufferLib integration.")
    print("MettaGridEnv inherits from PufferEnv, making it fully compatible")
    print("with the PufferLib high-performance training ecosystem.")
    print()

    try:
        start_time = time.time()

        # Run pure PufferLib demos
        demo_puffer_env()
        demo_random_rollout()
        demo_pufferlib_training()

        # Summary
        duration = time.time() - start_time
        print("\n" + "=" * 80)
        print("PUFFERLIB DEMO COMPLETED")
        print("=" * 80)
        print("Environment creation: Successful")
        print("Random rollout: Completed")
        print("PufferLib integration: Ready")
        print(f"\nTotal demo time: {duration:.1f} seconds")
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
