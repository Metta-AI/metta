#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "gymnasium>=0.29.0",
#     "omegaconf",
#     "typing-extensions",
#     "pydantic",
#     "stable-baselines3>=2.0",
# ]
# ///

"""Gym Demo - Pure Gymnasium + SB3 ecosystem integration.

This demo shows how to use MettaGridGymEnv (single-agent mode) with ONLY
Gymnasium and Stable Baselines3, without any Metta training infrastructure.

Run with: uv run python mettagrid/demos/demo_train_gym.py (from project root)
"""

import time

import numpy as np
from omegaconf import DictConfig

# Gym adapter imports
from metta.mettagrid import SingleAgentMettaGridGymEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum

# Training framework imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False


def create_test_config() -> DictConfig:
    """Create test configuration for single-agent Gym integration."""
    return DictConfig(
        {
            "game": {
                "max_steps": 60,
                "num_agents": 1,  # Single agent for SB3 compatibility
                "obs_width": 5,
                "obs_height": 5,
                "num_observation_tokens": 25,
                "inventory_item_names": ["heart", "ore_red", "battery_red"],
                "groups": {"agent": {"id": 0, "sprite": 0}},
                "agent": {
                    "default_resource_limit": 8,
                    "resource_limits": {"heart": 255},
                    "freeze_duration": 4,
                    "rewards": {"heart": 4.0, "ore_red": 0.4, "battery_red": 0.8},
                    "action_failure_penalty": 0.05,
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
                        "cooldown": 3,
                        "initial_resource_count": 0,
                        "color": 0,
                    },
                    "generator_red": {
                        "type_id": 5,
                        "input_resources": {"ore_red": 1},
                        "output_resources": {"battery_red": 1},
                        "max_output": -1,
                        "conversion_ticks": 1,
                        "cooldown": 2,
                        "initial_resource_count": 0,
                        "color": 0,
                    },
                    "altar": {
                        "type_id": 8,
                        "input_resources": {"battery_red": 2},
                        "output_resources": {"heart": 1},
                        "max_output": 4,
                        "conversion_ticks": 1,
                        "cooldown": 15,
                        "initial_resource_count": 1,
                        "color": 2,
                    },
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "agents": 1,  # Single agent for SB3 compatibility
                    "width": 8,
                    "height": 8,
                    "border_width": 1,
                    "objects": {
                        "mine_red": 2,
                        "generator_red": 1,
                        "altar": 1,
                    },
                },
            }
        }
    )


def demo_single_agent_gym():
    """Demonstrate single-agent Gymnasium environment."""
    print("SINGLE-AGENT GYM DEMO")
    print("=" * 60)

    config = create_test_config()
    curriculum = SingleTaskCurriculum("gym_demo", config)
    env = SingleAgentMettaGridGymEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    print("Single-agent Gym environment created")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Max steps: {env.max_steps}")

    observation, info = env.reset(seed=42)
    print(f"   - Reset successful: observation shape {observation.shape}")

    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"   - Step successful: obs {observation.shape}, reward {reward}")
    print(f"   - Terminated: {terminated}, Truncated: {truncated}")

    from gymnasium import spaces

    assert isinstance(env.observation_space, spaces.Box), "Single-agent obs space should be Box"
    assert isinstance(env.action_space, spaces.MultiDiscrete), "Single-agent action space should be MultiDiscrete"
    print("Single-agent Gymnasium compatibility verified")

    env.close()


def demo_sb3_training():
    """Demonstrate SB3 training with single-agent environment."""
    print("\nSTABLE BASELINES3 TRAINING DEMO")
    print("=" * 60)

    if not SB3_AVAILABLE:
        print("Stable Baselines3 not available")
        print("   Install with: pip install stable-baselines3")
        print("   Then you can use:")
        print("   - PPO, A2C, SAC, TD3 algorithms")
        print("   - Vectorized environments")
        print("   - Easy integration with single-agent MettaGrid")
        return

    try:
        config = create_test_config()
        curriculum = SingleTaskCurriculum("sb3_training", config)

        env = SingleAgentMettaGridGymEnv(
            curriculum=curriculum,
            render_mode=None,
            is_training=True,
        )

        print("Created single-agent environment for SB3")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {env.action_space}")

        model = PPO("MlpPolicy", env, verbose=0)
        print("Created PPO model")

        print("Training for 256 timesteps...")
        model.learn(total_timesteps=256)

        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(50):  # Test for 50 steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                obs, _ = env.reset()

        print("Training completed!")
        print(f"   - Trained model average reward ({steps} steps): {total_reward / steps:.3f}")
        print("   - SB3 training integration successful!")

        assert total_reward == total_reward, "Reward is NaN"  # not NaN
        assert steps > 0, "Expected at least one step"

        env.close()

    except Exception as e:
        print(f"SB3 training failed: {e}")
        raise


def demo_vectorized_envs():
    """Demonstrate vectorized environments with SB3."""
    print("\nVECTORIZED ENVIRONMENTS DEMO")
    print("=" * 60)

    if not SB3_AVAILABLE:
        print("Stable Baselines3 not available")
        print("   Vectorized training requires SB3")
        return

    try:
        config = create_test_config()
        curriculum = SingleTaskCurriculum("vectorized_demo", config)

        def make_env():
            def _init():
                return SingleAgentMettaGridGymEnv(
                    curriculum=curriculum,
                    render_mode=None,
                    is_training=True,
                )

            return _init

        vec_env = DummyVecEnv([make_env() for _ in range(4)])

        print("Created 4 vectorized environments")
        print(f"   - Observation space: {vec_env.observation_space}")
        print(f"   - Action space: {vec_env.action_space}")

        observations = vec_env.reset()
        print(f"   - Vectorized reset: {observations.shape}")

        actions = np.array([vec_env.action_space.sample() for _ in range(4)])
        observations, rewards, dones, infos = vec_env.step(actions)
        print(f"   - Vectorized step: obs {observations.shape}, rewards {rewards.shape}")

        assert observations.shape[0] == 4, "Expected 4 environments"
        assert rewards.shape == (4,), "Expected rewards for 4 environments"
        assert dones.shape == (4,), "Expected dones for 4 environments"

        print("Vectorized environments working correctly!")
        print("   - Ready for high-throughput SB3 training")
        print("   - Supports PPO, A2C, and other vectorized algorithms")

        vec_env.close()

    except Exception as e:
        print(f"Vectorized environments failed: {e}")
        raise


def main():
    """Run Gymnasium + SB3 adapter demo."""
    print("GYMNASIUM + SB3 ADAPTER DEMO")
    print("=" * 80)
    print("This demo shows SingleAgentMettaGridGymEnv integration with")
    print("Gymnasium and Stable Baselines3 (no internal training code).")
    print()

    try:
        start_time = time.time()

        # Run pure Gymnasium + SB3 demos
        demo_single_agent_gym()
        demo_sb3_training()
        demo_vectorized_envs()

        # Summary
        duration = time.time() - start_time
        print("\n" + "=" * 80)
        print("GYMNASIUM + SB3 DEMO COMPLETED")
        print("=" * 80)
        print("Single-agent environment: Working")
        print("SB3 training: Successful")
        print("Vectorized environments: Ready")
        print(f"\nTotal demo time: {duration:.1f} seconds")
        print("\nNext steps:")
        print("   - Train with different SB3 algorithms (PPO, A2C, SAC)")
        print("   - Use vectorized environments for faster training")
        print("   - Apply hyperparameter optimization")
        print("   - For multi-agent scenarios, consider PettingZoo adapter")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    raise SystemExit(main())
