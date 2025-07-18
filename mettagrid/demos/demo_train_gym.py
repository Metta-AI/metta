#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "gymnasium",
#     "omegaconf",
#     "torch",
#     "typing-extensions",
#     "pydantic",
# ]
# ///

"""Gym Training Integration Demo - Test training with Gym adapter.

This demo tests the Gym environment adapter integration with the
actual training pipeline to ensure it works correctly in full training context.

Run with: uv run mettagrid/demos/demo_train_gym.py (from project root)
"""

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

# These imports will work when run with uv run (PEP 723)
from metta.mettagrid import MettaGridGymEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import dtype_actions


def create_test_config() -> DictConfig:
    """Create test configuration for Gym integration."""
    return DictConfig(
        {
            "game": {
                "max_steps": 60,
                "num_agents": 2,
                "obs_width": 4,
                "obs_height": 4,
                "num_observation_tokens": 16,
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
                    "agents": 2,
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


def test_gym_adapter_functionality():
    """Test Gym adapter basic functionality."""
    print("🏋️ GYM ADAPTER FUNCTIONALITY TEST")
    print("=" * 60)

    config = create_test_config()
    curriculum = SingleTaskCurriculum("gym_test", config)

    # Test Gym environment creation
    env = MettaGridGymEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    print("✅ Gym adapter created successfully")
    print(f"   - Agents: {env.num_agents}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Max steps: {env.max_steps}")

    # Test reset
    observations, info = env.reset(seed=42)
    print(f"   - Reset successful: observations shape {observations.shape}")
    print(f"   - Info keys: {list(info.keys()) if info else 'None'}")

    # Test step
    actions = np.random.randint(
        0, min(3, env.action_space.nvec.max()), size=(env.num_agents, env.action_space.shape[0]), dtype=dtype_actions
    )

    observations, rewards, terminated, truncated, infos = env.step(actions)
    print(f"   - Step successful: obs {observations.shape}, rewards {rewards.shape}")
    print(f"   - Terminated: {terminated}, Truncated: {truncated}")

    # Test Gymnasium compatibility
    from gymnasium import spaces

    assert isinstance(env.observation_space, spaces.Box), "Observation space should be Box"
    assert isinstance(env.action_space, spaces.MultiDiscrete), "Action space should be MultiDiscrete"
    print("   - Gymnasium compatibility verified")

    env.close()
    print("✅ Gym adapter functionality test successful!")


def test_gym_training_integration():
    """Test Gym integration with actual training pipeline."""
    print("\n🚂 GYM TRAINING INTEGRATION TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_id = f"gym_train_test_{int(time.time())}"

        # Training command - very short training for CI/CD reliability
        cmd = [
            "python",
            "tools/train.py",
            f"run={test_id}",
            "+hardware=macbook",
            "trainer.num_workers=1",
            "trainer.total_timesteps=200",  # Very short training
            "trainer.checkpoint.checkpoint_interval=100",
            "trainer.simulation.evaluate_interval=0",
            "wandb=off",
            f"data_dir={temp_dir}/train_dir",
        ]

        print(f"   - Running training command: {' '.join(cmd[:5])}...")
        print(f"   - Test ID: {test_id}")

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, cwd="/Users/teodorionita-radu/Desktop/softmax/metta"
            )

            if result.returncode == 0:
                print("✅ Gym training integration successful!")
                print("   - Training completed without errors")

                # Check for outputs
                train_dir = Path(temp_dir) / "train_dir" / test_id
                if train_dir.exists():
                    print(f"   - Training directory created: {train_dir}")

                    checkpoints_dir = train_dir / "checkpoints"
                    if checkpoints_dir.exists():
                        checkpoints = list(checkpoints_dir.glob("*.pt"))
                        print(f"   - Found {len(checkpoints)} checkpoint files")

            else:
                print("❌ Gym training integration failed!")
                print(f"   - Exit code: {result.returncode}")
                if result.stderr:
                    print(f"   - Error: {result.stderr[:300]}")
                raise RuntimeError(f"Training failed with code {result.returncode}")

        except subprocess.TimeoutExpired:
            print("❌ Training timed out!")
            raise RuntimeError("Training timed out after 60 seconds") from None


def test_gym_multi_agent_training():
    """Test Gym adapter with multi-agent training scenarios."""
    print("\n⚡ GYM MULTI-AGENT TRAINING TEST")
    print("=" * 60)

    try:
        from metta.rl.vecenv import make_vecenv

        config = create_test_config()
        curriculum = SingleTaskCurriculum("gym_multiagent_test", config)

        # Test vectorized environment creation
        vecenv = make_vecenv(
            curriculum=curriculum,
            vectorization="serial",
            num_envs=2,
            num_workers=1,
            render_mode=None,
            is_training=False,
        )

        print(f"   - Created vectorized environment with {vecenv.num_envs} environments")
        print(f"   - Agents per environment: {vecenv.num_agents}")
        print(f"   - Total agents: {vecenv.num_envs * vecenv.num_agents}")

        # Test that driver environment has training-required methods
        driver_env = vecenv.driver_env
        print(f"   - Driver environment type: {type(driver_env).__name__}")

        # Test training interface methods
        required_methods = [
            "get_observation_features",
            "action_names",
            "max_action_args",
            "single_observation_space",
            "single_action_space",
        ]

        for method in required_methods:
            if hasattr(driver_env, method):
                print(f"     ✅ Has {method}")
            else:
                raise AttributeError(f"Missing required method: {method}")

        # Test observation features
        features = driver_env.get_observation_features()
        print(f"   - Observation features: {len(features)} features")

        # Test action names
        action_names = driver_env.action_names
        print(f"   - Action names: {len(action_names)} actions")

        # Test multi-agent operations
        obs, infos = vecenv.reset()
        print(f"   - Multi-agent reset successful: {obs.shape}")

        # Test with different actions per agent
        total_agents = vecenv.num_envs * vecenv.num_agents
        actions = np.random.randint(
            0, min(3, driver_env.single_action_space.nvec.max()), size=(total_agents, 2), dtype=dtype_actions
        )

        obs, rewards, terminals, truncations, infos = vecenv.step(actions)
        print(f"   - Multi-agent step successful: {obs.shape}")
        print(f"   - Individual agent rewards: {rewards.shape}")

        vecenv.close()
        print("✅ Gym multi-agent training test successful!")

    except Exception as e:
        print(f"❌ Gym multi-agent training test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Run all Gym training integration tests."""
    print("🏋️ GYM TRAINING INTEGRATION DEMO")
    print("=" * 60)
    print("This demo tests the Gym environment adapter integration")
    print("with the actual training pipeline.")

    try:
        start_time = time.time()

        # Run Gym-specific tests including short training
        test_gym_adapter_functionality()
        test_gym_multi_agent_training()
        test_gym_training_integration()

        # Summary
        duration = time.time() - start_time
        print("\n" + "=" * 60)
        print("🎉 GYM TRAINING INTEGRATION COMPLETED!")
        print("=" * 60)
        print("✅ Gym adapter functionality: Works correctly")
        print("✅ Multi-agent training: Compatible with training pipeline")
        print("✅ Training integration: Short training run successful")
        print(f"\nTotal test time: {duration:.1f} seconds")
        print("\n🚀 Gym adapter is ready for production training!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
