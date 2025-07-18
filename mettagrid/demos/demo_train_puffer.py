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

"""Puffer Training Integration Demo - Test training with Puffer adapter.

This demo tests the Puffer environment adapter integration with the
actual training pipeline to ensure it works correctly in full training context.

Run with: uv run python mettagrid/demos/demo_train_puffer.py (from project root)
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

# These imports will work when run with uv run (PEP 723)
from metta.mettagrid import MettaGridPufferEnv
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import dtype_actions


def create_test_config() -> DictConfig:
    """Create test configuration for Puffer integration."""
    return DictConfig(
        {
            "game": {
                "max_steps": 80,
                "num_agents": 3,
                "obs_width": 5,
                "obs_height": 5,
                "num_observation_tokens": 25,
                "inventory_item_names": ["heart", "ore_red", "battery_red"],
                "groups": {"agent": {"id": 0, "sprite": 0}},
                "agent": {
                    "default_resource_limit": 10,
                    "resource_limits": {"heart": 255},
                    "freeze_duration": 5,
                    "rewards": {"heart": 5.0, "ore_red": 0.5, "battery_red": 1.0},
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
                        "max_output": 5,
                        "conversion_ticks": 1,
                        "cooldown": 20,
                        "initial_resource_count": 1,
                        "color": 2,
                    },
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "agents": 3,
                    "width": 10,
                    "height": 10,
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


def test_puffer_adapter_functionality():
    """Test Puffer adapter basic functionality."""
    print("PUFFER ADAPTER FUNCTIONALITY TEST")
    print("=" * 60)

    config = create_test_config()
    curriculum = SingleTaskCurriculum("puffer_test", config)

    # Test Puffer environment creation
    env = MettaGridPufferEnv(
        curriculum=curriculum,
        render_mode=None,
        is_training=False,
    )

    print("Puffer adapter created successfully")
    print(f"   - Agents: {env.num_agents}")
    print(f"   - Observation space: {env.observation_space}")
    print(f"   - Action space: {env.action_space}")
    print(f"   - Max steps: {env.max_steps}")

    # Test reset
    observations, info = env.reset(seed=42)
    print(f"   - Reset successful: observations shape {observations.shape}")

    # Test step
    actions = np.random.randint(
        env.action_space.low,
        env.action_space.high,
        size=(env.num_agents, env.action_space.shape[1]),
        dtype=dtype_actions,
    )

    observations, rewards, terminals, truncations, infos = env.step(actions)
    print(f"   - Step successful: obs {observations.shape}, rewards {rewards.shape}")

    env.close()
    print("Puffer adapter functionality test successful!")


def test_puffer_training_integration():
    """Test Puffer integration with actual training pipeline."""
    print("\nPUFFER TRAINING INTEGRATION TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_id = f"puffer_train_test_{int(time.time())}"

        # Training command - very short training for CI/CD reliability
        cmd = [
            "python",
            "tools/train.py",
            f"run={test_id}",
            "+hardware=macbook",
            "trainer.num_workers=1",
            "trainer.total_timesteps=3",
            "trainer.checkpoint.checkpoint_interval=1",
            "trainer.simulation.evaluate_interval=0",
            "wandb=off",
            f"data_dir={temp_dir}/train_dir",
        ]

        print(f"   - Running training command: {' '.join(cmd[:5])}...")
        print(f"   - Test ID: {test_id}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")),
            )

            if result.returncode == 0:
                print("Puffer training integration successful!")
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
                print("Puffer training integration failed!")
                print(f"   - Exit code: {result.returncode}")
                if result.stderr:
                    print(f"   - Error: {result.stderr[:300]}")
                raise RuntimeError(f"Training failed with code {result.returncode}")

        except subprocess.TimeoutExpired:
            print("Training timed out!")
            raise RuntimeError("Training timed out after 60 seconds") from None


def test_puffer_vectorized_training():
    """Test Puffer adapter with vectorized training."""
    print("\nPUFFER VECTORIZED TRAINING TEST")
    print("=" * 60)

    try:
        from metta.rl.vecenv import make_vecenv

        config = create_test_config()
        curriculum = SingleTaskCurriculum("puffer_vec_test", config)

        # Test vectorized environment creation
        vecenv = make_vecenv(
            curriculum=curriculum,
            vectorization="serial",
            num_envs=3,
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
                print(f"     Has {method}")
            else:
                raise AttributeError(f"Missing required method: {method}")

        # Test observation features
        features = driver_env.get_observation_features()
        print(f"   - Observation features: {len(features)} features")

        # Test action names
        action_names = driver_env.action_names
        print(f"   - Action names: {len(action_names)} actions")

        # Test vectorized operations
        obs, infos = vecenv.reset()
        print(f"   - Vectorized reset successful: {obs.shape}")

        # Test step - generate actions correctly
        action_space = driver_env.single_action_space

        # For vectorized environments, we need actions per environment agent
        num_env_agents = vecenv.num_agents
        actions = np.random.randint(
            0, action_space.nvec, size=(num_env_agents, len(action_space.nvec)), dtype=dtype_actions
        )
        print(f"   - Generated actions shape: {actions.shape} for {num_env_agents} agents")

        obs, rewards, terminals, truncations, infos = vecenv.step(actions)
        print(f"   - Vectorized step successful: {obs.shape}")

        vecenv.close()
        print("Puffer vectorized training test successful!")

    except Exception as e:
        print(f"Puffer vectorized training test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Run all Puffer training integration tests."""
    print("PUFFER TRAINING INTEGRATION DEMO")
    print("=" * 60)
    print("This demo tests the Puffer environment adapter integration")
    print("with the actual training pipeline.")

    try:
        start_time = time.time()

        # Run Puffer-specific tests
        test_puffer_adapter_functionality()
        test_puffer_vectorized_training()
        test_puffer_training_integration()

        # Summary
        duration = time.time() - start_time
        print("\n" + "=" * 60)
        print("PUFFER TRAINING INTEGRATION COMPLETED")
        print("=" * 60)
        print("Puffer adapter functionality: Works correctly")
        print("Vectorized training: Compatible with training pipeline")
        print("Training integration: Full training pipeline works")
        print(f"\nTotal test time: {duration:.1f} seconds")
        print("\nPuffer adapter is ready for production training")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
