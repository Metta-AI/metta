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

"""Core Training Integration Demo - Test training with Core adapter.

This demo tests the Core environment adapter integration with the
actual training pipeline to ensure it works correctly in full training context.

Run with: uv run python mettagrid/demos/demo_train_core.py (from project root)
"""

import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

# These imports will work when run with uv run (PEP 723)
from metta.mettagrid import MettaGridCore
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import dtype_actions


def create_test_config() -> DictConfig:
    """Create test configuration for Core integration."""
    return DictConfig(
        {
            "game": {
                "max_steps": 40,
                "num_agents": 2,
                "obs_width": 3,
                "obs_height": 3,
                "num_observation_tokens": 9,
                "inventory_item_names": ["heart", "ore_red"],
                "groups": {"agent": {"id": 0, "sprite": 0}},
                "agent": {
                    "default_resource_limit": 6,
                    "resource_limits": {"heart": 255},
                    "freeze_duration": 3,
                    "rewards": {"heart": 3.0, "ore_red": 0.3},
                    "action_failure_penalty": 0.02,
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
                        "cooldown": 2,
                        "initial_resource_count": 0,
                        "color": 0,
                    },
                    "altar": {
                        "type_id": 8,
                        "input_resources": {"ore_red": 3},
                        "output_resources": {"heart": 1},
                        "max_output": 3,
                        "conversion_ticks": 1,
                        "cooldown": 10,
                        "initial_resource_count": 1,
                        "color": 2,
                    },
                },
                "map_builder": {
                    "_target_": "metta.mettagrid.room.random.Random",
                    "agents": 2,
                    "width": 7,
                    "height": 7,
                    "border_width": 1,
                    "objects": {
                        "mine_red": 2,
                        "altar": 1,
                    },
                },
            }
        }
    )


def test_core_adapter_functionality():
    """Test Core adapter basic functionality."""
    print("CORE ADAPTER FUNCTIONALITY TEST")
    print("=" * 60)

    config = create_test_config()
    curriculum = SingleTaskCurriculum("core_test", config)

    # Test Core environment creation
    task = curriculum.get_task()

    # Create level
    from metta.common.util.instantiate import instantiate

    map_builder_config = task.env_cfg().game.map_builder
    map_builder = instantiate(map_builder_config, _recursive_=True)
    level = map_builder.build()

    # Create C++ config
    from omegaconf import OmegaConf

    from metta.mettagrid.mettagrid_c_config import from_mettagrid_config

    game_config_dict = OmegaConf.to_container(task.env_cfg().game)
    if "map_builder" in game_config_dict:
        del game_config_dict["map_builder"]

    c_cfg = from_mettagrid_config(game_config_dict)

    # Test Core environment
    core_env = MettaGridCore(c_cfg, level.grid.tolist(), 42)

    print("Core adapter created successfully")
    print(f"   - Agents: {core_env.num_agents}")
    print(f"   - Observation space: {core_env.observation_space}")
    print(f"   - Action space: {core_env.action_space}")
    print(f"   - Max steps: {core_env.max_steps}")

    # Test buffer setup
    observations = np.zeros((core_env.num_agents, 9, 3), dtype=np.uint8)
    terminals = np.zeros(core_env.num_agents, dtype=bool)
    truncations = np.zeros(core_env.num_agents, dtype=bool)
    rewards = np.zeros(core_env.num_agents, dtype=np.float32)

    core_env.set_buffers(observations, terminals, truncations, rewards)
    print("   - Buffer setup successful")

    # Test initial observations
    initial_obs = core_env.get_initial_observations()
    print(f"   - Initial observations shape: {initial_obs.shape}")

    # Test step
    actions = np.random.randint(
        0, min(3, core_env.action_space.nvec.max()), size=(core_env.num_agents, 2), dtype=dtype_actions
    )

    core_env.step(actions)
    print("   - Step successful")
    print(f"   - Rewards: {rewards}")
    print(f"   - Terminals: {terminals}")

    print("Core adapter functionality test successful!")


def test_core_training_integration():
    """Test Core integration with actual training pipeline."""
    print("\nCORE TRAINING INTEGRATION TEST")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_id = f"core_train_test_{int(time.time())}"

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
                cmd, capture_output=True, text=True, timeout=50, cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            )

            if result.returncode == 0:
                print("Core training integration successful!")
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
                print("Core training integration failed!")
                print(f"   - Exit code: {result.returncode}")
                if result.stderr:
                    print(f"   - Error: {result.stderr[:300]}")
                raise RuntimeError(f"Training failed with code {result.returncode}")

        except subprocess.TimeoutExpired:
            print("Training timed out!")
            raise RuntimeError("Training timed out after 50 seconds") from None


def test_core_environment_pipeline():
    """Test Core adapter with environment pipeline."""
    print("\nCORE ENVIRONMENT PIPELINE TEST")
    print("=" * 60)

    try:
        from metta.rl.vecenv import make_vecenv

        config = create_test_config()
        curriculum = SingleTaskCurriculum("core_pipeline_test", config)

        # Test that core environment works through the pipeline
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

        # Test that driver environment has training-required methods
        driver_env = vecenv.driver_env
        print(f"   - Driver environment type: {type(driver_env).__name__}")

        # Test that core environment is accessible
        core_env = driver_env.core_env
        if core_env is not None:
            print(f"   - Core environment accessible: {type(core_env).__name__}")
            print(f"   - Core environment agents: {core_env.num_agents}")
            print(f"   - Core environment max steps: {core_env.max_steps}")
        else:
            print("   - Warning: Core environment not accessible")

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

        # Test pipeline operations
        obs, infos = vecenv.reset()
        print(f"   - Pipeline reset successful: {obs.shape}")

        # Test step
        action_space = driver_env.single_action_space
        num_env_agents = vecenv.num_agents
        actions = np.random.randint(
            0, action_space.nvec, size=(num_env_agents, len(action_space.nvec)), dtype=dtype_actions
        )

        obs, rewards, terminals, truncations, infos = vecenv.step(actions)
        print(f"   - Pipeline step successful: {obs.shape}")

        vecenv.close()
        print("Core environment pipeline test successful!")

    except Exception as e:
        print(f"Core environment pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def main():
    """Run all Core training integration tests."""
    print("CORE TRAINING INTEGRATION DEMO")
    print("=" * 60)
    print("This demo tests the Core environment adapter integration")
    print("with the actual training pipeline.")

    try:
        start_time = time.time()

        # Run Core-specific tests including short training
        test_core_adapter_functionality()
        test_core_environment_pipeline()
        test_core_training_integration()

        # Summary
        duration = time.time() - start_time
        print("\n" + "=" * 60)
        print("CORE TRAINING INTEGRATION COMPLETED")
        print("=" * 60)
        print("Core adapter functionality: Works correctly")
        print("Environment pipeline: Compatible with training pipeline")
        print("Training integration: Short training run successful")
        print(f"\nTotal test time: {duration:.1f} seconds")
        print("\nCore adapter is ready for production training")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
