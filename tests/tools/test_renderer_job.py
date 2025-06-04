"""
Test suite for probe environments to ensure they work correctly with the training system.
Tests all four debug environments with minimal training steps (~30 seconds total).
"""

import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


class TestProbeEnvironments:
    """Test probe environments work with training system."""

    # Map of environment names to their map file paths
    PROBE_ENVIRONMENTS = {
        "tiny_two_altars": "configs/env/mettagrid/maps/debug/tiny_two_altars.map",
        "simple_obstacles": "configs/env/mettagrid/maps/debug/simple_obstacles.map",
        "resource_collection": "configs/env/mettagrid/maps/debug/resource_collection.map",
        "mixed_objects": "configs/env/mettagrid/maps/debug/mixed_objects.map",
    }

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for training outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_map_files_exist(self):
        """Test that all probe environment map files exist."""
        for env_name, map_path in self.PROBE_ENVIRONMENTS.items():
            path = Path(map_path)
            assert path.exists(), f"Map file for {env_name} not found at {map_path}"
            assert path.is_file(), f"Map path for {env_name} is not a file: {map_path}"

    def test_debug_config_exists(self):
        """Test that the generic debug config exists."""
        config_path = Path("configs/env/mettagrid/debug.yaml")
        assert config_path.exists(), "Generic debug config not found"
        assert config_path.is_file(), "Debug config path is not a file"

    @pytest.mark.parametrize("env_name,map_path", PROBE_ENVIRONMENTS.items())
    def test_environment_training(self, env_name, map_path):
        """Test that each probe environment can be used for training."""
        # Create a temporary run name
        run_name = f"test_probe_{env_name}"

        # Construct training command using generic debug config with environment variable
        cmd = [
            "python",
            "-m",
            "tools.train",
            f"run={run_name}",
            "+env=mettagrid/debug",  # Use generic debug config
            "+hardware=macbook",
            "trainer.total_timesteps=250",  # Very short training
            "trainer.num_workers=1",
            "wandb=off",
        ]

        try:
            # Set environment variable to specify the map
            env = os.environ.copy()
            env["DEBUG_MAP_URI"] = map_path

            # Run training with timeout
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                check=True,
            )

            # Check that training completed successfully
            assert result.returncode == 0, f"Training failed for {env_name}: {result.stderr}"
            assert "Training complete" in result.stdout, f"Training did not complete for {env_name}"

        except subprocess.TimeoutExpired:
            pytest.fail(f"Training timed out for {env_name}")
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Training failed for {env_name}: {e.stderr}")
        finally:
            # Clean up training directory
            train_dir = Path(f"./train_dir/{run_name}")
            if train_dir.exists():
                import shutil

                shutil.rmtree(train_dir, ignore_errors=True)

    def test_all_environments_complete_quickly(self, temp_data_dir):
        """Test that all environments can be trained within reasonable time limit."""

        total_start_time = time.time()
        successful_envs = 0

        for env_name, _map_path in self.PROBE_ENVIRONMENTS.items():
            run_name = f"test_batch_{env_name}_{int(time.time())}"

            cmd = [
                "python",
                "-m",
                "tools.train",
                f"run={run_name}",
                "+env=mettagrid/debug",
                f"data_dir={temp_data_dir}",
                "trainer.total_timesteps=100",  # Even shorter for batch test
                "trainer.num_workers=1",
                "wandb=off",
                "+hardware=macbook",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,  # 30 second timeout per environment
                    cwd=Path.cwd(),
                )

                if result.returncode == 0:
                    successful_envs += 1
                else:
                    print(f"⚠ {env_name} failed in batch test. STDERR: {result.stderr}")

            except subprocess.TimeoutExpired:
                print(f"⚠ {env_name} timed out in batch test")
            except Exception as e:
                print(f"⚠ {env_name} failed in batch test: {str(e)}")

        total_elapsed = time.time() - total_start_time

        # All environments should complete successfully
        assert successful_envs == len(self.PROBE_ENVIRONMENTS), (
            f"Only {successful_envs}/{len(self.PROBE_ENVIRONMENTS)} environments completed successfully"
        )

        # Total time should be reasonable (target: ~30 seconds total)
        assert total_elapsed < 90, f"Total batch training took too long ({total_elapsed:.1f}s)"

        print(f"✓ All {len(self.PROBE_ENVIRONMENTS)} environments completed in {total_elapsed:.1f}s")
