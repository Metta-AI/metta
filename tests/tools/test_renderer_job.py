"""
Test suite for renderer job to ensure debug environments work correctly.
Simplified tests that are more robust in CI environments.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestRendererJob:
    """Test renderer job works with debug environments."""

    # Map of environment names to their map file paths
    DEBUG_ENVIRONMENTS = {
        "tiny_two_altars": "configs/env/mettagrid/maps/debug/tiny_two_altars.map",
        "simple_obstacles": "configs/env/mettagrid/maps/debug/simple_obstacles.map",
        "resource_collection": "configs/env/mettagrid/maps/debug/resource_collection.map",
        "mixed_objects": "configs/env/mettagrid/maps/debug/mixed_objects.map",
    }

    def test_map_files_exist(self):
        """Test that all debug environment map files exist."""
        for env_name, map_path in self.DEBUG_ENVIRONMENTS.items():
            path = Path(map_path)
            assert path.exists(), f"Map file for {env_name} not found at {map_path}"
            assert path.is_file(), f"Map path for {env_name} is not a file: {map_path}"

    def test_debug_config_exists(self):
        """Test that the generic debug config exists."""
        config_path = Path("configs/env/mettagrid/debug.yaml")
        assert config_path.exists(), "Generic debug config not found"
        assert config_path.is_file(), "Debug config path is not a file"

    def test_renderer_job_config_exists(self):
        """Test that the renderer job config exists."""
        config_path = Path("configs/renderer_job.yaml")
        assert config_path.exists(), "Renderer job config not found"
        assert config_path.is_file(), "Renderer job config path is not a file"

    @pytest.mark.slow
    def test_renderer_with_debug_environments(self):
        """Test that renderer can load and initialize debug environments."""
        # Simple renderer test with very short duration
        for env_name, map_path in self.DEBUG_ENVIRONMENTS.items():
            cmd = [
                "python",
                "-m",
                "tools.renderer",
                f"run=test_renderer_{env_name}",
                f"renderer_job.environment.root.params.uri={map_path}",
                "renderer_job.num_steps=3",  # Very short test
                "renderer_job.sleep_time=0",
                "renderer_job.policy_type=simple",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,  # Short timeout
                    cwd=Path.cwd(),
                )

                # Check that renderer started and ran (exit code 0 or completed some steps)
                assert result.returncode == 0, (
                    f"Renderer failed for {env_name}. STDOUT: {result.stdout[-500:]} STDERR: {result.stderr[-500:]}"
                )

            except subprocess.TimeoutExpired:
                pytest.fail(f"Renderer timed out for {env_name}")
            except Exception as e:
                pytest.fail(f"Renderer test failed for {env_name}: {str(e)}")

    def test_miniscope_renderer_imports(self):
        """Test that MiniscopeRenderer can be imported and initialized."""
        try:
            from metta.mettagrid.renderer.miniscope import MiniscopeRenderer

            # Test basic initialization
            object_type_names = ["agent", "wall", "altar", "mine", "generator"]
            renderer = MiniscopeRenderer(object_type_names)

            # Test that it has the expected attributes
            assert hasattr(renderer, "MINISCOPE_SYMBOLS")
            assert hasattr(renderer, "render")
            assert hasattr(renderer, "_symbol_for")

            # Test that symbols dictionary is populated
            assert len(renderer.MINISCOPE_SYMBOLS) > 0
            assert "wall" in renderer.MINISCOPE_SYMBOLS
            assert "agent" in renderer.MINISCOPE_SYMBOLS
            assert renderer.MINISCOPE_SYMBOLS["wall"] == "🧱"
            assert renderer.MINISCOPE_SYMBOLS["agent"] == "🤖"

        except ImportError as e:
            pytest.fail(f"Failed to import MiniscopeRenderer: {str(e)}")

    def test_agents_count_in_maps(self):
        """Test that each debug map has exactly 2 agents."""
        for env_name, map_path in self.DEBUG_ENVIRONMENTS.items():
            with open(map_path, "r") as f:
                content = f.read()

            agent_count = content.count("@")
            assert agent_count == 2, f"Map {env_name} should have exactly 2 agents (@), but found {agent_count}"

    @pytest.mark.slow
    @pytest.mark.parametrize("env_name,map_path", DEBUG_ENVIRONMENTS.items())
    def test_basic_training_validation(self, env_name, map_path):
        """Test very basic training validation - just that the environment loads."""
        # Use a minimal training run that just validates environment loading
        run_name = f"validation_{env_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\n=== Debug Info for {env_name} ===")
            print(f"Temp directory: {temp_dir}")
            print(f"Working directory: {Path.cwd()}")
            print(f"Map path: {map_path}")

            # Check if map file exists
            full_map_path = Path.cwd() / map_path
            print(f"Full map path: {full_map_path}")
            print(f"Map file exists: {full_map_path.exists()}")

            # Detect if running in CI
            optional_ci_config = "+user=ci" if os.environ.get("CI", "").lower() == "true" else None

            cmd = list(
                filter(
                    None,
                    [
                        "python",
                        "-m",
                        "tools.train",
                        f"run={run_name}",
                        optional_ci_config,
                        f"data_dir={temp_dir}",
                        "trainer.simulation.replay_dir=${run_dir}/replays/",
                        "trainer.curriculum=/env/mettagrid/debug",
                        "trainer.total_timesteps=50",  # Minimal training
                        "trainer.num_workers=1",
                        "wandb=off",
                    ],
                )
            )

            # Set environment variable to specify the map
            env = os.environ.copy()
            env["DEBUG_MAP_URI"] = map_path

            # Set dummy AWS credentials to bypass AWS configuration check
            env["AWS_ACCESS_KEY_ID"] = "dummy_access_key_for_testing"
            env["AWS_SECRET_ACCESS_KEY"] = "dummy_secret_key_for_testing"

            # Add more verbose logging
            env["HYDRA_FULL_ERROR"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            print(f"\nRunning command: {' '.join(cmd)}")
            print(f"DEBUG_MAP_URI: {env.get('DEBUG_MAP_URI')}")

            try:
                timeout = 300
                print(f'Running cmd "{cmd}" with timeout {timeout} sec')
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=Path.cwd(),
                )
            except subprocess.TimeoutExpired as e:
                print(f"\n=== Command timed out after {timeout} seconds ===")

                # Decode bytes to string, defaulting to empty string if None
                stdout_text = e.stdout.decode("utf-8") if e.stdout else "None"
                stderr_text = e.stderr.decode("utf-8") if e.stderr else "None"

                print(f"Partial STDOUT: {stdout_text}")
                print(f"Partial STDERR: {stderr_text}")
                pytest.fail(f"Training validation timed out for {env_name}")
            except Exception as e:
                print("\n=== Unexpected error running subprocess ===")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                pytest.fail(f"Unexpected error during training validation for {env_name}: {str(e)}")

            # Print full output for debugging
            print("\n=== Full STDOUT ===")
            print(result.stdout)
            print("\n=== Full STDERR ===")
            print(result.stderr)

            # List contents of temp directory after run
            print("\n=== Temp directory contents after run ===")
            try:
                for root, _dirs, files in os.walk(temp_dir):
                    level = root.replace(temp_dir, "").count(os.sep)
                    indent = " " * 2 * level
                    print(f"{indent}{os.path.basename(root)}/")
                    sub_indent = " " * 2 * (level + 1)
                    for file in files:
                        print(f"{sub_indent}{file}")
            except Exception as e:
                print(f"Error listing directory contents: {e}")

            # More lenient success criteria
            if result.returncode != 0:
                # Look for specific error patterns in the output
                error_patterns = {
                    "ImportError": "Import error detected",
                    "FileNotFoundError": "File not found error",
                    "KeyError": "Configuration key error",
                    "AttributeError": "Attribute error",
                    "ValueError": "Value error",
                    "TypeError": "Type error",
                    "RuntimeError": "Runtime error",
                    "AssertionError": "Assertion error",
                    "ModuleNotFoundError": "Module not found error",
                }

                combined_output = result.stdout + result.stderr
                detected_errors = []
                for pattern, description in error_patterns.items():
                    if pattern in combined_output:
                        detected_errors.append(description)

                # Print detailed error information for debugging
                error_msg = (
                    f"Training validation failed for {env_name}.\n"
                    f"Return code: {result.returncode}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Working directory: {Path.cwd()}\n"
                    f"Temp directory: {temp_dir}\n"
                    f"Detected errors: {', '.join(detected_errors) if detected_errors else 'None detected'}\n"
                    f"STDOUT (last 2000 chars): ...{result.stdout[-2000:]}\n"
                    f"STDERR (last 2000 chars): ...{result.stderr[-2000:]}"
                )
                pytest.fail(error_msg)
