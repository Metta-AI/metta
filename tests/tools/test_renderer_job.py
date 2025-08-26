"""
Test suite for renderer job to ensure debug environments work correctly.
Simplified tests that are more robust in CI environments.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from metta.common.util.fs import get_repo_root
from metta.mettagrid.config import building
from metta.mettagrid.map_builder.random import RandomMapBuilder
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    AgentConfig,
    AgentRewards,
    EnvConfig,
    GameConfig,
)


class TestRendererJob:
    """Test renderer job works with debug environments."""

    REPO_ROOT = get_repo_root()

    # Map of environment names to their map file paths
    # NOTE: These paths no longer exist - kept for reference of what was tested
    DEBUG_ENVIRONMENTS = {
        "tiny_two_altars": f"{REPO_ROOT}/configs/env/mettagrid/maps/debug/tiny_two_altars.map",
        "simple_obstacles": f"{REPO_ROOT}/configs/env/mettagrid/maps/debug/simple_obstacles.map",
        "resource_collection": f"{REPO_ROOT}/configs/env/mettagrid/maps/debug/resource_collection.map",
        "mixed_objects": f"{REPO_ROOT}/configs/env/mettagrid/maps/debug/mixed_objects.map",
    }

    @staticmethod
    def make_debug_env(name: str) -> EnvConfig:
        """Create a debug environment programmatically."""
        if name == "tiny_two_altars":
            # Simple environment with two altars
            return EnvConfig(
                label=name,
                game=GameConfig(
                    num_agents=2,
                    max_steps=100,
                    objects={
                        "wall": building.wall,
                        "altar": building.altar,
                    },
                    actions=ActionsConfig(
                        move=ActionConfig(),
                        rotate=ActionConfig(),
                        get_items=ActionConfig(),
                    ),
                    agent=AgentConfig(
                        rewards=AgentRewards(
                            inventory={
                                "heart": 1,
                            },
                        ),
                    ),
                    map_builder=RandomMapBuilder.Config(
                        agents=2,
                        width=10,
                        height=10,
                        border_object="wall",
                        border_width=1,
                    ),
                ),
            )
        elif name == "simple_obstacles":
            # Environment with walls as obstacles
            return EnvConfig(
                label=name,
                game=GameConfig(
                    num_agents=2,
                    max_steps=100,
                    objects={
                        "wall": building.wall,
                    },
                    actions=ActionsConfig(
                        move=ActionConfig(),
                        rotate=ActionConfig(),
                    ),
                    agent=AgentConfig(
                        rewards=AgentRewards(
                            inventory={
                                "heart": 1,
                            },
                        ),
                    ),
                    map_builder=RandomMapBuilder.Config(
                        agents=2,
                        width=15,
                        height=15,
                        border_object="wall",
                        border_width=2,
                    ),
                ),
            )
        else:
            # Default environment
            return EnvConfig(
                label=name,
                game=GameConfig(
                    num_agents=2,
                    max_steps=100,
                    objects={
                        "wall": building.wall,
                        "altar": building.altar,
                    },
                    actions=ActionsConfig(
                        move=ActionConfig(),
                        rotate=ActionConfig(),
                        get_items=ActionConfig(),
                    ),
                    agent=AgentConfig(
                        rewards=AgentRewards(
                            inventory={
                                "heart": 1,
                            },
                        ),
                    ),
                    map_builder=RandomMapBuilder.Config(
                        agents=2,
                        width=20,
                        height=20,
                        border_object="wall",
                        border_width=1,
                    ),
                ),
            )

    def test_programmatic_env_creation(self):
        """Test that debug environments can be created programmatically."""
        # Test creating each type of debug environment
        for env_name in ["tiny_two_altars", "simple_obstacles", "resource_collection", "mixed_objects"]:
            env_config = self.make_debug_env(env_name)
            assert env_config is not None, f"Failed to create environment {env_name}"
            assert env_config.game.num_agents == 2, f"Environment {env_name} should have 2 agents"
            assert env_config.label == env_name, f"Environment label mismatch for {env_name}"

    def test_debug_env_validation(self):
        """Test that programmatically created debug environments are valid."""
        # Create and validate a debug environment
        env_config = self.make_debug_env("tiny_two_altars")

        # Validate essential components
        assert hasattr(env_config, "game"), "Environment missing game config"
        assert hasattr(env_config.game, "actions"), "Game missing actions config"
        assert hasattr(env_config.game, "objects"), "Game missing objects config"
        assert hasattr(env_config.game, "agent"), "Game missing agent config"
        assert hasattr(env_config.game, "map_builder"), "Game missing map_builder config"

        # Validate actions are properly configured
        assert env_config.game.actions.move is not None, "Move action not configured"
        assert env_config.game.actions.rotate is not None, "Rotate action not configured"

    @pytest.mark.skip(reason="Renderer changed from Hydra to Pydantic config - needs refactor")
    # TODO: (richard) #dehydration
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

    def test_programmatic_env_with_mettagrid(self):
        """Test that programmatically created environments work with MettaGridEnv."""
        try:
            from metta.mettagrid.mettagrid_env import MettaGridEnv

            # Create a simple debug environment
            env_config = self.make_debug_env("tiny_two_altars")

            # Initialize MettaGridEnv with the programmatic config
            env = MettaGridEnv(env_config)

            # Test basic environment operations
            obs, info = env.reset()
            assert obs is not None, "Environment reset failed to return observation"
            assert obs.shape[0] == 2, "Observation should be for 2 agents"

            # Test that action space is properly configured
            assert env.action_space is not None, "Action space not configured"

            # Close the environment
            env.close()

        except ImportError as e:
            pytest.fail(f"Failed to import MettaGridEnv: {str(e)}")
        except Exception as e:
            pytest.fail(f"Failed to create/use programmatic environment: {str(e)}")

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

    def test_agents_count_in_environments(self):
        """Test that each debug environment has exactly 2 agents."""
        for env_name in ["tiny_two_altars", "simple_obstacles", "resource_collection", "mixed_objects"]:
            env_config = self.make_debug_env(env_name)
            assert env_config.game.num_agents == 2, (
                f"Environment {env_name} should have exactly 2 agents, but has {env_config.game.num_agents}"
            )
            # Also check map_builder agent count matches
            if hasattr(env_config.game.map_builder, "agents"):
                assert env_config.game.map_builder.agents == 2, f"Map builder for {env_name} should configure 2 agents"

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "env_name", ["tiny_two_altars", "simple_obstacles", "resource_collection", "mixed_objects"]
    )
    def test_basic_training_validation(self, env_name):
        """Test very basic training validation - just that the environment loads."""
        # Use a minimal training run that just validates environment loading
        run_name = f"validation_{env_name}"

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"\n=== Debug Info for {env_name} ===")
            print(f"Temp directory: {temp_dir}")
            print(f"Working directory: {Path.cwd()}")
            print(f"Creating environment programmatically: {env_name}")

            # Create environment programmatically
            env_config = self.make_debug_env(env_name)
            print(f"Environment created: {env_config.label}")

            # Detect if running in CI
            optional_ci_config = "+user=ci" if os.environ.get("CI", "").lower() == "true" else None

            cmd = list(
                filter(
                    None,
                    [
                        "python",
                        "-m",
                        "metta.tools.train",
                        f"run={run_name}",
                        optional_ci_config,
                        f"data_dir={temp_dir}",
                        "trainer.simulation.replay_dir=${run_dir}/replays/",
                        "trainer.curriculum=/env/mettagrid/debug",
                        "trainer.total_timesteps=50",  # Minimal training
                        "trainer.rollout_workers=1",
                        "trainer.simulation.skip_git_check=true",  # Skip git check for tests
                        "wandb=off",
                    ],
                )
            )

            # Set environment variable (no longer needed for programmatic envs)
            env = os.environ.copy()
            # env["DEBUG_MAP_URI"] = map_path  # No longer using map files

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
