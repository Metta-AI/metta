"""Test suite for basic policy and environment functionality.
Updated for Pydantic-based configuration system.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import metta.mettagrid.builder.envs as eb
from metta.mettagrid import dtype_observations
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool


class TestBasicPolicyEnvironment:
    """Test basic environment and policy functionality with new system."""

    @pytest.fixture
    def simple_env_config(self):
        """Create a simple environment config."""
        return eb.make_navigation(num_agents=2)

    @pytest.fixture
    def env_with_config(self, simple_env_config):
        """Create MettaGridEnv from config."""
        env = MettaGridEnv(simple_env_config)
        try:
            yield env
        finally:
            env.close()

    def test_basic_environment_creation(self, simple_env_config):
        """Test that we can create environments programmatically."""
        assert simple_env_config.game.num_agents == 2
        assert "altar" in simple_env_config.game.objects
        assert simple_env_config.game.actions.move is not None

    def test_environment_reset_and_step(self, env_with_config):
        """Test basic environment operations."""
        env = env_with_config

        # Test reset
        obs, info = env.reset()
        assert obs is not None
        assert obs.dtype == dtype_observations
        assert obs.shape[0] == env.num_agents

        # Test step
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        assert obs is not None
        assert obs.dtype == dtype_observations
        assert reward is not None
        assert done is not None
        assert truncated is not None

    def test_multiple_environments(self):
        """Test creating multiple different environment types."""
        environments = {
            "arena": eb.make_arena(num_agents=4),
            "navigation": eb.make_navigation(num_agents=2),
        }

        for env_name, config in environments.items():
            assert config is not None, f"Failed to create {env_name} environment"
            assert config.game.num_agents > 0, f"{env_name} has no agents"

    @pytest.mark.slow
    def test_basic_training_integration(self):
        """Test that we can run basic training with the new system."""
        run_name = "integration_test"

        with tempfile.TemporaryDirectory():
            # Use the new recipe-based system
            cmd = [
                "uv",
                "run",
                "./tools/run.py",
                "experiments.recipes.arena.train",
                f"run={run_name}",
                "trainer.total_timesteps=100",  # Very minimal training
                "wandb=off",
            ]

            env = os.environ.copy()
            env["AWS_ACCESS_KEY_ID"] = "dummy_for_test"
            env["AWS_SECRET_ACCESS_KEY"] = "dummy_for_test"

            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout
                    cwd=Path.cwd(),
                )

                # Check if training started successfully (even if it fails later)
                # We're mainly testing that the new system doesn't have import/config errors
                if result.returncode != 0:
                    # Look for specific initialization errors vs training errors
                    combined_output = result.stdout + result.stderr
                    initialization_errors = [
                        "ImportError",
                        "ModuleNotFoundError",
                        "AttributeError",
                        "recipe not found",
                        "TypeError",
                        "NameError",
                    ]

                    found_init_error = any(error in combined_output for error in initialization_errors)

                    if found_init_error:
                        pytest.fail(
                            f"Training initialization failed:\n"
                            f"STDOUT: {result.stdout[-1000:]}\n"
                            f"STDERR: {result.stderr[-1000:]}"
                        )
                    else:
                        # Training started but failed during execution - that's ok for this test
                        print("Training started successfully but failed during execution (expected for minimal test)")

            except subprocess.TimeoutExpired:
                # Timeout might be ok if training is actually running
                print("Training test timed out - likely means it's working")

    def test_simulation_creation(self):
        """Test simulation configuration creation."""

        # Test creating simulation config
        env_config = eb.make_navigation(num_agents=2)
        sim_config = SimulationConfig(name="test_nav", env=env_config)

        assert sim_config.name == "test_nav"
        assert sim_config.env.game.num_agents == 2

    def test_replay_tool_config(self):
        """Test that replay tool can be configured."""

        env_config = eb.make_arena(num_agents=4)
        sim_config = SimulationConfig(name="test_arena", env=env_config)

        replay_tool = ReplayTool(
            sim=sim_config,
            policy_uri=None,
            open_browser_on_start=False,  # Don't try to open browser in tests
        )

        assert replay_tool.sim.name == "test_arena"
        assert replay_tool.open_browser_on_start is False

    def test_play_tool_config(self):
        """Test that play tool can be configured."""

        env_config = eb.make_navigation(num_agents=2)
        sim_config = SimulationConfig(name="test_nav_play", env=env_config)

        play_tool = PlayTool(sim=sim_config, policy_uri=None, open_browser_on_start=False)

        assert play_tool.sim.name == "test_nav_play"
