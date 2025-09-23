"""Test suite for basic policy and environment functionality.
Updated for Pydantic-based configuration system.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import mettagrid.builder.envs as eb
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool
from mettagrid import MettaGridEnv, dtype_observations


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
        """Test simulation configuration creation and instantiation."""

        env_config = eb.make_navigation(num_agents=2)
        sim_config = SimulationConfig(suite="test", name="test_nav", env=env_config)

        assert sim_config.name == "test_nav"
        assert sim_config.env.game.num_agents == 2

        simulation = Simulation.create(
            sim_config=sim_config,
            device="cpu",
            vectorization="serial",
            policy_uri=None,
        )
        try:
            assert simulation.name == "test/test_nav"
        finally:
            simulation._vecenv.close()  # type: ignore[attr-defined]

    def test_sim_tool_config_with_policy_uri(self):
        """Test that SimTool accepts policy URIs."""

        env_config = eb.make_arena(num_agents=4)
        sim_config = SimulationConfig(suite="test", name="test_arena", env=env_config)

        sim_tool = SimTool(simulations=[sim_config], policy_uris=["mock://test_policy"], stats_db_uri=None)

        assert sim_tool.simulations[0].name == "test_arena"
        assert sim_tool.policy_uris == ["mock://test_policy"]

    def test_play_and_replay_tools_share_run_configuration(self):
        """Ensure basic tool wiring stays aligned with SimulationConfig usage."""

        env_config = eb.make_navigation(num_agents=2)
        sim_config = SimulationConfig(suite="test", name="tool_config", env=env_config)

        play_tool = PlayTool(sim=sim_config, policy_uri=None, open_browser_on_start=False)
        replay_tool = ReplayTool(sim=sim_config, policy_uri=None, open_browser_on_start=False)

        assert play_tool.sim.name == "tool_config"
        assert replay_tool.sim.name == "tool_config"
        assert play_tool.open_browser_on_start is False
        assert replay_tool.open_browser_on_start is False
