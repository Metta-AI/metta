"""Test suite for basic policy and environment functionality.
Updated for Pydantic-based configuration system.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

import mettagrid.builder.envs as eb
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import SingleTaskGenerator
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvalTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
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
    def test_basic_training_integration(self, monkeypatch):
        """Test that we can run basic training with the new system."""
        run_name = "integration_test"

        captured_runs: list[tuple[list[str], dict[str, object]]] = []

        def _fake_run(cmd: list[str], **kwargs):
            captured_runs.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _fake_run)

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

            subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=Path.cwd(),
            )

        assert captured_runs, "Training command was not invoked"
        train_cmd, kwargs = captured_runs[0]
        assert train_cmd[:4] == ["uv", "run", "./tools/run.py", "experiments.recipes.arena.train"]
        assert kwargs["cwd"] == Path.cwd()
        assert kwargs["env"]["AWS_ACCESS_KEY_ID"] == "dummy_for_test"

    def test_simulation_creation(self, monkeypatch):
        """Test simulation configuration creation and instantiation."""

        env_config = eb.make_navigation(num_agents=2)
        sim_config = SimulationConfig(suite="test", name="test_nav", env=env_config)

        assert sim_config.name == "test_nav"
        assert sim_config.env.game.num_agents == 2

        def _small_curriculum(cls, mg_config):
            return CurriculumConfig(
                task_generator=SingleTaskGenerator.Config(env=mg_config),
                num_active_tasks=1,
                max_task_id=1,
            )

        monkeypatch.setattr(CurriculumConfig, "from_mg", classmethod(_small_curriculum))
        simulation = Simulation.create(
            sim_config=sim_config,
            device="cpu",
            vectorization="serial",
            policy_uri=None,
        )
        try:
            assert simulation.full_name == "test/test_nav"
        finally:
            simulation._vecenv.close()  # type: ignore[attr-defined]

    def test_eval_tool_config_with_policy_uri(self):
        """Test that EvalTool accepts policy URIs."""

        env_config = eb.make_arena(num_agents=4)
        sim_config = SimulationConfig(suite="test", name="test_arena", env=env_config)

        eval_tool = EvalTool(simulations=[sim_config], policy_uris=["mock://test_policy"], stats_db_uri=None)

        assert eval_tool.simulations[0].name == "test_arena"
        assert eval_tool.policy_uris == ["mock://test_policy"]

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
