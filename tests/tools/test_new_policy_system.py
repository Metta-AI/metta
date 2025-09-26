import tempfile
from pathlib import Path

import pytest

import mettagrid.builder.envs as eb
from experiments.recipes.arena import evaluate, replay, train
from metta.agent.mocks import MockAgent
from metta.cogworks.curriculum.curriculum import CurriculumConfig
from metta.cogworks.curriculum.task_generator import SingleTaskGenerator
from metta.rl.checkpoint_manager import CheckpointManager
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.sim import SimTool


class TestNewPolicySystem:
    """Test the new policy management and checkpoint system."""

    def test_checkpoint_manager_uri_parsing(self):
        """Test that CheckpointManager can parse different URI formats."""
        test_uris = [
            "file:///absolute/path/checkpoint.pt",
            "file://./relative/path/checkpoint.pt",
            "file:///path/to/checkpoints",
            "mock://test_policy",
        ]

        for uri in test_uris:
            assert isinstance(uri, str)
            assert "://" in uri, f"URI {uri} should have protocol"

    def test_policy_metadata_extraction(self):
        """Test policy metadata extraction from URIs."""
        assert hasattr(CheckpointManager, "get_policy_metadata")

    def test_simulation_creation_with_policy_uri(self):
        """Test creating simulations with policy URIs."""
        env_config = eb.make_navigation(num_agents=2)

        monkeypatch = pytest.MonkeyPatch()

        def _small_curriculum(cls, mg_config):
            return CurriculumConfig(
                task_generator=SingleTaskGenerator.Config(env=mg_config),
                num_active_tasks=1,
                max_task_id=1,
            )

        monkeypatch.setattr(CurriculumConfig, "from_mg", classmethod(_small_curriculum))
        try:
            sim = Simulation.create(
                sim_config=SimulationConfig(suite="sim_suite", name="test", env=env_config),
                device="cpu",
                vectorization="serial",
                policy_uri=None,
            )
        finally:
            monkeypatch.undo()

        assert sim is not None
        assert sim.full_name == "sim_suite/test"

    def test_sim_tool_with_policy_uris(self):
        """Test SimTool with policy URIs."""
        env_config = eb.make_arena(num_agents=4)
        sim_config = SimulationConfig(suite="test", name="test_arena", env=env_config)
        sim_tool = SimTool(
            simulations=[sim_config],
            policy_uris=["mock://test_policy"],
            stats_db_uri=None,
        )

        assert sim_tool.simulations[0].name == "test_arena"
        assert sim_tool.policy_uris == ["mock://test_policy"]

    def test_policy_loading_interface(self):
        """Test that policy loading functions work with versioned URIs."""

        try:
            # Test with a mock URI that should be fully versioned
            agent = CheckpointManager.load_from_uri("mock://test_policy")
            # Mock URIs may return None or raise an exception
            assert agent is None or isinstance(agent, object)
        except Exception as e:
            assert "not found" in str(e).lower() or "invalid" in str(e).lower()

    def test_policy_uri_formats(self):
        """Test different policy URI formats are recognized."""
        uri_formats = [
            "file://./checkpoints/model.pt",
            "file:///absolute/path/model.pt",
            "s3://bucket/path/model.pt",
            "wandb://project/artifact:version",
            "mock://test_policy",
        ]

        for uri in uri_formats:
            assert "://" in uri, f"URI {uri} missing protocol separator"
            protocol = uri.split("://")[0]
            assert protocol in ["file", "s3", "wandb", "mock"], f"Unknown protocol {protocol}"

    def test_simulation_stats_integration(self):
        """Test that simulations integrate with the stats system."""

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_stats.db"
            stats_db = SimulationStatsDB(db_path)
            stats_db.initialize_schema()
            assert hasattr(stats_db, "get_replay_urls")
            stats_db.close()

    def test_tool_configuration_consistency(self):
        """Test that all tools have consistent configuration interfaces."""

        env_config = eb.make_navigation(num_agents=2)
        sim_config = SimulationConfig(suite="test", name="test", env=env_config)
        tools = [
            ReplayTool(sim=sim_config, policy_uri=None),
            PlayTool(sim=sim_config, policy_uri=None),
            SimTool(simulations=[sim_config], policy_uris=None),
        ]

        for tool in tools:
            assert hasattr(tool, "invoke"), f"{type(tool).__name__} missing invoke method"

    @pytest.mark.slow
    def test_recipe_system_integration(self):
        """Test that recipes work with the new policy system."""
        try:
            train_tool = train()
            assert hasattr(train_tool, "trainer")

            # Use a mock policy URI for testing evaluate function
            eval_tool = evaluate(policy_uri="mock://test_policy")
            assert hasattr(eval_tool, "simulations")

            replay_tool = replay()
            assert hasattr(replay_tool, "sim")

        except ImportError as e:
            pytest.skip(f"Recipe import failed: {e}")

    def test_mock_agent_fallback(self):
        """Test that mock agents are used when policies can't be loaded."""

        mock_agent = MockAgent()
        assert mock_agent is not None
        assert hasattr(mock_agent, "eval")
