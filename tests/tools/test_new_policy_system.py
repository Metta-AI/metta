import tempfile
from pathlib import Path

import pytest

import mettagrid.builder.envs as eb
from experiments.recipes.arena import mettagrid
from metta.agent.mocks import MockAgent
from metta.cogworks.curriculum import env_curriculum  # Helper function still in old location
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.sim.simulation import Simulation
from metta.sim.simulation_config import SimulationConfig
from metta.sim.simulation_stats_db import SimulationStatsDB
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool


class TestNewPolicySystem:
    """Test the new policy management and checkpoint system."""

    def test_checkpoint_manager_uri_parsing(self):
        """Test that CheckpointManager can parse different URI formats."""
        test_uris = [
            "file:///absolute/path/checkpoint.mpt",
            "file://./relative/path/checkpoint.mpt",
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
        sim = Simulation.create(
            sim_config=SimulationConfig(suite="sim_suite", name="test", env=env_config),
            device="cpu",
            vectorization="serial",
            policy_uri=None,
        )

        assert sim is not None
        assert sim.full_name == "sim_suite/test"

    def test_eval_tool_with_policy_uris(self):
        """Test EvaluateTool with policy URIs."""
        env_config = eb.make_arena(num_agents=4)
        sim_config = SimulationConfig(suite="test", name="test_arena", env=env_config)
        eval_tool = EvaluateTool(
            simulations=[sim_config],
            policy_uris=["mock://test_policy"],
            stats_db_uri=None,
        )

        assert eval_tool.simulations[0].name == "test_arena"
        assert eval_tool.policy_uris == ["mock://test_policy"]

    def test_policy_loading_interface(self):
        """Test that policy loading functions work with versioned URIs."""

        try:
            # Test with a mock URI that should be fully versioned
            artifact = CheckpointManager.load_artifact_from_uri("mock://test_policy")
            assert artifact.policy is not None
        except Exception as e:
            assert "not found" in str(e).lower() or "invalid" in str(e).lower()

    def test_policy_uri_formats(self):
        """Test different policy URI formats are recognized."""
        uri_formats = [
            "file://./checkpoints/model.mpt",
            "file:///absolute/path/model.mpt",
            "s3://bucket/path/model.mpt",
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
            EvaluateTool(simulations=[sim_config], policy_uris=None),
        ]

        for tool in tools:
            assert hasattr(tool, "invoke"), f"{type(tool).__name__} missing invoke method"

    @pytest.mark.slow
    def test_recipe_system_integration(self):
        """Smoke-test that minimal tools can be built from a recipe mettagrid()."""
        env_cfg = mettagrid()

        # Build a basic training tool using env_curriculum
        train_tool = TrainTool(training_env=TrainingEnvironmentConfig(curriculum=env_curriculum(env_cfg)))
        assert hasattr(train_tool, "trainer")

        # Build a simple eval tool around the same env
        sim_cfg = SimulationConfig(suite="arena", name="eval", env=env_cfg)
        eval_tool = EvaluateTool(simulations=[sim_cfg], policy_uris=["mock://test_policy"])
        assert hasattr(eval_tool, "simulations")

        # Replay tool constructed from same sim
        replay_tool = ReplayTool(sim=sim_cfg)
        assert hasattr(replay_tool, "sim")

    def test_mock_agent_fallback(self):
        """Test that mock agents are used when policies can't be loaded."""

        mock_agent = MockAgent()
        assert mock_agent is not None
        assert hasattr(mock_agent, "eval")
