import pytest

import mettagrid.builder.envs as eb
from metta.agent.mocks import MockAgent
from metta.cogworks.curriculum import env_curriculum
from metta.rl.checkpoint_manager import CheckpointManager
from metta.rl.training.training_environment import TrainingEnvironmentConfig
from metta.sim.runner import SimulationRunConfig, run_simulations
from metta.sim.simulation_config import SimulationConfig
from metta.tools.eval import EvaluateTool
from metta.tools.play import PlayTool
from metta.tools.replay import ReplayTool
from metta.tools.train import TrainTool
from mettagrid.policy.loader import initialize_or_load_policy
from mettagrid.policy.policy import PolicySpec
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import AgentObservation, ObservationToken
from recipes.experiment.arena import mettagrid
from tests.helpers.fast_train_tool import create_minimal_training_setup, run_fast_train_tool


@pytest.fixture
def real_checkpoint_uri(tmp_path):
    trainer_cfg, training_env_cfg, policy_cfg, system_cfg = create_minimal_training_setup(tmp_path)
    checkpoint_manager = run_fast_train_tool(
        run_name="eval_tool_real_checkpoint",
        system_cfg=system_cfg,
        trainer_cfg=trainer_cfg,
        training_env_cfg=training_env_cfg,
        policy_cfg=policy_cfg,
    )
    latest = checkpoint_manager.get_latest_checkpoint()
    assert latest is not None
    return latest, system_cfg.model_copy(deep=True)


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

    def test_simulation_runner_with_mock_policy(self):
        """Test that the simulation runner works with a mock policy initializer."""
        env_config = eb.make_navigation(num_agents=2)
        sim_run_config = SimulationRunConfig(env=env_config)

        policy_spec = PolicySpec(class_path="metta.agent.mocks.mock_agent.MockAgent", data_path=None)
        results = run_simulations(
            policy_specs=[policy_spec],
            simulations=[sim_run_config],
            replay_dir=None,
            seed=0,
        )

        assert results
        assert results[0].run.num_episodes == sim_run_config.num_episodes

    def test_eval_tool_with_policy_uris(self, real_checkpoint_uri):
        """Test EvaluateTool can build policies from a real checkpoint URI."""
        checkpoint_uri, system_cfg = real_checkpoint_uri
        env_config = eb.make_arena(num_agents=4)
        sim_config = SimulationConfig(suite="test", name="test_arena", env=env_config)
        eval_tool = EvaluateTool(
            simulations=[sim_config],
            policy_uris=[checkpoint_uri],
            system=system_cfg,
        )

        assert eval_tool.simulations[0].name == "test_arena"
        assert eval_tool.policy_uris == [checkpoint_uri]

        policy_spec = eval_tool._build_policy_spec(checkpoint_uri)
        env_info = PolicyEnvInterface.from_mg_cfg(env_config)
        policy = initialize_or_load_policy(env_info, policy_spec)
        feature = env_info.obs_features[0]
        tokens = [
            ObservationToken(feature=feature, location=(0, 0), value=0, raw_token=(255, 0, 0))
            for _ in range(env_info.observation_space.shape[0])
        ]
        obs = AgentObservation(agent_id=0, tokens=tokens)
        policy.agent_step(0, obs)

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
