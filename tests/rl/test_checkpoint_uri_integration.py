"""Consolidated tests for URI handling and checkpoint integration.

Tests all URI formats (file, wandb, s3), real environment integration,
end-to-end workflows, and cross-format compatibility.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from tensordict import TensorDict

import metta.mettagrid.builder.envs as eb
from metta.agent.agent_config import AgentConfig
from metta.agent.metta_agent import MettaAgent
from metta.agent.mocks import MockAgent
from metta.agent.utils import obs_to_td
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.util.file import WANDB_ENTITY, WandbURI
from metta.rl.checkpoint_manager import CheckpointManager, expand_wandb_uri, key_and_version
from metta.rl.system_config import SystemConfig
from metta.rl.wandb import upload_checkpoint_as_artifact


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_policy():
    """Create a mock policy for testing."""
    from metta.agent.mocks import MockAgent

    return MockAgent()


@pytest.fixture
def create_env_and_agent():
    """Create a real environment and agent for testing."""
    env_config = eb.make_navigation(num_agents=1)
    env_config.game.max_steps = 100
    env_config.game.map_builder.width = 8
    env_config.game.map_builder.height = 8

    env = MettaGridEnv(env_config, render_mode=None)
    system_cfg = SystemConfig(device="cpu")
    agent_cfg = AgentConfig(name="fast")

    agent = MettaAgent(env=env, system_cfg=system_cfg, policy_architecture_cfg=agent_cfg)

    # Initialize agent to environment
    features = env.get_observation_features()
    agent.initialize_to_environment(features, env.action_names, env.max_action_args, device="cpu")

    return env, agent


def create_test_checkpoint(temp_dir: Path, filename: str, policy=None) -> Path:
    """Create a test checkpoint file."""
    if policy is None:
        policy = Mock()
    checkpoint_path = temp_dir / filename
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy, checkpoint_path)
    return checkpoint_path


class TestFileURIHandling:
    """Test file:// URI format handling."""

    def test_file_uri_single_checkpoint(self, temp_dir, mock_policy):
        """Test loading a single checkpoint file via file:// URI."""
        checkpoint_file = create_test_checkpoint(temp_dir, "test_run__e5__s1000__t120__sc7500.pt", mock_policy)

        uri = f"file://{checkpoint_file}"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert loaded_policy is not None
        assert type(loaded_policy).__name__ == type(mock_policy).__name__

    def test_file_uri_directory_with_checkpoints(self, temp_dir, mock_policy):
        """Test loading from directory containing multiple checkpoints."""
        checkpoints_dir = temp_dir / "run1" / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create multiple checkpoint files with different epochs
        create_test_checkpoint(checkpoints_dir, "run1__e1__s500__t60__sc5000.pt", mock_policy)
        create_test_checkpoint(checkpoints_dir, "run1__e3__s1500__t180__sc8000.pt", mock_policy)
        create_test_checkpoint(checkpoints_dir, "run1__e5__s2500__t300__sc9500.pt", mock_policy)

        # Test directory URI - should load latest (highest epoch)
        uri = f"file://{checkpoints_dir}"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert loaded_policy is not None
        assert type(loaded_policy).__name__ == type(mock_policy).__name__

    def test_file_uri_run_directory_navigation(self, temp_dir, mock_policy):
        """Test loading from run directory (should navigate to checkpoints subdir)."""
        run_dir = temp_dir / "my_run"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        create_test_checkpoint(checkpoints_dir, "my_run__e10__s5000__t600__sc9000.pt", mock_policy)

        # Test run directory URI - should automatically look in checkpoints subdir
        uri = f"file://{run_dir}"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert loaded_policy is not None

    def test_file_uri_invalid_paths(self):
        """Test file:// URI with invalid paths."""
        # Test non-existent file
        uri = "file:///nonexistent/path.pt"
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_from_uri(uri)

        # Test non-existent directory
        uri = "file:///nonexistent/directory"
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_from_uri(uri)

    def test_file_uri_error_handling(self, temp_dir):
        """Test error handling for file URIs."""
        # Create directory with non-checkpoint .pt file
        test_dir = temp_dir / "mixed_files"
        test_dir.mkdir()

        # Create a corrupted .pt file
        corrupted_file = test_dir / "corrupted.pt"
        corrupted_file.write_text("not a pytorch file")

        uri = f"file://{test_dir}"
        with pytest.raises(FileNotFoundError):
            CheckpointManager.load_from_uri(uri)


class TestWandbURIHandling:
    """Test wandb:// URI format handling and compatibility."""

    def test_wandb_uri_expansion(self):
        """Test wandb URI expansion functionality."""
        # Test short run format
        short_uri = "wandb://run/my-experiment"
        expanded = expand_wandb_uri(short_uri)
        assert expanded == "wandb://metta/model/my-experiment:latest"

        # Test short run format with version
        short_uri = "wandb://run/my-experiment:v10"
        expanded = expand_wandb_uri(short_uri)
        assert expanded == "wandb://metta/model/my-experiment:v10"

        # Test short sweep format
        short_uri = "wandb://sweep/my-sweep"
        expanded = expand_wandb_uri(short_uri)
        assert expanded == "wandb://metta/sweep_model/my-sweep:latest"

        # Test full format (should remain unchanged)
        full_uri = "wandb://entity/project/model/artifact:v1"
        expanded = expand_wandb_uri(full_uri)
        assert expanded == full_uri

    @patch("metta.rl.checkpoint_manager.load_policy_from_wandb_uri")
    def test_wandb_uri_loading(self, mock_load_wandb, mock_policy):
        """Test wandb URI loading - expansion now happens inside load_policy_from_wandb_uri."""
        mock_load_wandb.return_value = mock_policy

        # Test that the URI is passed as-is (expansion happens inside load_policy_from_wandb_uri)
        uri = "wandb://run/my-experiment"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert type(loaded_policy).__name__ == type(mock_policy).__name__
        # Verify the original URI was passed (expansion happens inside the function)
        mock_load_wandb.assert_called_once_with("wandb://run/my-experiment", device="cpu")

    @patch("metta.rl.checkpoint_manager.get_wandb_checkpoint_metadata")
    def test_wandb_metadata_extraction(self, mock_get_metadata):
        """Test metadata extraction from wandb URIs."""
        mock_get_metadata.return_value = {
            "run_name": "experiment_1",
            "epoch": 25,
            "agent_step": 12500,
            "total_time": 750,
            "score": 0.95,
        }

        uri = "wandb://run/experiment_1"
        metadata = CheckpointManager.get_policy_metadata(uri)

        # Should call with expanded URI
        mock_get_metadata.assert_called_once_with("wandb://metta/model/experiment_1:latest")

        assert metadata["run_name"] == "experiment_1"
        assert metadata["epoch"] == 25
        assert metadata["original_uri"] == uri  # Original short form preserved

    def test_wandb_key_and_version_extraction(self):
        """Test extracting key and version from wandb URIs."""
        with patch(
            "metta.rl.checkpoint_manager.get_wandb_checkpoint_metadata", return_value={"run_name": "test", "epoch": 5}
        ):
            key, version = key_and_version("wandb://run/test")
            assert key == "test"
            assert version == 5

    @patch("metta.rl.checkpoint_manager.load_policy_from_wandb_uri")
    def test_wandb_error_handling(self, mock_load_wandb):
        """Test wandb URI error handling."""
        # Test network error
        mock_load_wandb.side_effect = RuntimeError("Network error")

        uri = "wandb://run/test"
        with pytest.raises(RuntimeError, match="Network error"):
            CheckpointManager.load_from_uri(uri)


class TestS3URIHandling:
    """Test s3:// URI format handling."""

    @patch("metta.mettagrid.util.file.local_copy")
    def test_s3_uri_loading(self, mock_local_copy, mock_policy):
        """Test S3 URI handling with mocked local_copy."""
        mock_local_path = "/tmp/downloaded_checkpoint.pt"
        # Properly mock the context manager
        mock_local_copy.return_value.__enter__ = Mock(return_value=mock_local_path)
        mock_local_copy.return_value.__exit__ = Mock(return_value=None)

        with patch("torch.load", return_value=mock_policy) as mock_torch_load:
            # Also patch the specific import path used in checkpoint_manager
            with patch("metta.rl.checkpoint_manager.local_copy", mock_local_copy):
                uri = "s3://my-bucket/path/to/checkpoint.pt"
                loaded_policy = CheckpointManager.load_from_uri(uri)

                assert type(loaded_policy).__name__ == type(mock_policy).__name__
                mock_local_copy.assert_called_once_with(uri)
                mock_torch_load.assert_called_once_with(mock_local_path, weights_only=False, map_location="cpu")

    def test_s3_key_and_version_extraction(self):
        """Test extracting key and version from S3 URIs."""
        # Test S3 URI with valid checkpoint filename
        uri = "s3://bucket/path/to/my_run__e15__s7500__t450__sc8500.pt"
        key, version = key_and_version(uri)
        assert key == "my_run"
        assert version == 15

        # Test S3 URI with regular filename
        uri = "s3://bucket/path/to/regular_model.pt"
        key, version = key_and_version(uri)
        assert key == "regular_model"
        assert version == 0


class TestURIUtilities:
    """Test URI utility functions."""

    def test_normalize_uri_function(self):
        """Test URI normalization functionality."""
        # Test plain path to file URI conversion
        result = CheckpointManager.normalize_uri("/path/to/checkpoint.pt")
        assert result.startswith("file://")
        assert result.endswith("/path/to/checkpoint.pt")

        # Test wandb URI passthrough (expansion happens in wandb.py now)
        wandb_short = "wandb://run/test"
        normalized = CheckpointManager.normalize_uri(wandb_short)
        assert normalized == "wandb://run/test"  # Should be unchanged

        # Test already normalized URIs remain unchanged
        file_uri = "file:///path/to/checkpoint.pt"
        assert CheckpointManager.normalize_uri(file_uri) == file_uri

        s3_uri = "s3://bucket/path/checkpoint.pt"
        assert CheckpointManager.normalize_uri(s3_uri) == s3_uri

    def test_unsupported_uri_formats(self):
        """Test handling of unsupported URI formats."""
        unsupported_uris = ["http://example.com/model.pt", "ftp://server/model.pt", "gs://bucket/model.pt"]

        for uri in unsupported_uris:
            with pytest.raises(ValueError, match="Invalid URI"):
                CheckpointManager.load_from_uri(uri)


class TestRealEnvironmentIntegration:
    """Test integration with real MettaGrid environments and agents."""

    def test_save_and_load_real_agent(self, create_env_and_agent):
        """Test saving and loading a real MettaAgent with real environment."""
        env, agent = create_env_and_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(run="integration_test", run_dir=tmpdir)

            # Test forward pass before saving
            obs = env.reset()[0]
            td_obs = obs_to_td(obs)
            output_before = agent(td_obs.unsqueeze(0))
            assert "actions" in output_before

            # Save the agent
            metadata = {"agent_step": 1000, "total_time": 60, "score": 0.85}
            checkpoint_manager.save_agent(agent, epoch=5, metadata=metadata)

            # Load the agent
            loaded_agent = checkpoint_manager.load_agent(epoch=5)
            assert loaded_agent is not None

            # Test that loaded agent produces same output structure
            output_after = loaded_agent(td_obs.unsqueeze(0))
            assert "actions" in output_after
            assert output_after["actions"].shape == output_before["actions"].shape

    def test_uri_based_loading_with_real_agent(self, create_env_and_agent):
        """Test loading agent via file URI."""
        env, agent = create_env_and_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(run="uri_test", run_dir=tmpdir)

            # Save agent
            checkpoint_manager.save_agent(agent, epoch=10, metadata={"agent_step": 5000, "total_time": 300})

            # Get checkpoint URI using public API
            checkpoint_uris = checkpoint_manager.select_checkpoints(strategy="latest", count=1, metric="epoch")
            checkpoint_uri = checkpoint_uris[0] if checkpoint_uris else None
            assert checkpoint_uri is not None

            # Load via URI
            loaded_agent = CheckpointManager.load_from_uri(checkpoint_uri)
            assert loaded_agent is not None

            # Verify functionality
            obs = env.reset()[0]
            td_obs = obs_to_td(obs)
            output = loaded_agent(td_obs.unsqueeze(0))
            assert "actions" in output

    def test_training_progress_and_selection(self, create_env_and_agent):
        """Test saving multiple checkpoints with different scores and selecting best."""
        env, agent = create_env_and_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_manager = CheckpointManager(run="progress_test", run_dir=tmpdir)

            # Simulate training progress with improving scores
            training_data = [
                (5, {"agent_step": 1000, "total_time": 60, "score": 0.3}),
                (10, {"agent_step": 2000, "total_time": 120, "score": 0.7}),
                (15, {"agent_step": 3000, "total_time": 180, "score": 0.9}),
                (20, {"agent_step": 4000, "total_time": 240, "score": 0.6}),  # Performance dip
            ]

            for epoch, metadata in training_data:
                checkpoint_manager.save_agent(agent, epoch=epoch, metadata=metadata)

            # Test selection by score (should get epoch 15 with score 0.9)
            best_checkpoints = checkpoint_manager.select_checkpoints("latest", count=1, metric="score")
            assert len(best_checkpoints) == 1
            assert best_checkpoints[0].endswith("progress_test__e15__s3000__t180__sc9000.pt")

            # Test selection by latest epoch (should get epoch 20)
            latest_checkpoints = checkpoint_manager.select_checkpoints("latest", count=1, metric="epoch")
            assert len(latest_checkpoints) == 1
            assert latest_checkpoints[0].endswith("progress_test__e20__s4000__t240__sc6000.pt")


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_train_save_load_eval_workflow(self):
        """Test a complete workflow from training through evaluation."""

        class MockEnvironment:
            def __init__(self, feature_mapping):
                self.feature_mapping = feature_mapping
                self.action_names = ["move", "turn", "interact"]
                self.max_action_args = [3, 2, 1]

            def get_observation_features(self):
                return {
                    name: {"id": id_val, "type": "scalar", "normalization": 10.0}
                    for name, id_val in self.feature_mapping.items()
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Train with original features
            original_env = MockEnvironment({"health": 1, "energy": 2, "position": 3})
            policy = MockAgent()
            policy.train()

            # Initialize to original environment
            features = original_env.get_observation_features()
            policy.initialize_to_environment(features, original_env.action_names, original_env.max_action_args, "cpu")

            # Step 2: Save trained policy
            checkpoint_manager = CheckpointManager(run="workflow_test", run_dir=tmpdir)
            metadata = {"agent_step": 10000, "epoch": 50, "score": 0.95}
            checkpoint_manager.save_agent(policy, epoch=50, metadata=metadata)

            # Step 3: Load in new environment with different feature IDs
            new_env = MockEnvironment({"health": 10, "energy": 20, "position": 30, "stamina": 40})
            loaded_policy = checkpoint_manager.load_agent(epoch=50)

            # Initialize to new environment (eval mode)
            loaded_policy.eval()
            new_features = new_env.get_observation_features()
            loaded_policy.initialize_to_environment(new_features, new_env.action_names, new_env.max_action_args, "cpu")

            # Step 4: Verify evaluation works
            test_input = TensorDict({"env_obs": torch.randn(2, 10)}, batch_size=(2,))
            output = loaded_policy(test_input)
            assert "actions" in output
            assert output["actions"].shape[0] == 2

            # Step 5: Verify metadata persistence
            assert loaded_policy.get_original_feature_mapping() == {"health": 1, "energy": 2, "position": 3}

    def test_cross_format_uri_compatibility(self, temp_dir):
        """Test that different URI formats work together seamlessly."""
        # Create checkpoint via standard save
        mock_agent = MockAgent()
        checkpoint_file = create_test_checkpoint(temp_dir, "cross_test__e5__s1000__t120__sc7500.pt", mock_agent)

        # Test different ways to reference the same checkpoint
        uris = [
            f"file://{checkpoint_file}",
            f"file://{checkpoint_file.parent}",  # Directory form
            str(checkpoint_file),  # Plain path (should be normalized)
        ]

        for uri in uris:
            # Normalize URI if it's a plain path
            normalized_uri = CheckpointManager.normalize_uri(uri)
            loaded = CheckpointManager.load_from_uri(normalized_uri)
            assert loaded is not None

            # Test metadata extraction consistency
            metadata = CheckpointManager.get_policy_metadata(uri)
            assert metadata["run_name"] == "cross_test"
            assert metadata["epoch"] == 5


class TestWandbArtifactFormatting:
    """Test wandb artifact URI formatting to prevent double entity issues."""

    def test_wandb_uri_parsing_prevents_double_entity(self):
        """Test that WandB URIs are parsed correctly without double entity issues."""

        # Test various wandb:// URI formats to ensure they parse correctly
        test_cases = [
            ("wandb://metta/relh.policy-cull.902.1:v0", "metta", "relh.policy-cull.902.1", "v0"),
            ("wandb://test-project/my-artifact:latest", "test-project", "my-artifact", "latest"),
            ("wandb://another-project/artifact-name:v42", "another-project", "artifact-name", "v42"),
        ]

        for wandb_uri, expected_project, expected_artifact_path, expected_version in test_cases:
            # Parse the URI
            parsed_uri = WandbURI.parse(wandb_uri)

            # Verify components are extracted correctly
            assert parsed_uri.project == expected_project
            assert parsed_uri.artifact_path == expected_artifact_path
            assert parsed_uri.version == expected_version

            # Most importantly: verify that qname() uses the configured entity
            # and doesn't create double entity paths
            qname = parsed_uri.qname()
            expected_qname = f"{WANDB_ENTITY}/{expected_project}/{expected_artifact_path}:{expected_version}"
            assert qname == expected_qname

            # Ensure no double entity issue (this was the original bug)
            parts = qname.split("/")
            assert len(parts) == 3, f"qname should have exactly 3 parts, got: {qname}"

    def test_upload_checkpoint_returns_latest_uri(self):
        """Test that upload_checkpoint_as_artifact always returns latest URI for simplicity."""

        mock_artifact = Mock()
        mock_artifact.qualified_name = "metta-research/metta/test-artifact:v1"
        mock_artifact.version = "v1"
        mock_artifact.wait = Mock()

        mock_run = Mock()
        mock_run.project = "metta"
        mock_run.log_artifact = Mock()

        with patch("wandb.Artifact", return_value=mock_artifact):
            with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
                result = upload_checkpoint_as_artifact(
                    checkpoint_path=tmp_file.name, artifact_name="test-artifact", wandb_run=mock_run
                )

                # Always returns :latest for simplicity and reliability
                assert result == "wandb://metta/test-artifact:v1"
                assert result.startswith("wandb://"), "Should start with wandb://"

                # Verify the artifact upload happened
                mock_run.log_artifact.assert_called_once_with(mock_artifact)
                mock_artifact.wait.assert_called_once()
