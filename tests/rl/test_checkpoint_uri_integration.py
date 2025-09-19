"""Integration tests for CheckpointManager URI handling.

Tests file and S3 URI formats plus real environment integration.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from tensordict import TensorDict

import mettagrid.builder.envs as eb
from metta.agent.agent_config import PolicyArchitectureConfig
from metta.agent.metta_agent import MettaAgent
from metta.agent.mocks import MockAgent
from metta.agent.utils import obs_to_td
from metta.rl.checkpoint_manager import CheckpointManager, key_and_version
from metta.rl.system_config import SystemConfig
from mettagrid import MettaGridEnv


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
    agent_cfg = PolicyArchitectureConfig(name="fast")

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


def checkpoint_filename(run: str, epoch: int) -> str:
    return f"{run}:v{epoch}.pt"


class TestFileURIHandling:
    """Test file:// URI format handling."""

    def test_file_uri_single_checkpoint(self, temp_dir, mock_policy):
        """Test loading a single checkpoint file via file:// URI."""
        checkpoint_file = create_test_checkpoint(temp_dir, checkpoint_filename("solo_run", 5), mock_policy)

        uri = f"file://{checkpoint_file}"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert loaded_policy is not None
        assert type(loaded_policy).__name__ == type(mock_policy).__name__

    def test_file_uri_directory_with_checkpoints(self, temp_dir, mock_policy):
        """Test loading from directory containing multiple checkpoints."""
        checkpoints_dir = temp_dir / "run1" / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create multiple checkpoint files with different epochs
        create_test_checkpoint(checkpoints_dir, checkpoint_filename("run1", 1), mock_policy)
        create_test_checkpoint(checkpoints_dir, checkpoint_filename("run1", 3), mock_policy)
        create_test_checkpoint(checkpoints_dir, checkpoint_filename("run1", 5), mock_policy)

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

        create_test_checkpoint(checkpoints_dir, checkpoint_filename("my_run", 10), mock_policy)

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


class TestS3URIHandling:
    """Test s3:// URI format handling."""

    @patch("mettagrid.util.file.local_copy")
    def test_s3_uri_loading(self, mock_local_copy, mock_policy):
        """Test S3 URI handling through CheckpointManager."""
        mock_local_path = "/tmp/downloaded_checkpoint.pt"
        # Properly mock the context manager
        mock_local_copy.return_value.__enter__ = Mock(return_value=mock_local_path)
        mock_local_copy.return_value.__exit__ = Mock(return_value=None)

        with patch("torch.load", return_value=mock_policy) as mock_torch_load:
            # Also patch the specific import path used in checkpoint_manager
            with patch("metta.rl.checkpoint_manager.local_copy", mock_local_copy):
                uri = "s3://my-bucket/path/to/test_run:v5.pt"
                loaded_policy = CheckpointManager.load_from_uri(uri)

                assert type(loaded_policy).__name__ == type(mock_policy).__name__
                mock_local_copy.assert_called_once_with(uri)
                mock_torch_load.assert_called_once_with(mock_local_path, weights_only=False, map_location="cpu")

    def test_s3_key_and_version_extraction(self):
        """Test extracting key and version from S3 URIs."""
        # Test S3 URI with valid checkpoint filename
        uri = "s3://bucket/test_run/checkpoints/test_run:v15.pt"
        key, version = key_and_version(uri)
        assert key == "test_run"
        assert version == 15

        # Test S3 URI with regular filename
        uri = "s3://bucket/path/to/regular_model.pt"
        key, version = key_and_version(uri)
        assert key == "regular_model"
        assert version == 0


class TestCheckpointURINormalization:
    """Test URI normalization and validation."""

    def test_normalize_uri_function(self):
        """Test URI normalization functionality."""
        # Test plain path to file URI conversion
        result = CheckpointManager.normalize_uri("/path/to/checkpoint.pt")
        assert result.startswith("file://")
        assert result.endswith("/path/to/checkpoint.pt")

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
    """Test CheckpointManager with real MettaGrid environments."""

    def test_save_and_load_real_agent(self, create_env_and_agent):
        """Test saving and loading a real MettaAgent."""
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
            checkpoint_uris = checkpoint_manager.select_checkpoints(strategy="latest", count=1)
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
        """Test saving multiple checkpoints and selecting best."""
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

            # Latest checkpoint should reflect highest epoch
            latest_checkpoints = checkpoint_manager.select_checkpoints("latest", count=1)
            assert len(latest_checkpoints) == 1
            assert latest_checkpoints[0].endswith("progress_test:v20.pt")

            # Request all checkpoints and verify order (descending epoch)
            all_checkpoints = checkpoint_manager.select_checkpoints("all")
            expected_suffixes = [
                "progress_test:v20.pt",
                "progress_test:v15.pt",
                "progress_test:v10.pt",
                "progress_test:v5.pt",
            ]
            assert [uri.split("/")[-1] for uri in all_checkpoints] == expected_suffixes


class TestEndToEndWorkflows:
    """Test complete training and evaluation workflows."""

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
        run_dir = temp_dir / "cross_test" / "checkpoints"
        checkpoint_file = create_test_checkpoint(run_dir, checkpoint_filename("cross_test", 5), mock_agent)

        # Test different ways to reference the same checkpoint
        uris = [
            f"file://{checkpoint_file}",
            f"file://{checkpoint_file.parent}",  # checkpoints directory form
            f"file://{run_dir.parent}",  # run directory
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
