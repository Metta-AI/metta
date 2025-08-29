"""Comprehensive tests for URI handling across all supported formats.

This test suite validates URI handling compatibility between the main branch
(PolicyStore) and richard-policy-cull branch (CheckpointManager) approaches,
with special focus on wandb URI support and comprehensive coverage of all
URI formats supported by CheckpointManager.load_from_uri().
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from metta.rl.checkpoint_manager import CheckpointManager, key_and_version, parse_checkpoint_filename


class TestURIHandlingComprehensive:
    """Test comprehensive URI handling across all supported formats."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_policy(self):
        """Create a mock policy for testing."""
        policy = Mock()
        policy.forward = Mock(return_value={"actions": torch.tensor([[1, 0]])})
        return policy

    def create_test_checkpoint(self, temp_dir: Path, filename: str, policy=None) -> Path:
        """Create a test checkpoint file."""
        if policy is None:
            policy = self.mock_policy
        checkpoint_path = temp_dir / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(policy, checkpoint_path)
        return checkpoint_path

    def test_file_uri_single_checkpoint(self, temp_dir, mock_policy):
        """Test loading a single checkpoint file via file:// URI."""
        # Create a valid checkpoint file with metadata in filename
        checkpoint_file = self.create_test_checkpoint(temp_dir, "test_run.e5.s1000.t120.sc7500.pt", mock_policy)

        # Test file:// URI format
        uri = f"file://{checkpoint_file}"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert loaded_policy is not None
        # Verify it's the same mock policy
        assert loaded_policy == mock_policy

    def test_file_uri_directory_with_checkpoints(self, temp_dir, mock_policy):
        """Test loading from directory containing multiple checkpoints."""
        # Create a checkpoints directory with multiple files
        checkpoints_dir = temp_dir / "run1" / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create multiple checkpoint files with different epochs
        self.create_test_checkpoint(checkpoints_dir, "run1.e1.s500.t60.sc5000.pt", mock_policy)
        self.create_test_checkpoint(checkpoints_dir, "run1.e3.s1500.t180.sc8000.pt", mock_policy)
        self.create_test_checkpoint(checkpoints_dir, "run1.e5.s2500.t300.sc9500.pt", mock_policy)

        # Test directory URI - should load latest (highest epoch)
        uri = f"file://{checkpoints_dir}"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert loaded_policy is not None
        assert loaded_policy == mock_policy

    def test_file_uri_run_directory_navigation(self, temp_dir, mock_policy):
        """Test loading from run directory (should navigate to checkpoints subdir)."""
        # Create run directory structure
        run_dir = temp_dir / "my_run"
        checkpoints_dir = run_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)

        # Create checkpoint in checkpoints subdirectory
        self.create_test_checkpoint(checkpoints_dir, "my_run.e10.s5000.t600.sc9000.pt", mock_policy)

        # Test run directory URI - should automatically look in checkpoints subdir
        uri = f"file://{run_dir}"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert loaded_policy is not None

    def test_file_uri_invalid_paths(self):
        """Test file:// URI with invalid paths."""
        # Test non-existent file
        uri = "file:///nonexistent/path.pt"
        result = CheckpointManager.load_from_uri(uri)
        assert result is None

        # Test non-existent directory
        uri = "file:///nonexistent/directory"
        result = CheckpointManager.load_from_uri(uri)
        assert result is None

    @patch("metta.mettagrid.util.file.local_copy")
    def test_s3_uri_handling(self, mock_local_copy, mock_policy):
        """Test S3 URI handling with mocked local_copy."""
        # Mock the local_copy context manager
        mock_local_path = "/tmp/downloaded_checkpoint.pt"
        mock_local_copy.return_value.__enter__ = Mock(return_value=mock_local_path)
        mock_local_copy.return_value.__exit__ = Mock(return_value=None)

        # Mock torch.load to return our mock policy
        with patch("torch.load", return_value=mock_policy) as mock_torch_load:
            uri = "s3://my-bucket/path/to/checkpoint.pt"
            loaded_policy = CheckpointManager.load_from_uri(uri)

            assert loaded_policy == mock_policy
            mock_local_copy.assert_called_once_with(uri)
            mock_torch_load.assert_called_once_with(mock_local_path, weights_only=False)

    def test_s3_uri_key_and_version_extraction(self):
        """Test extracting key and version from S3 URIs."""
        # Test S3 URI with valid checkpoint filename
        uri = "s3://bucket/path/to/my_run.e15.s7500.t450.sc8500.pt"
        key, version = key_and_version(uri)
        assert key == "my_run"
        assert version == 15

        # Test S3 URI with regular filename
        uri = "s3://bucket/path/to/regular_model.pt"
        key, version = key_and_version(uri)
        assert key == "regular_model"
        assert version == 0

    @patch("metta.rl.wandb.load_policy_from_wandb_uri")
    def test_wandb_uri_handling(self, mock_load_wandb, mock_policy):
        """Test basic wandb URI handling."""
        mock_load_wandb.return_value = mock_policy

        uri = "wandb://entity/project/model/artifact:v1"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert loaded_policy == mock_policy
        mock_load_wandb.assert_called_once_with(uri, device="cpu")

    @patch("metta.rl.wandb.get_wandb_checkpoint_metadata")
    def test_wandb_uri_key_and_version_extraction(self, mock_get_metadata):
        """Test extracting key and version from wandb URIs with metadata."""
        # Test with metadata available
        mock_get_metadata.return_value = {
            "run_name": "experiment_1",
            "epoch": 25,
            "agent_step": 12500,
            "total_time": 750,
            "score": 0.95,
        }

        uri = "wandb://entity/project/model/experiment_1:v25"
        key, version = key_and_version(uri)

        assert key == "experiment_1"
        assert version == 25
        mock_get_metadata.assert_called_once_with(uri)

    def test_wandb_uri_key_and_version_fallback(self):
        """Test wandb URI key/version extraction fallback when no metadata."""
        with patch("metta.rl.wandb.get_wandb_checkpoint_metadata", return_value=None):
            with patch("metta.mettagrid.util.file.WandbURI.parse") as mock_parse:
                # Mock the parsed WandbURI object
                mock_wandb_uri = Mock()
                mock_wandb_uri.artifact_path = "entity/project/model/my_artifact:v3"
                mock_parse.return_value = mock_wandb_uri

                uri = "wandb://entity/project/model/my_artifact:v3"
                key, version = key_and_version(uri)

                assert key == "my_artifact"  # Extracted from artifact path
                assert version == 0  # Fallback when no metadata

    def test_unsupported_uri_formats(self):
        """Test handling of unsupported URI formats."""
        # Test unsupported protocol
        with pytest.raises(ValueError, match="Unsupported URI format"):
            CheckpointManager.load_from_uri("http://example.com/model.pt")

        with pytest.raises(ValueError, match="Unsupported URI format"):
            CheckpointManager.load_from_uri("ftp://server/model.pt")

        with pytest.raises(ValueError, match="Unsupported URI format"):
            CheckpointManager.load_from_uri("gs://bucket/model.pt")

    def test_normalize_uri_function(self):
        """Test URI normalization functionality."""
        # Test plain path to file URI conversion
        result = CheckpointManager.normalize_uri("/path/to/checkpoint.pt")
        assert result.startswith("file://")
        assert result.endswith("/path/to/checkpoint.pt")

        # Test already normalized URIs remain unchanged
        file_uri = "file:///path/to/checkpoint.pt"
        assert CheckpointManager.normalize_uri(file_uri) == file_uri

        wandb_uri = "wandb://entity/project/model/artifact:v1"
        assert CheckpointManager.normalize_uri(wandb_uri) == wandb_uri

        s3_uri = "s3://bucket/path/checkpoint.pt"
        assert CheckpointManager.normalize_uri(s3_uri) == s3_uri


class TestWandbURICompatibility:
    """Test wandb URI compatibility focusing on formats supported by main branch."""

    @patch("metta.rl.wandb.load_policy_from_wandb_uri")
    def test_wandb_full_format_uri(self, mock_load_wandb):
        """Test full wandb URI format: wandb://entity/project/artifact_type/name:version"""
        mock_policy = Mock()
        mock_load_wandb.return_value = mock_policy

        uri = "wandb://my-entity/my-project/model/experiment-1:v15"
        loaded_policy = CheckpointManager.load_from_uri(uri)

        assert loaded_policy == mock_policy
        mock_load_wandb.assert_called_once_with(uri, device="cpu")

    @patch("metta.rl.wandb.load_policy_from_wandb_uri")
    def test_wandb_short_run_format(self, mock_load_wandb):
        """Test short wandb URI format: wandb://run/name[:version]"""
        mock_policy = Mock()
        mock_load_wandb.return_value = mock_policy

        # Test with version
        uri = "wandb://run/my-experiment:v10"
        loaded_policy = CheckpointManager.load_from_uri(uri)
        assert loaded_policy == mock_policy

        # Test without version
        uri = "wandb://run/my-experiment"
        loaded_policy = CheckpointManager.load_from_uri(uri)
        assert loaded_policy == mock_policy

    @patch("metta.rl.wandb.load_policy_from_wandb_uri")
    def test_wandb_short_sweep_format(self, mock_load_wandb):
        """Test short wandb URI format: wandb://sweep/name[:version]"""
        mock_policy = Mock()
        mock_load_wandb.return_value = mock_policy

        # Test with version
        uri = "wandb://sweep/my-sweep:v5"
        loaded_policy = CheckpointManager.load_from_uri(uri)
        assert loaded_policy == mock_policy

        # Test without version
        uri = "wandb://sweep/my-sweep"
        loaded_policy = CheckpointManager.load_from_uri(uri)
        assert loaded_policy == mock_policy

    @patch("metta.rl.wandb.get_wandb_checkpoint_metadata")
    def test_wandb_metadata_extraction_comprehensive(self, mock_get_metadata):
        """Test comprehensive metadata extraction from wandb URIs."""
        # Test complete metadata
        complete_metadata = {
            "run_name": "complex_experiment",
            "epoch": 42,
            "agent_step": 25000,
            "total_time": 1800,
            "score": 0.987,
        }
        mock_get_metadata.return_value = complete_metadata

        uri = "wandb://entity/project/model/complex_experiment:v42"
        metadata = CheckpointManager.get_policy_metadata(uri)

        expected_metadata = {
            "run_name": "complex_experiment",
            "epoch": 42,
            "agent_step": 25000,
            "total_time": 1800,
            "score": 0.987,
            "uri": uri,
        }

        # Check all expected fields are present
        for key, expected_value in expected_metadata.items():
            assert metadata[key] == expected_value

    @patch("metta.rl.wandb.get_wandb_checkpoint_metadata")
    def test_wandb_metadata_extraction_partial(self, mock_get_metadata):
        """Test metadata extraction when only partial metadata is available."""
        # Test with no metadata (returns None)
        mock_get_metadata.return_value = None

        uri = "wandb://entity/project/model/no_metadata:v1"
        metadata = CheckpointManager.get_policy_metadata(uri)

        # Should still return basic metadata with fallback values
        assert metadata["uri"] == uri
        assert "run_name" in metadata
        assert "epoch" in metadata

    def test_wandb_uri_error_conditions(self):
        """Test wandb URI handling under various error conditions."""
        # Test with invalid wandb URI (should be handled by load_policy_from_wandb_uri)
        with patch("metta.rl.wandb.load_policy_from_wandb_uri", return_value=None):
            uri = "wandb://invalid/format"
            result = CheckpointManager.load_from_uri(uri)
            assert result is None

        # Test with network error (should be handled gracefully)
        with patch("metta.rl.wandb.load_policy_from_wandb_uri", side_effect=RuntimeError("Network error")):
            uri = "wandb://entity/project/model/artifact:v1"
            with pytest.raises(RuntimeError):
                CheckpointManager.load_from_uri(uri)


class TestURIHandlingEdgeCases:
    """Test edge cases and error conditions in URI handling."""

    def test_malformed_file_uris(self):
        """Test handling of malformed file URIs."""
        # Test file URI without proper path
        with pytest.raises((IndexError, FileNotFoundError, ValueError)):
            CheckpointManager.load_from_uri("file://")

        # Test file URI with relative path (should still work)
        with patch("pathlib.Path.is_file", return_value=True):
            with patch("torch.load", return_value=Mock()):
                result = CheckpointManager.load_from_uri("file://./relative/path.pt")
                assert result is not None

    def test_empty_directory_handling(self, tmp_path):
        """Test handling of empty directories."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        uri = f"file://{empty_dir}"
        result = CheckpointManager.load_from_uri(uri)
        assert result is None

    def test_directory_with_non_checkpoint_files(self, tmp_path):
        """Test directory containing non-checkpoint .pt files."""
        test_dir = tmp_path / "mixed_files"
        test_dir.mkdir()

        # Create a non-checkpoint .pt file
        non_checkpoint = test_dir / "regular_model.pt"
        mock_policy = Mock()
        torch.save(mock_policy, non_checkpoint)

        # Should load the first .pt file found
        uri = f"file://{test_dir}"
        loaded_policy = CheckpointManager.load_from_uri(uri)
        assert loaded_policy is not None

    def test_parse_checkpoint_filename_edge_cases(self):
        """Test edge cases in checkpoint filename parsing."""
        # Test with zero score
        filename = "run.e0.s0.t0.sc0.pt"
        parsed = parse_checkpoint_filename(filename)
        assert parsed == ("run", 0, 0, 0, 0.0)

        # Test with maximum score (0.9999)
        filename = "run.e1.s1000.t300.sc9999.pt"
        parsed = parse_checkpoint_filename(filename)
        assert abs(parsed[4] - 0.9999) < 0.0001

        # Test with very large numbers
        filename = "run.e999.s999999.t86400.sc5000.pt"
        parsed = parse_checkpoint_filename(filename)
        assert parsed == ("run", 999, 999999, 86400, 0.5)

    def test_key_and_version_unknown_format(self):
        """Test key_and_version with unknown URI format."""
        # Should return default values for unknown formats
        key, version = key_and_version("unknown://protocol/path")
        assert key == "unknown"
        assert version == 0

        # Test with empty string
        key, version = key_and_version("")
        assert key == "unknown"
        assert version == 0


class TestURIHandlingIntegration:
    """Integration tests combining multiple URI handling features."""

    @pytest.fixture
    def comprehensive_test_setup(self, tmp_path):
        """Set up a comprehensive test environment with multiple URI types."""
        # Create file-based checkpoints
        file_checkpoints_dir = tmp_path / "file_checkpoints"
        file_checkpoints_dir.mkdir()

        mock_policy = Mock()
        checkpoint_files = []

        for epoch in [1, 3, 5]:
            filename = f"integration_test.e{epoch}.s{epoch * 1000}.t{epoch * 60}.sc{5000 + epoch * 500}.pt"
            checkpoint_path = file_checkpoints_dir / filename
            torch.save(mock_policy, checkpoint_path)
            checkpoint_files.append(checkpoint_path)

        return {
            "checkpoints_dir": file_checkpoints_dir,
            "checkpoint_files": checkpoint_files,
            "mock_policy": mock_policy,
        }

    def test_cross_uri_format_compatibility(self, comprehensive_test_setup):
        """Test that different URI formats can be used interchangeably."""
        setup = comprehensive_test_setup
        mock_policy = setup["mock_policy"]

        # Test file URI loading
        file_uri = f"file://{setup['checkpoint_files'][0]}"
        loaded_from_file = CheckpointManager.load_from_uri(file_uri)
        assert loaded_from_file == mock_policy

        # Test directory-based file URI
        dir_uri = f"file://{setup['checkpoints_dir']}"
        loaded_from_dir = CheckpointManager.load_from_uri(dir_uri)
        assert loaded_from_dir == mock_policy

    def test_metadata_consistency_across_formats(self, comprehensive_test_setup):
        """Test that metadata extraction is consistent across URI formats."""
        setup = comprehensive_test_setup

        # Test metadata from file URI
        file_uri = f"file://{setup['checkpoint_files'][0]}"  # epoch 1
        file_metadata = CheckpointManager.get_policy_metadata(file_uri)

        assert file_metadata["run_name"] == "integration_test"
        assert file_metadata["epoch"] == 1
        assert file_metadata["agent_step"] == 1000
        assert file_metadata["total_time"] == 60
        assert file_metadata["score"] == 0.55  # 5500/10000

        # Test key_and_version extraction
        key, version = key_and_version(file_uri)
        assert key == "integration_test"
        assert version == 1

    @patch("metta.rl.wandb.load_policy_from_wandb_uri")
    @patch("metta.rl.wandb.get_wandb_checkpoint_metadata")
    def test_uri_format_fallback_behavior(self, mock_get_metadata, mock_load_wandb, comprehensive_test_setup):
        """Test fallback behavior when primary methods fail."""
        setup = comprehensive_test_setup
        mock_policy = setup["mock_policy"]

        # Test wandb URI with metadata fallback
        mock_get_metadata.return_value = None  # No metadata available
        mock_load_wandb.return_value = mock_policy

        with patch("metta.mettagrid.util.file.WandbURI.parse") as mock_parse:
            mock_wandb_uri = Mock()
            mock_wandb_uri.artifact_path = "entity/project/model/fallback_test:v1"
            mock_parse.return_value = mock_wandb_uri

            uri = "wandb://entity/project/model/fallback_test:v1"

            # Should still load policy despite missing metadata
            loaded_policy = CheckpointManager.load_from_uri(uri)
            assert loaded_policy == mock_policy

            # Metadata should use fallback values
            metadata = CheckpointManager.get_policy_metadata(uri)
            assert metadata["run_name"] == "fallback_test"
            assert metadata["epoch"] == 0  # Fallback value

    def test_uri_normalization_integration(self):
        """Test URI normalization in real usage scenarios."""
        # Test that normalized URIs work with actual loading
        plain_path = "/tmp/test/checkpoint.pt"
        normalized_uri = CheckpointManager.normalize_uri(plain_path)

        assert normalized_uri.startswith("file://")
        assert plain_path in normalized_uri

        # The normalize function should create absolute paths
        assert normalized_uri.startswith("file:///") or normalized_uri.startswith("file://C:")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
