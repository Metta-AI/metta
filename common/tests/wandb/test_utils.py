"""Tests for common W&B utility functions."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from metta.common.wandb.context import WandbRun
from metta.common.wandb.utils import (
    abort_requested,
    expand_wandb_uri,
    get_wandb_artifact,
    get_wandb_artifact_metadata,
    get_wandb_run,
    load_artifact_file,
    upload_file_as_artifact,
)


class TestURIExpansion:
    """Test wandb URI expansion functionality."""

    def test_expand_wandb_uri_run_format(self):
        """Test expansion of short run format URIs."""
        # Test short run format - should use METTA_WANDB_ENTITY fallback
        short_uri = "wandb://run/my-experiment"
        expanded = expand_wandb_uri(short_uri)
        assert expanded == "wandb://metta-research/metta/model/my-experiment:latest"

        # Test short run format with version
        short_uri = "wandb://run/my-experiment:v10"
        expanded = expand_wandb_uri(short_uri)
        assert expanded == "wandb://metta-research/metta/model/my-experiment:v10"

    def test_expand_wandb_uri_sweep_format(self):
        """Test expansion of short sweep format URIs."""
        short_uri = "wandb://sweep/my-sweep"
        expanded = expand_wandb_uri(short_uri)
        assert expanded == "wandb://metta-research/metta/sweep_model/my-sweep:latest"

        # With version
        short_uri = "wandb://sweep/my-sweep:v5"
        expanded = expand_wandb_uri(short_uri)
        assert expanded == "wandb://metta-research/metta/sweep_model/my-sweep:v5"

    def test_expand_wandb_uri_full_format_unchanged(self):
        """Test that full format URIs remain unchanged."""
        full_uri = "wandb://entity/project/model/artifact:v1"
        expanded = expand_wandb_uri(full_uri)
        assert expanded == full_uri

    def test_expand_wandb_uri_custom_project(self):
        """Test URI expansion with custom default project."""
        short_uri = "wandb://run/my-run"
        expanded = expand_wandb_uri(short_uri, default_project="custom-project")
        assert expanded == "wandb://metta-research/custom-project/model/my-run:latest"

    def test_expand_wandb_uri_non_wandb_unchanged(self):
        """Test that non-wandb URIs are returned unchanged."""
        non_wandb_uris = ["file:///path/to/file.pt", "s3://bucket/path/to/model.pt", "/regular/file/path.pt"]
        for uri in non_wandb_uris:
            assert expand_wandb_uri(uri) == uri

    def test_expand_wandb_uri_edge_cases(self):
        """Test edge cases in URI expansion."""
        # Empty string after wandb://
        assert expand_wandb_uri("wandb://") == "wandb://"

        # Multiple colons in artifact name
        uri = "wandb://run/my:special:run:v1"
        expanded = expand_wandb_uri(uri)
        assert expanded.endswith("my:special:run:v1")

        # Special characters in names
        uri = "wandb://run/my-run_with.dots-and_underscores"
        expanded = expand_wandb_uri(uri)
        assert "my-run_with.dots-and_underscores" in expanded


class TestArtifactOperations:
    """Test artifact upload, download, and metadata operations."""

    def test_upload_file_as_artifact(self):
        """Test uploading a file as a wandb artifact."""
        mock_artifact = Mock()
        mock_artifact.qualified_name = "metta-research/metta/test-artifact:v1"
        mock_artifact.version = "v1"
        mock_artifact.wait = Mock()

        mock_run = Mock()
        mock_run.project = "metta"
        mock_run.log_artifact = Mock()

        with patch("wandb.Artifact", return_value=mock_artifact):
            with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
                result = upload_file_as_artifact(
                    file_path=tmp_file.name, artifact_name="test-artifact", wandb_run=mock_run
                )

                assert result == "wandb://metta/test-artifact:v1"
                assert result.startswith("wandb://")

                # Verify the artifact upload happened
                mock_run.log_artifact.assert_called_once_with(mock_artifact)
                mock_artifact.wait.assert_called_once()

    def test_upload_file_with_metadata(self):
        """Test uploading a file with metadata."""
        mock_artifact = Mock()
        mock_artifact.version = "v2"
        mock_artifact.wait = Mock()

        mock_run = Mock()
        mock_run.project = "test-project"
        mock_run.log_artifact = Mock()

        metadata = {"epoch": 10, "score": 0.95}

        with patch("wandb.Artifact") as MockArtifact:
            MockArtifact.return_value = mock_artifact
            with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
                result = upload_file_as_artifact(
                    file_path=tmp_file.name,
                    artifact_name="test-model",
                    artifact_type="model",
                    metadata=metadata,
                    wandb_run=mock_run,
                )

                # Check artifact was created with correct parameters
                MockArtifact.assert_called_once_with(name="test-model", type="model", metadata=metadata)
                assert result == "wandb://test-project/test-model:v2"

    def test_upload_file_with_additional_files(self):
        """Test uploading multiple files to an artifact."""
        mock_artifact = Mock()
        mock_artifact.version = "v1"
        mock_artifact.wait = Mock()
        mock_artifact.add_file = Mock()

        mock_run = Mock()
        mock_run.project = "metta"
        mock_run.log_artifact = Mock()

        with patch("wandb.Artifact", return_value=mock_artifact):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create main and additional files
                main_file = Path(tmpdir) / "model.pt"
                config_file = Path(tmpdir) / "config.yaml"
                main_file.write_text("model")
                config_file.write_text("config")

                _result = upload_file_as_artifact(
                    file_path=str(main_file),
                    artifact_name="model-with-config",
                    wandb_run=mock_run,
                    additional_files=[str(config_file)],
                )

                # Should have added both files
                assert mock_artifact.add_file.call_count == 2
                mock_artifact.add_file.assert_any_call(str(main_file), name="model.pt")
                mock_artifact.add_file.assert_any_call(str(config_file))

    def test_upload_file_no_active_run(self):
        """Test upload behavior when no wandb run is active."""
        with patch("wandb.run", None):
            with tempfile.NamedTemporaryFile() as tmp_file:
                result = upload_file_as_artifact(file_path=tmp_file.name, artifact_name="test")
                assert result is None

    @patch("metta.common.wandb.utils.get_wandb_artifact")
    def test_get_artifact_metadata(self, mock_get_artifact):
        """Test extracting metadata from an artifact."""
        mock_artifact = Mock()
        mock_artifact.metadata = {"key": "value", "epoch": 5}
        mock_get_artifact.return_value = mock_artifact

        metadata = get_wandb_artifact_metadata("wandb://entity/project/artifact:v1")

        assert metadata == {"key": "value", "epoch": 5}
        mock_get_artifact.assert_called_once()

    @patch("metta.common.wandb.utils.get_wandb_artifact")
    def test_get_artifact_metadata_none(self, mock_get_artifact):
        """Test handling when artifact has no metadata."""
        mock_artifact = Mock()
        mock_artifact.metadata = None
        mock_get_artifact.return_value = mock_artifact

        metadata = get_wandb_artifact_metadata("wandb://entity/project/artifact:v1")
        assert metadata is None

    def test_get_artifact_metadata_non_wandb_uri(self):
        """Test that non-wandb URIs return None."""
        metadata = get_wandb_artifact_metadata("file:///path/to/file.pt")
        assert metadata is None


class TestArtifactLoading:
    """Test loading files from artifacts."""

    @patch("metta.common.wandb.utils.get_wandb_artifact")
    @patch("metta.common.wandb.utils.download_artifact")
    def test_load_artifact_file(self, mock_download, mock_get_artifact):
        """Test loading a specific file from an artifact."""
        mock_artifact = Mock()
        mock_get_artifact.return_value = mock_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock downloaded file
            target_file = Path(tmpdir) / "model.pt"
            target_file.write_text("model content")

            # Mock the download to create the file
            def create_file(_artifact, root):
                (Path(root) / "model.pt").write_text("model content")

            mock_download.side_effect = create_file

            # Test loading
            loaded_path = load_artifact_file("wandb://run/test", filename="model.pt")

            assert loaded_path.exists()
            assert loaded_path.suffix == ".pt"

    @patch("metta.common.wandb.utils.get_wandb_artifact")
    @patch("metta.common.wandb.utils.download_artifact")
    def test_load_artifact_file_with_pattern(self, mock_download, mock_get_artifact):
        """Test loading file using fallback pattern."""
        mock_artifact = Mock()
        mock_get_artifact.return_value = mock_artifact

        with tempfile.TemporaryDirectory() as _:
            # Mock download creates multiple .pt files
            def create_files(artifact, root):
                (Path(root) / "checkpoint_1.pt").write_text("checkpoint 1")
                (Path(root) / "checkpoint_2.pt").write_text("checkpoint 2")

            mock_download.side_effect = create_files

            # Should load first matching file
            loaded_path = load_artifact_file("wandb://run/test", fallback_pattern="*.pt")

            assert loaded_path.exists()
            assert loaded_path.suffix == ".pt"


class TestRunManagement:
    """Test run management utilities."""

    @patch("metta.common.wandb.utils.get_wandb_run")
    def test_abort_requested_with_abort_tag(self, mock_get_run):
        """Test abort detection when abort tag is present."""
        mock_run_obj = Mock()
        mock_run_obj.tags = ["training", "abort", "experiment"]
        mock_get_run.return_value = mock_run_obj

        mock_wandb_run = Mock(spec=WandbRun)
        mock_wandb_run.path = "entity/project/run_id"

        assert abort_requested(mock_wandb_run) is True
        mock_get_run.assert_called_once_with("entity/project/run_id")

    @patch("metta.common.wandb.utils.get_wandb_run")
    def test_abort_requested_without_abort_tag(self, mock_get_run):
        """Test abort detection when abort tag is not present."""
        mock_run_obj = Mock()
        mock_run_obj.tags = ["training", "experiment"]
        mock_get_run.return_value = mock_run_obj

        mock_wandb_run = Mock(spec=WandbRun)
        mock_wandb_run.path = "entity/project/run_id"

        assert abort_requested(mock_wandb_run) is False

    def test_abort_requested_no_run(self):
        """Test abort detection with no wandb run."""
        assert abort_requested(None) is False

    @patch("metta.common.wandb.utils.get_wandb_run")
    def test_abort_requested_api_error(self, mock_get_run):
        """Test abort detection handles API errors gracefully."""
        mock_get_run.side_effect = Exception("API Error")

        mock_wandb_run = Mock(spec=WandbRun)
        mock_wandb_run.path = "entity/project/run_id"

        # Should return False on API error (don't abort on errors)
        assert abort_requested(mock_wandb_run) is False


class TestRetryMechanism:
    """Test the retry mechanism for API calls."""

    @patch("metta.common.wandb.utils.wandb_api.run")
    def test_get_wandb_run_with_retry(self, mock_api_run):
        """Test that get_wandb_run retries on failure."""
        # First two calls fail, third succeeds
        mock_api_run.side_effect = [ConnectionError("Network error"), TimeoutError("Timeout"), Mock(tags=["test"])]

        result = get_wandb_run("test/path")
        assert result.tags == ["test"]
        assert mock_api_run.call_count == 3

    @patch("metta.common.wandb.utils.wandb_api.artifact")
    def test_get_wandb_artifact_with_retry(self, mock_api_artifact):
        """Test that get_wandb_artifact retries on failure."""
        # First call fails, second succeeds
        mock_artifact = Mock()
        mock_api_artifact.side_effect = [ConnectionError("Network error"), mock_artifact]

        result = get_wandb_artifact("test:v1")
        assert result == mock_artifact
        assert mock_api_artifact.call_count == 2
