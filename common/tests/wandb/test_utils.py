"""Tests for common W&B utility functions."""

import tempfile
from unittest.mock import Mock, patch

from metta.common.wandb.utils import (
    get_wandb_artifact_metadata,
    upload_file_as_artifact,
)


class TestArtifactOperations:
    """Test artifact operations with minimal mocking."""

    def test_get_artifact_metadata_non_wandb_uri(self):
        """Test that non-wandb URIs return None - no mocks needed!"""
        assert get_wandb_artifact_metadata("file:///path/to/file.pt") is None
        assert get_wandb_artifact_metadata("s3://bucket/file.pt") is None
        assert get_wandb_artifact_metadata("http://example.com/file") is None

    @patch("wandb.Artifact")
    def test_upload_file_basic(self, mock_artifact_class):
        """Test basic file upload functionality."""
        # Create a simple mock artifact
        mock_artifact = Mock()
        mock_artifact.version = "v1"
        mock_artifact.qualified_name = "test-project/test-artifact:v1"
        mock_artifact_class.return_value = mock_artifact

        # Create a fake run
        fake_run = Mock()
        fake_run.project = "test-project"

        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            result = upload_file_as_artifact(file_path=tmp_file.name, artifact_name="test-artifact", wandb_run=fake_run)

            # Verify the result
            assert result == "wandb://test-project/test-artifact:v1"

            # Verify artifact was created and logged
            mock_artifact_class.assert_called_once()
            fake_run.log_artifact.assert_called_once_with(mock_artifact)

    def test_upload_file_no_active_run(self):
        """Test upload returns None when no run is active."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            result = upload_file_as_artifact(file_path=tmp_file.name, artifact_name="test")
            assert result is None

    @patch("wandb.Artifact")
    def test_upload_file_with_metadata(self, mock_artifact_class):
        """Test that metadata is passed correctly."""
        metadata = {"epoch": 10, "accuracy": 0.95}

        fake_run = Mock()
        fake_run.project = "test"

        with tempfile.NamedTemporaryFile() as tmp_file:
            upload_file_as_artifact(
                file_path=tmp_file.name, artifact_name="model", metadata=metadata, wandb_run=fake_run
            )

            # Verify metadata was passed to artifact constructor
            mock_artifact_class.assert_called_once_with(name="model", type="model", metadata=metadata)


class TestErrorHandling:
    """Test error handling in key functions."""

    @patch("metta.common.wandb.utils.get_wandb_artifact")
    def test_get_artifact_metadata_handles_errors(self, mock_get_artifact):
        """Test metadata extraction handles errors gracefully."""
        mock_get_artifact.side_effect = Exception("API Error")

        result = get_wandb_artifact_metadata("wandb://entity/project/artifact:v1")
        assert result is None  # Should return None on error

    @patch("wandb.Artifact")
    def test_upload_continues_after_wait_error(self, mock_artifact_class):
        """Test upload continues even if wait() fails."""
        mock_artifact = Mock()
        mock_artifact.version = "v1"
        mock_artifact.wait.side_effect = Exception("Timeout")
        mock_artifact_class.return_value = mock_artifact

        fake_run = Mock()
        fake_run.project = "test"

        with tempfile.NamedTemporaryFile() as tmp_file:
            # Should not raise, just log error
            result = upload_file_as_artifact(file_path=tmp_file.name, artifact_name="test", wandb_run=fake_run)

            # Should still return the URI
            assert result == "wandb://test/test:v1"
