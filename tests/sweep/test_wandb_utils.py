"""Tests for wandb sweep utilities."""

from unittest.mock import Mock, patch

from metta.sweep.wandb_utils import sweep_id_from_name


class TestSweepIdFromName:
    """Test sweep_id_from_name function."""

    @patch("metta.sweep.wandb_utils.wandb.Api")
    def test_sweep_id_from_name_found(self, mock_api_class):
        """Test that function returns sweep name directly (deprecated function)."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Call function
        result = sweep_id_from_name("test_entity", "test_project", "test_sweep")

        # Should return sweep name directly
        assert result == "test_sweep"
        # API should not be called since we're just returning the name
        mock_api_class.assert_not_called()

    @patch("metta.sweep.wandb_utils.wandb.Api")
    def test_sweep_id_from_name_not_found(self, mock_api_class):
        """Test that function returns sweep name even when 'not found' (deprecated)."""
        # Setup mocks
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Call function
        result = sweep_id_from_name("test_entity", "test_project", "test_sweep")

        # Should still return sweep name
        assert result == "test_sweep"
        mock_api_class.assert_not_called()

    @patch("metta.sweep.wandb_utils.wandb.Api")
    def test_sweep_id_from_name_project_not_found(self, mock_api_class):
        """Test that function returns sweep name regardless (deprecated)."""
        # Call function
        result = sweep_id_from_name("test_entity", "nonexistent_project", "test_sweep")

        # Should still return sweep name
        assert result == "test_sweep"

    @patch("metta.sweep.wandb_utils.wandb.Api")
    def test_sweep_id_from_name_with_network_retry(self, mock_api_class):
        """Test that function returns sweep name without retries (deprecated)."""
        # Call function
        result = sweep_id_from_name("test_entity", "test_project", "test_sweep")

        # Should return sweep name
        assert result == "test_sweep"

    @patch("metta.sweep.wandb_utils.wandb.Api")
    def test_sweep_id_from_name_all_retries_fail(self, mock_api_class):
        """Test that function returns sweep name without API calls (deprecated)."""
        # Call function
        result = sweep_id_from_name("test_entity", "test_project", "test_sweep")

        # Should still return sweep name
        assert result == "test_sweep"
