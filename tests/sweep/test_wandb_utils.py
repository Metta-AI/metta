"""Tests for wandb sweep utilities."""

from unittest.mock import Mock, patch

from metta.sweep.wandb_utils import sweep_id_from_name


class TestSweepIdFromName:
    """Test the sweep_id_from_name function."""

    @patch("wandb.Api")
    def test_sweep_id_from_name_found(self, mock_api_class):
        """Test successfully finding a sweep by name."""
        # Mock the API and project
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        mock_project = Mock()
        mock_api.project.return_value = mock_project

        # Mock sweeps
        mock_sweep = Mock()
        mock_sweep.name = "test_sweep"
        mock_sweep.id = "sweep123"

        mock_project.sweeps.return_value = [
            Mock(name="other_sweep", id="other123"),
            mock_sweep,
            Mock(name="another_sweep", id="another123"),
        ]

        # Call the function
        result = sweep_id_from_name("test_project", "test_entity", "test_sweep")

        assert result == "sweep123"
        mock_api.project.assert_called_once_with("test_project", "test_entity")

    @patch("wandb.Api")
    def test_sweep_id_from_name_not_found(self, mock_api_class):
        """Test when sweep is not found."""
        # Mock the API and project
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        mock_project = Mock()
        mock_api.project.return_value = mock_project

        # Mock sweeps without the target sweep
        mock_project.sweeps.return_value = [
            Mock(name="other_sweep", id="other123"),
            Mock(name="another_sweep", id="another123"),
        ]

        # Call the function
        result = sweep_id_from_name("test_project", "test_entity", "test_sweep")

        assert result is None

    @patch("wandb.Api")
    def test_sweep_id_from_name_project_not_found(self, mock_api_class):
        """Test when project doesn't exist."""
        # Mock the API
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        # Make project access fail
        mock_api.project.side_effect = Exception("Project not found")

        # Call the function
        result = sweep_id_from_name("nonexistent_project", "test_entity", "test_sweep")

        assert result is None

    @patch("wandb.Api")
    def test_sweep_id_from_name_with_network_retry(self, mock_api_class):
        """Test that network errors trigger retries."""
        # This test verifies that the retry decorator works by mocking
        # the entire API to fail and then succeed

        # We'll simulate the retry behavior by having the API constructor
        # fail initially, then succeed
        call_count = 0

        def api_constructor_side_effect():
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                raise Exception(f"Network error {call_count}")

            # On the third call, return a successful mock
            mock_api = Mock()
            mock_project = Mock()
            mock_api.project.return_value = mock_project

            mock_sweep = Mock()
            mock_sweep.name = "test_sweep"
            mock_sweep.id = "sweep123"
            mock_project.sweeps.return_value = [mock_sweep]

            return mock_api

        mock_api_class.side_effect = api_constructor_side_effect

        # Call the function with minimal retry delay by patching sleep
        with patch("time.sleep"):
            result = sweep_id_from_name("test_project", "test_entity", "test_sweep")

        # Should succeed after retries
        assert result == "sweep123"
        assert call_count == 3  # Should have been called 3 times

    @patch("wandb.Api")
    def test_sweep_id_from_name_all_retries_fail(self, mock_api_class):
        """Test that function returns None when all retries fail."""
        # Mock the API to always fail
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_api.project.side_effect = Exception("Persistent network error")

        # Call the function with minimal retry delay
        with patch("time.sleep"):  # Mock sleep to speed up test
            result = sweep_id_from_name("test_project", "test_entity", "test_sweep")

        assert result is None
