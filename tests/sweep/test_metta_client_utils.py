"""Tests for metta_client_utils functions."""

from unittest.mock import Mock, patch

import pytest

from metta.sweep.metta_client_utils import (
    create_sweep_in_metta,
    get_next_run_id_from_metta,
    get_sweep_id_from_metta,
)


class TestMettaClientUtils:
    """Test the centralized client utility functions."""

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_get_sweep_id_from_metta_exists(self, mock_client_class):
        """Test getting sweep ID when sweep exists."""
        # Mock client and response
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_info = Mock()
        mock_info.exists = True
        mock_info.wandb_sweep_id = "wandb_123"
        mock_client.get_sweep.return_value = mock_info

        result = get_sweep_id_from_metta("test_sweep")

        assert result == "wandb_123"
        mock_client.get_sweep.assert_called_once_with("test_sweep")

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_get_sweep_id_from_metta_not_exists(self, mock_client_class):
        """Test getting sweep ID when sweep doesn't exist."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_info = Mock()
        mock_info.exists = False
        mock_client.get_sweep.return_value = mock_info

        result = get_sweep_id_from_metta("nonexistent_sweep")

        assert result is None
        mock_client.get_sweep.assert_called_once_with("nonexistent_sweep")

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_get_next_run_id_from_metta(self, mock_client_class):
        """Test atomic run ID generation."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_next_run_id.return_value = "test_sweep.r.42"

        result = get_next_run_id_from_metta("test_sweep")

        assert result == "test_sweep.r.42"
        mock_client.get_next_run_id.assert_called_once_with("test_sweep")

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_get_next_run_id_from_metta_sequential_calls(self, mock_client_class):
        """Test that sequential calls return different run IDs."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_next_run_id.side_effect = ["test_sweep.r.0", "test_sweep.r.1", "test_sweep.r.2"]

        result1 = get_next_run_id_from_metta("test_sweep")
        result2 = get_next_run_id_from_metta("test_sweep")
        result3 = get_next_run_id_from_metta("test_sweep")

        assert result1 == "test_sweep.r.0"
        assert result2 == "test_sweep.r.1"
        assert result3 == "test_sweep.r.2"
        assert mock_client.get_next_run_id.call_count == 3

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_create_sweep_in_metta_new_sweep(self, mock_client_class):
        """Test creating a new sweep via client."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.created = True
        mock_response.sweep_id = "uuid-123"
        mock_client.create_sweep.return_value = mock_response

        result = create_sweep_in_metta("test_sweep", "entity", "project", "wandb_123")

        assert result == mock_response
        assert result.created is True
        assert result.sweep_id == "uuid-123"
        mock_client.create_sweep.assert_called_once_with("test_sweep", "project", "entity", "wandb_123")

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_create_sweep_in_metta_existing_sweep(self, mock_client_class):
        """Test creating a sweep that already exists (idempotent)."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.created = False  # Existing sweep
        mock_response.sweep_id = "existing-uuid-456"
        mock_client.create_sweep.return_value = mock_response

        result = create_sweep_in_metta("existing_sweep", "entity", "project", "wandb_456")

        assert result == mock_response
        assert result.created is False
        assert result.sweep_id == "existing-uuid-456"
        mock_client.create_sweep.assert_called_once_with("existing_sweep", "project", "entity", "wandb_456")

    @patch("metta.sweep.metta_client_utils.SweepClient")
    @patch("metta.sweep.metta_client_utils.get_machine_token")
    def test_sweep_client_initialization(self, mock_get_token, mock_client_class):
        """Test that SweepClient is initialized with correct default parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_get_token.return_value = "test_token"

        get_sweep_id_from_metta("test_sweep")

        # Verify SweepClient was initialized with default URL and machine token
        mock_client_class.assert_called_once_with(base_url="http://localhost:8000", auth_token="test_token")

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_error_propagation_get_sweep_id(self, mock_client_class):
        """Test that errors from SweepClient propagate correctly."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_sweep.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            get_sweep_id_from_metta("test_sweep")

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_error_propagation_get_next_run_id(self, mock_client_class):
        """Test that errors from get_next_run_id propagate correctly."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_next_run_id.side_effect = Exception("Sweep not found")

        with pytest.raises(Exception, match="Sweep not found"):
            get_next_run_id_from_metta("nonexistent_sweep")

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_error_propagation_create_sweep(self, mock_client_class):
        """Test that errors from create_sweep propagate correctly."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.create_sweep.side_effect = Exception("Authentication failed")

        with pytest.raises(Exception, match="Authentication failed"):
            create_sweep_in_metta("test_sweep", "entity", "project", "wandb_123")

    @patch("metta.sweep.metta_client_utils.SweepClient")
    def test_concurrent_run_id_generation_simulation(self, mock_client_class):
        """Test simulation of concurrent run ID generation (different clients)."""
        # Simulate multiple workers getting different run IDs
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_client_class.side_effect = [mock_client1, mock_client2]

        mock_client1.get_next_run_id.return_value = "test_sweep.r.5"
        mock_client2.get_next_run_id.return_value = "test_sweep.r.6"

        result1 = get_next_run_id_from_metta("test_sweep")
        result2 = get_next_run_id_from_metta("test_sweep")

        assert result1 == "test_sweep.r.5"
        assert result2 == "test_sweep.r.6"
        assert result1 != result2  # Verify they're different (no collision)
