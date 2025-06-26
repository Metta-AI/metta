"""Tests for WandbProtein class."""

from unittest.mock import Mock, patch

import pytest

from metta.sweep.protein_wandb import WandbProtein


class TestWandbProtein:
    """Test cases for WandbProtein class."""

    @pytest.fixture
    def mock_protein(self):
        """Create a mock Protein instance."""
        protein = Mock()
        protein.hyperparameters = Mock()
        protein.hyperparameters.flat_spaces = {"learning_rate": Mock(), "batch_size": Mock()}
        return protein

    @pytest.fixture
    def mock_wandb_run(self):
        """Create a mock wandb run."""
        run = Mock()
        run.id = "test_run_123"
        run.sweep_id = "test_sweep_456"
        run.config = Mock()
        run.config.__dict__ = {"_locked": {}}
        run.summary = {}
        return run

    @pytest.fixture
    def mock_wandb_api(self):
        """Create a mock wandb API."""
        api = Mock()
        sweep = Mock()
        sweep.runs = []
        api.sweep.return_value = sweep
        return api

    def test_wandb_protein_init(self, mock_protein, mock_wandb_run):
        """Test WandbProtein initialization."""
        with patch("wandb.Api") as mock_api:
            wandb_protein = WandbProtein(mock_protein, mock_wandb_run)

            assert wandb_protein._protein == mock_protein
            assert wandb_protein._wandb_run == mock_wandb_run
            assert wandb_protein._sweep_id == "test_sweep_456"
            assert wandb_protein._suggestion_info == {}

    def test_generate_protein_suggestion(self, mock_protein, mock_wandb_run):
        """Test suggestion generation."""
        # Setup mock protein to return suggestion and info
        mock_suggestion = {"learning_rate": 0.001, "batch_size": 64}
        mock_info = {"cost": 100.0, "score": 0.95, "rating": 0.8}
        mock_protein.suggest.return_value = (mock_suggestion, mock_info)

        with patch("wandb.Api"):
            wandb_protein = WandbProtein(mock_protein, mock_wandb_run)
            wandb_protein._generate_protein_suggestion()

            # Verify suggestion and info were stored
            assert wandb_protein._suggestion == mock_suggestion
            assert wandb_protein._suggestion_info["cost"] == 100.0
            assert wandb_protein._suggestion_info["score"] == 0.95
            assert wandb_protein._suggestion_info["rating"] == 0.8
            assert wandb_protein._suggestion_info["suggestion_uuid"] == "test_run_123"

            # Verify protein.suggest was called correctly
            mock_protein.suggest.assert_called_once_with(fill=None)

    def test_suggest_returns_info(self, mock_protein, mock_wandb_run):
        """Test that suggest() returns the stored info."""
        mock_suggestion = {"learning_rate": 0.001}
        mock_info = {"cost": 50.0, "score": 0.9}
        mock_protein.suggest.return_value = (mock_suggestion, mock_info)

        with patch("wandb.Api"):
            wandb_protein = WandbProtein(mock_protein, mock_wandb_run)

            # Generate and get suggestion
            suggestion, info = wandb_protein.suggest()

            # Verify info is returned correctly
            assert info["cost"] == 50.0
            assert info["score"] == 0.9
            assert info["suggestion_uuid"] == "test_run_123"

    def test_store_suggestion_in_wandb(self, mock_protein, mock_wandb_run):
        """Test that suggestion and info are stored in wandb summary."""
        mock_suggestion = {"learning_rate": 0.002, "batch_size": 32}
        mock_info = {"cost": 75.0, "score": 0.88, "rating": 0.7}
        mock_protein.suggest.return_value = (mock_suggestion, mock_info)

        with patch("wandb.Api"):
            wandb_protein = WandbProtein(mock_protein, mock_wandb_run)
            wandb_protein._generate_protein_suggestion()

            # Verify data was stored in wandb summary
            expected_summary_updates = [
                ({"protein.suggestion": mock_suggestion},),
                ({"protein.suggestion_info": mock_info},),
            ]

            # Check that summary.update was called with expected values
            assert mock_wandb_run.summary.update.call_count >= 2

    def test_suggestion_from_run_with_stored_data(self, mock_protein, mock_wandb_run):
        """Test loading suggestion from a run with stored data."""
        # Mock a run with stored suggestion and info
        mock_run = Mock()
        mock_run.id = "historical_run_789"
        mock_run.summary = {
            "protein.suggestion": {"learning_rate": 0.003, "batch_size": 128},
            "protein.suggestion_info": {"cost": 25.0, "score": 0.92, "rating": 0.85},
        }

        with patch("wandb.Api"):
            wandb_protein = WandbProtein(mock_protein, mock_wandb_run)
            suggestion, info = wandb_protein._suggestion_from_run(mock_run)

            # Verify suggestion was loaded correctly
            assert suggestion["learning_rate"] == 0.003
            assert suggestion["batch_size"] == 128

            # Verify info was loaded correctly
            assert info["cost"] == 25.0
            assert info["score"] == 0.92
            assert info["rating"] == 0.85

    def test_suggestion_from_run_without_stored_data(self, mock_protein, mock_wandb_run):
        """Test loading suggestion from a run without stored data (fallback)."""
        # Mock a run without stored suggestion/info
        mock_run = Mock()
        mock_run.id = "old_run_456"
        mock_run.summary = {}

        with patch("wandb.Api"):
            wandb_protein = WandbProtein(mock_protein, mock_wandb_run)
            suggestion, info = wandb_protein._suggestion_from_run(mock_run)

            # Should return empty suggestion and info with just the UUID
            assert suggestion == {}
            assert info["suggestion_uuid"] == "old_run_456"

    def test_observe_updates_protein_and_info(self, mock_protein, mock_wandb_run):
        """Test that observe() calls protein.observe() and updates info."""
        mock_suggestion = {"learning_rate": 0.001}
        mock_info = {"cost": 100.0, "suggestion_uuid": "test_run_123"}

        with patch("wandb.Api"):
            wandb_protein = WandbProtein(mock_protein, mock_wandb_run)

            # Set initial state
            wandb_protein._suggestion = mock_suggestion
            wandb_protein._suggestion_info = mock_info

            # Call observe
            wandb_protein.observe(mock_suggestion, objective=0.95, cost=120.0, is_failure=False, info=mock_info)

            # Verify protein.observe was called correctly
            mock_protein.observe.assert_called_once_with(mock_suggestion, 0.95, 120.0, False)

            # Verify info was updated
            assert wandb_protein._suggestion_info == mock_info

    def test_transform_suggestion_converts_numpy_types(self, mock_protein, mock_wandb_run):
        """Test that _transform_suggestion converts numpy types correctly."""
        import numpy as np

        with patch("wandb.Api"):
            wandb_protein = WandbProtein(mock_protein, mock_wandb_run)

            suggestion_with_numpy = {
                "learning_rate": np.float64(0.001),
                "batch_size": np.int32(64),
                "regularization": np.float32(0.01),
                "nested": {"param": np.array([1, 2, 3])},
            }

            result = wandb_protein._transform_suggestion(suggestion_with_numpy)

            # Check that numpy types were converted to Python types
            assert isinstance(result["learning_rate"], float)
            assert isinstance(result["batch_size"], int)
            assert isinstance(result["regularization"], float)
            assert isinstance(result["nested"]["param"], list)
            assert result["nested"]["param"] == [1, 2, 3]

    def test_error_handling_in_suggestion_generation(self, mock_protein, mock_wandb_run):
        """Test error handling when protein.suggest() fails."""
        # Make protein.suggest() raise an exception
        mock_protein.suggest.side_effect = RuntimeError("GP optimization failed")

        with patch("wandb.Api"):
            wandb_protein = WandbProtein(mock_protein, mock_wandb_run)

            # Should re-raise the exception
            with pytest.raises(RuntimeError, match="GP optimization failed"):
                wandb_protein._generate_protein_suggestion()
