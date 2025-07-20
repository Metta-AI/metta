"""Unit tests for protein fill parameter fix."""

from unittest.mock import Mock, patch

import pytest

from metta.sweep.protein_wandb import WandbProtein


class TestProteinFillParameter:
    """Test that the fill parameter is correctly handled in protein suggest."""

    @pytest.fixture
    def mock_protein(self):
        """Create a mock Protein instance."""
        protein = Mock()
        # Track what fill parameter was passed
        protein.suggest_calls = []

        def track_suggest(fill):
            protein.suggest_calls.append(fill)
            return ({"trainer/optimizer/learning_rate": 0.001}, {"cost": 100.0, "score": 0.95})

        protein.suggest.side_effect = track_suggest
        protein.observe = Mock()
        return protein

    @pytest.fixture
    def mock_wandb_run(self):
        """Create a mock wandb run."""
        run = Mock()
        run.sweep_id = "test_sweep"
        run.entity = "test_entity"
        run.project = "test_project"
        run.id = "test_run"
        run.name = "test"
        run.summary = Mock()
        run.summary.get.return_value = None
        run.summary.update = Mock()

        # Create config mock separately
        config = Mock()
        config._locked = {}
        config.update = Mock()
        run.config = config

        return run

    @patch("wandb.Api")
    def test_generate_protein_suggestion_passes_none_fill(self, mock_api, mock_protein, mock_wandb_run):
        """Test that _generate_protein_suggestion always passes None as fill parameter."""
        # Mock API to return no previous runs
        mock_api.return_value.runs.return_value = []

        # Create WandbProtein
        WandbProtein(mock_protein, mock_wandb_run)

        # Verify that suggest was called with fill=None
        assert len(mock_protein.suggest_calls) == 1
        assert mock_protein.suggest_calls[0] is None

    @patch("wandb.Api")
    def test_fill_none_even_with_previous_runs(self, mock_api, mock_protein, mock_wandb_run):
        """Test that fill=None is used even when loading previous runs with suggestion_info."""
        # Create a historical run with suggestion_info
        historical_run = Mock()
        historical_run.id = "historical"
        historical_run.name = "historical"
        historical_run.summary = {
            "protein.state": "success",
            "protein.objective": 0.85,
            "protein.cost": 120.0,
            "protein.suggestion": {"trainer": {"optimizer": {"learning_rate": 0.002}}},
            "protein.suggestion_info": {"cost": 120.0, "score": 0.85, "rating": 0.9},
        }

        # Mock API to return the historical run
        mock_api.return_value.runs.return_value = [historical_run]

        # Create WandbProtein
        wandb_protein = WandbProtein(mock_protein, mock_wandb_run)

        # Verify that suggest was still called with fill=None
        assert len(mock_protein.suggest_calls) == 1
        assert mock_protein.suggest_calls[0] is None

        # Verify that _suggestion_info was set from current run's protein.suggest()
        assert hasattr(wandb_protein, "_suggestion_info")
        # The info comes from the mock protein's suggest return value
        assert wandb_protein._suggestion_info == {"cost": 100.0, "score": 0.95}

    @patch("wandb.Api")
    def test_no_type_error_with_cleaned_data(self, mock_api, mock_protein, mock_wandb_run):
        """Test that there's no TypeError even if suggestion_info contains cleaned/string data."""
        # Create a run where suggestion_info might have been deep_cleaned to strings
        historical_run = Mock()
        historical_run.id = "historical"
        historical_run.name = "historical"
        historical_run.summary = {
            "protein.state": "success",
            "protein.objective": 0.85,
            "protein.cost": 120.0,
            "protein.suggestion": {"trainer": {"optimizer": {"learning_rate": 0.002}}},
            "protein.suggestion_info": {
                "nested": "cleaned_to_string",  # Simulates deep_clean converting to string
                "cost": 120.0,
            },
        }

        # Mock API
        mock_api.return_value.runs.return_value = [historical_run]

        # This should not raise TypeError anymore
        WandbProtein(mock_protein, mock_wandb_run)

        # Verify no error and suggest was called with None
        assert len(mock_protein.suggest_calls) == 1
        assert mock_protein.suggest_calls[0] is None
