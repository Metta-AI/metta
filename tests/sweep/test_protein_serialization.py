"""Unit tests for protein WandB serialization and cleaning functionality."""

import json
from unittest.mock import Mock, patch

import numpy as np
import pytest

from metta.sweep.protein_wandb import WandbProtein


class MockSummarySubDict(dict):
    """Mock WandB SummarySubDict for testing."""

    def _as_dict(self):
        return dict(self)


class TestWandbProteinSerialization:
    """Test serialization and cleaning of WandB objects in protein sweep."""

    @pytest.fixture
    def mock_protein(self):
        """Create a mock Protein instance."""
        protein = Mock()
        protein.suggest.return_value = (
            {"trainer": {"optimizer": {"learning_rate": 0.001}}},
            {"cost": 100.0, "score": 0.95},
        )
        return protein

    @pytest.fixture
    def mock_wandb_run(self):
        """Create a mock wandb run with summary."""
        run = Mock()
        run.sweep_id = "test_sweep_123"
        run.entity = "test_entity"
        run.project = "test_project"
        run.id = "test_run_id"
        run.name = "test_run"
        run.summary = Mock()
        run.summary.get.return_value = None
        run.summary.update = Mock()

        # Create config mock separately to avoid attribute issues
        config = Mock()
        config._locked = {}
        config.update = Mock()
        run.config = config

        return run

    def test_deep_clean_numpy_types(self, mock_protein, mock_wandb_run):
        """Test that _deep_clean properly handles numpy types."""
        with patch("wandb.Api"):
            protein_wandb = WandbProtein(mock_protein, mock_wandb_run)

            # Test various numpy types
            test_data = {
                "float32": np.float32(3.14),
                "float64": np.float64(2.718),
                "int32": np.int32(42),
                "int64": np.int64(100),
                "array": np.array([1, 2, 3]),
                "nested": {"learning_rate": np.float32(0.001), "batch_size": np.int32(32)},
            }

            cleaned = protein_wandb._deep_clean(test_data)

            # Verify all values are JSON serializable
            json_str = json.dumps(cleaned)
            assert json_str is not None

            # Verify types are converted correctly
            assert isinstance(cleaned["float32"], float)
            assert isinstance(cleaned["float64"], float)
            assert isinstance(cleaned["int32"], int)
            assert isinstance(cleaned["int64"], int)
            assert isinstance(cleaned["array"], list)
            assert isinstance(cleaned["nested"]["learning_rate"], float)
            assert isinstance(cleaned["nested"]["batch_size"], int)

    def test_deep_clean_wandb_summary_subdict(self, mock_protein, mock_wandb_run):
        """Test that _deep_clean properly handles WandB SummarySubDict objects."""
        with patch("wandb.Api"):
            protein_wandb = WandbProtein(mock_protein, mock_wandb_run)

            # Create a mock SummarySubDict
            summary_dict = MockSummarySubDict(
                {"trainer": MockSummarySubDict({"learning_rate": 0.001, "batch_size": 32}), "score": 0.95}
            )

            cleaned = protein_wandb._deep_clean(summary_dict)

            # Verify it's converted to regular dict and JSON serializable
            json_str = json.dumps(cleaned)
            assert json_str is not None
            assert isinstance(cleaned, dict)
            assert isinstance(cleaned["trainer"], dict)
            assert cleaned["trainer"]["learning_rate"] == 0.001

    def test_deep_clean_mixed_types(self, mock_protein, mock_wandb_run):
        """Test _deep_clean with mixed numpy and WandB types."""
        with patch("wandb.Api"):
            protein_wandb = WandbProtein(mock_protein, mock_wandb_run)

            # Complex nested structure
            test_data = {
                "config": MockSummarySubDict(
                    {"learning_rate": np.float32(0.001), "layers": [np.int32(64), np.int32(128), np.int32(64)]}
                ),
                "metrics": {"scores": np.array([0.1, 0.2, 0.3]), "final_score": np.float64(0.95)},
                "info": MockSummarySubDict({"run_id": "test123"}),
            }

            cleaned = protein_wandb._deep_clean(test_data)

            # Verify everything is JSON serializable
            json_str = json.dumps(cleaned)
            assert json_str is not None

            # Verify structure is preserved
            assert cleaned["config"]["learning_rate"] == pytest.approx(0.001)
            assert cleaned["config"]["layers"] == [64, 128, 64]
            assert cleaned["metrics"]["scores"] == [0.1, 0.2, 0.3]
            assert cleaned["info"]["run_id"] == "test123"

    def test_suggestion_info_not_cleaned_as_fill(self, mock_protein, mock_wandb_run):
        """Test that suggestion_info is cleaned and stored properly."""
        with patch("wandb.Api"):
            # Set up protein to return a suggestion with numpy types
            mock_protein.suggest.return_value = (
                {"trainer": {"optimizer": {"learning_rate": np.float32(0.001)}}},
                {"cost": np.float64(100.0), "score": np.float32(0.95)},
            )

            # Create protein
            protein_wandb = WandbProtein(mock_protein, mock_wandb_run)

            # Verify suggest was called with fill=None
            assert mock_protein.suggest.called
            mock_protein.suggest.assert_called_with(fill=None)

            # Verify suggestion and info were cleaned and stored
            assert protein_wandb._suggestion["trainer"]["optimizer"]["learning_rate"] == pytest.approx(0.001, rel=1e-5)
            assert protein_wandb._suggestion_info["cost"] == 100.0
            assert protein_wandb._suggestion_info["score"] == pytest.approx(0.95)

            # Verify they were saved to wandb summary
            calls = mock_wandb_run.summary.update.call_args_list
            suggestion_saved = False
            for call in calls:
                if "protein.suggestion" in call[0][0]:
                    suggestion_saved = True
                    # Verify the saved suggestion is cleaned
                    saved_suggestion = call[0][0]["protein.suggestion"]
                    assert isinstance(saved_suggestion["trainer"]["optimizer"]["learning_rate"], float)
                    break
            assert suggestion_saved

    def test_suggestion_from_run_cleans_wandb_objects(self, mock_protein, mock_wandb_run):
        """Test that _suggestion_from_run properly extracts and cleans data."""
        with patch("wandb.Api"):
            protein_wandb = WandbProtein(mock_protein, mock_wandb_run)

            # Mock a run with WandB objects in summary
            mock_run = Mock()
            mock_run.id = "historical_run_123"
            mock_run.summary = {
                "protein.suggestion": MockSummarySubDict({"trainer": {"learning_rate": np.float32(0.005)}}),
                "protein.suggestion_info": MockSummarySubDict({"cost": np.float64(150.0), "score": 0.98}),
            }

            suggestion, info = protein_wandb._suggestion_from_run(mock_run)

            # Verify data is extracted correctly
            assert suggestion["trainer"]["learning_rate"] == pytest.approx(0.005)
            assert info["cost"] == pytest.approx(150.0)
            assert info["suggestion_uuid"] == "historical_run_123"

    @patch("wandb.Api")
    def test_full_initialization_with_serialization(self, mock_api, mock_protein, mock_wandb_run):
        """Test full initialization flow with proper serialization."""
        # Mock API to return no previous runs
        mock_api.return_value.runs.return_value = []

        # Create WandbProtein
        WandbProtein(mock_protein, mock_wandb_run)

        # Verify summary was updated with cleaned data
        calls = mock_wandb_run.summary.update.call_args_list

        # Should have 3 calls: initializing, suggestion, and running
        assert len(calls) >= 3

        # Check that protein.suggestion was saved
        suggestion_call = None
        for call in calls:
            if "protein.suggestion" in call[0][0]:
                suggestion_call = call[0][0]
                break

        assert suggestion_call is not None
        # Verify the suggestion is JSON serializable
        json.dumps(suggestion_call["protein.suggestion"])
        json.dumps(suggestion_call["protein.suggestion_info"])
