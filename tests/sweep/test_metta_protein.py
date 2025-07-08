"""Unit tests for MettaProtein class."""

from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein


class TestMettaProtein:
    """Test MettaProtein class functionality."""

    @pytest.fixture
    def sweep_config(self):
        """Create a test sweep configuration."""
        return OmegaConf.create(
            {
                "parameters": {
                    "trainer": {
                        "optimizer": {"learning_rate": {"min": 0.0001, "max": 0.01}},
                        "batch_size": {"values": [16, 32, 64]},
                    }
                },
                "method": "bayes",
                "metric": "eval/mean_score",
                "goal": "maximize",
                "protein": {
                    "search_center": {"trainer/optimizer/learning_rate": 0.001, "trainer/batch_size": 32},
                    "search_radius": {"trainer/optimizer/learning_rate": 0.5, "trainer/batch_size": 0.5},
                    "kernel": "matern",
                    "gamma": 0.25,
                    "xi": 0.001,
                },
            }
        )

    @pytest.fixture
    def mock_wandb_run(self):
        """Create a mock wandb run."""
        run = Mock()
        run.sweep_id = "test_sweep"
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

    @patch("metta.sweep.protein_metta.Protein")
    @patch("wandb.Api")
    def test_metta_protein_initialization(self, mock_api, mock_protein_class, sweep_config, mock_wandb_run):
        """Test that MettaProtein properly initializes with configuration."""
        # Mock Protein instance
        mock_protein = Mock()
        mock_protein.suggest.return_value = (
            {"trainer/optimizer/learning_rate": 0.005, "trainer/batch_size": 32},
            {"cost": 100.0, "score": 0.95},
        )
        mock_protein_class.return_value = mock_protein

        # Mock API
        mock_api.return_value.runs.return_value = []

        # Create MettaProtein
        MettaProtein(sweep_config, mock_wandb_run)

        # Verify Protein was initialized with correct sweep_config
        mock_protein_class.assert_called_once()
        call_args = mock_protein_class.call_args

        # First argument should be the sweep config with parameters
        sweep_config_arg = call_args[0][0]
        assert "trainer" in sweep_config_arg
        assert sweep_config_arg["method"] == "bayes"
        assert sweep_config_arg["metric"] == "eval/mean_score"
        assert sweep_config_arg["goal"] == "maximize"

        # Verify protein config was passed as kwargs
        kwargs = call_args[1]
        assert kwargs["search_center"]["trainer/optimizer/learning_rate"] == 0.001
        assert kwargs["kernel"] == "matern"
        assert kwargs["gamma"] == 0.25

    @patch("metta.sweep.protein_metta.Protein")
    @patch("wandb.Api")
    def test_transform_suggestion_cleans_numpy(self, mock_api, mock_protein_class, sweep_config, mock_wandb_run):
        """Test that _transform_suggestion properly cleans numpy types."""
        import numpy as np

        # Mock Protein to return numpy types
        mock_protein = Mock()
        mock_protein.suggest.return_value = (
            {"trainer/optimizer/learning_rate": np.float32(0.005), "trainer/batch_size": np.int64(32)},
            {"cost": np.float64(100.0)},
        )
        mock_protein_class.return_value = mock_protein

        # Mock API
        mock_api.return_value.runs.return_value = []

        # Create MettaProtein
        metta_protein = MettaProtein(sweep_config, mock_wandb_run)

        # Get suggestion - MettaProtein now cleans numpy types before storing
        suggestion, info = metta_protein.suggest()

        # Verify numpy types were cleaned to Python native types
        assert not isinstance(suggestion["trainer/optimizer/learning_rate"], np.floating)
        assert not isinstance(suggestion["trainer/batch_size"], np.integer)
        assert isinstance(suggestion["trainer/optimizer/learning_rate"], float)
        assert isinstance(suggestion["trainer/batch_size"], int)

        # Verify values are correct (with some floating point tolerance)
        assert abs(suggestion["trainer/optimizer/learning_rate"] - 0.005) < 1e-6
        assert suggestion["trainer/batch_size"] == 32

    @patch("metta.sweep.protein_metta.Protein")
    @patch("wandb.Api")
    def test_config_with_list_config(self, mock_api, mock_protein_class, mock_wandb_run):
        """Test MettaProtein handles ListConfig properly."""
        # Create a ListConfig instead of DictConfig
        list_config = OmegaConf.create(
            [
                {
                    "parameters": {"trainer": {"learning_rate": {"min": 0.001, "max": 0.01}}},
                    "method": "bayes",
                    "metric": "score",
                    "goal": "maximize",
                    "protein": {"kernel": "rbf"},
                }
            ]
        )

        # Mock Protein
        mock_protein = Mock()
        mock_protein.suggest.return_value = ({"trainer/learning_rate": 0.005}, {})
        mock_protein_class.return_value = mock_protein

        # Mock API
        mock_api.return_value.runs.return_value = []

        # Should handle ListConfig by using first element
        with pytest.raises(AttributeError):
            # ListConfig doesn't have 'parameters' attribute directly
            MettaProtein(list_config, mock_wandb_run)

    @patch("metta.sweep.protein_metta.Protein")
    @patch("wandb.Api")
    def test_config_defaults(self, mock_api, mock_protein_class, mock_wandb_run):
        """Test that default values are used when not specified in config."""
        # Minimal config without method, metric, goal
        minimal_config = OmegaConf.create(
            {"parameters": {"trainer": {"learning_rate": {"min": 0.001, "max": 0.01}}}, "protein": {"kernel": "rbf"}}
        )

        # Mock Protein
        mock_protein = Mock()
        mock_protein.suggest.return_value = ({"trainer/learning_rate": 0.005}, {})
        mock_protein_class.return_value = mock_protein

        # Mock API
        mock_api.return_value.runs.return_value = []

        # Create MettaProtein
        MettaProtein(minimal_config, mock_wandb_run)

        # Verify defaults were used
        call_args = mock_protein_class.call_args[0][0]
        assert call_args["method"] == "bayes"
        assert call_args["metric"] == "eval/mean_score"
        assert call_args["goal"] == "maximize"

    @patch("metta.sweep.protein_metta.Protein")
    @patch("wandb.Api")
    def test_wandb_config_override(self, mock_api, mock_protein_class, sweep_config, mock_wandb_run):
        """Test that WandB config is properly overridden with Protein suggestions."""
        # Mock Protein
        mock_protein = Mock()
        mock_protein.suggest.return_value = (
            {"trainer/optimizer/learning_rate": 0.007, "trainer/batch_size": 64},
            {"cost": 50.0},
        )
        mock_protein_class.return_value = mock_protein

        # Mock API
        mock_api.return_value.runs.return_value = []

        # Create MettaProtein
        MettaProtein(sweep_config, mock_wandb_run)

        # Verify WandB config was updated
        mock_wandb_run.config.update.assert_called()
        update_call = mock_wandb_run.config.update.call_args

        # Should have updated with Protein's suggestions
        config_update = update_call[0][0]
        assert config_update["trainer/optimizer/learning_rate"] == 0.007
        assert config_update["trainer/batch_size"] == 64

        # Should allow value changes
        assert update_call[1]["allow_val_change"] is True
