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
                "method": "bayes",
                "metric": "eval/mean_score",
                "goal": "maximize",
                "parameters": {
                    "trainer": {
                        "optimizer": {"learning_rate": {"min": 0.0001, "max": 0.01}},
                        "batch_size": {"values": [16, 32, 64]},
                    }
                },
                "protein": {
                    "search_center": {"trainer/optimizer/learning_rate": 0.001, "trainer/batch_size": 32},
                    "search_radius": {"trainer/optimizer/learning_rate": 0.5, "trainer/batch_size": 0.5},
                    "kernel": "matern",
                    "gamma": 0.25,
                    "xi": 0.001,
                },
            }
        )

    @patch("metta.sweep.protein_metta.Protein")
    def test_metta_protein_initialization(self, mock_protein_class, sweep_config):
        """Test that MettaProtein properly initializes with configuration."""
        # Mock Protein instance
        mock_protein = Mock()
        mock_protein.suggest.return_value = (
            {"trainer/optimizer/learning_rate": 0.005, "trainer/batch_size": 32},
            {"cost": 100.0, "score": 0.95},
        )
        mock_protein_class.return_value = mock_protein

        # Create MettaProtein
        MettaProtein(sweep_config)

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
    def test_transform_suggestion_cleans_numpy(self, mock_protein_class, sweep_config):
        """Test that suggestions have numpy types cleaned."""
        # Mock Protein with numpy types in response
        mock_protein = Mock()
        import numpy as np

        suggestion_with_numpy = {
            "learning_rate": np.float64(0.005),
            "batch_size": np.int32(32),
        }
        mock_protein.suggest.return_value = (suggestion_with_numpy, {})
        mock_protein_class.return_value = mock_protein

        # Create MettaProtein
        metta_protein = MettaProtein(sweep_config)

        # Get suggestion - MettaProtein now cleans numpy types before storing
        suggestion, info = metta_protein.suggest()

        # Verify types were cleaned
        assert isinstance(suggestion["learning_rate"], float)
        assert isinstance(suggestion["batch_size"], int)
        assert suggestion["learning_rate"] == 0.005
        assert suggestion["batch_size"] == 32

    @patch("metta.sweep.protein_metta.Protein")
    def test_config_with_nested_structure(self, mock_protein_class):
        """Test MettaProtein handles nested config structures properly."""
        # Create a config with deeply nested protein parameters
        nested_config = OmegaConf.create(
            {
                "method": "bayes",
                "metric": "accuracy",
                "goal": "maximize",
                "parameters": {
                    "trainer": {
                        "optimizer": {
                            "learning_rate": {"min": 0.001, "max": 0.01},
                            "momentum": {"min": 0.8, "max": 0.99},
                        }
                    }
                },
                "protein": {
                    "kernel": "rbf",
                    "search_center": {"trainer/optimizer/learning_rate": 0.005, "trainer/optimizer/momentum": 0.9},
                },
            }
        )

        # Mock Protein
        mock_protein = Mock()
        mock_protein.suggest.return_value = ({"trainer/optimizer/learning_rate": 0.007}, {})
        mock_protein_class.return_value = mock_protein

        # Should handle nested config properly
        MettaProtein(nested_config)

        # Verify protein was initialized with the nested config
        mock_protein_class.assert_called_once()
        call_args = mock_protein_class.call_args
        assert call_args[0][0]["method"] == "bayes"

    @patch("metta.sweep.protein_metta.Protein")
    def test_config_defaults(self, mock_protein_class):
        """Test MettaProtein with minimal configuration."""
        minimal_config = OmegaConf.create(
            {
                "method": "bayes",
                "metric": "accuracy",
                "goal": "maximize",
                "parameters": {"learning_rate": {"min": 0.001, "max": 0.01}},
                "protein": {"kernel": "rbf"},
            }
        )

        # Mock Protein
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein

        # Create MettaProtein
        MettaProtein(minimal_config)

        # Verify defaults were used
        mock_protein_class.assert_called_once()
        args, kwargs = mock_protein_class.call_args

        protein_config = args[0]
        assert protein_config["metric"] == "accuracy"
        assert protein_config["goal"] == "maximize"
        assert protein_config["method"] == "bayes"

    @patch("metta.sweep.protein_metta.Protein")
    def test_wandb_config_override(self, mock_protein_class, sweep_config):
        """Test that protein configuration is properly set up."""
        # Mock Protein
        mock_protein = Mock()
        mock_protein_class.return_value = mock_protein

        # Create MettaProtein
        MettaProtein(sweep_config)

        # Verify Protein was initialized (this replaces the wandb config override test)
        mock_protein_class.assert_called_once()
        call_args = mock_protein_class.call_args

        # Verify the config structure was passed correctly
        protein_config = call_args[0][0]
        assert "trainer" in protein_config
        assert protein_config["metric"] == "eval/mean_score"
