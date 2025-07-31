"""Tests for MettaProtein class."""

from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein


@pytest.fixture
def base_config():
    """Base configuration for MettaProtein tests."""
    return {
        "protein": {
            "max_suggestion_cost": 3600,
            "resample_frequency": 0,
            "num_random_samples": 50,
            "global_search_scale": 1,
            "random_suggestions": 1024,
            "suggestions_per_pareto": 256,
        },
        "metric": "reward",
        "goal": "maximize",
        "method": "bayes",
        "parameters": {},
    }


class TestMettaProtein:
    """Test cases for MettaProtein class."""

    @patch("metta.sweep.protein_metta.Protein")
    def test_metta_protein_init_with_full_config(self, mock_protein, base_config):
        """Test MettaProtein initialization with complete config."""
        config = OmegaConf.create(
            {
                **base_config,
                "protein": {
                    "max_suggestion_cost": 7200,
                    "resample_frequency": 0,
                    "num_random_samples": 100,
                    "global_search_scale": 2,
                    "random_suggestions": 1024,
                    "suggestions_per_pareto": 256,
                },
                "parameters": {
                    "trainer": {
                        "optimizer": {
                            "learning_rate": {
                                "distribution": "log_normal",
                                "min": 1e-5,
                                "max": 1e-2,
                                "scale": "auto",
                                "mean": 3e-4,
                            }
                        }
                    },
                },
            }
        )

        mock_protein_instance = Mock()
        mock_protein.return_value = mock_protein_instance

        _ = MettaProtein(config)

        # Verify Protein was called with correct parameters
        mock_protein.assert_called_once()
        args, kwargs = mock_protein.call_args

        # Check that parameters were passed correctly
        protein_config = args[0]
        assert protein_config["metric"] == "reward"
        assert protein_config["goal"] == "maximize"
        assert protein_config["method"] == "bayes"
        assert "trainer" in protein_config

        # Check protein-specific parameters were passed as kwargs
        assert kwargs["max_suggestion_cost"] == 7200
        assert kwargs["num_random_samples"] == 100
        assert kwargs["global_search_scale"] == 2

    @patch("metta.sweep.protein_metta.Protein")
    def test_metta_protein_init_with_defaults(self, mock_protein, base_config):
        """Test MettaProtein initialization with minimal config."""
        config = OmegaConf.create(
            {
                **base_config,
                "metric": "accuracy",
                "goal": "minimize",
                "parameters": {
                    "batch_size": {"distribution": "uniform", "min": 16, "max": 128, "scale": "auto", "mean": 64},
                },
            }
        )

        mock_protein_instance = Mock()
        mock_protein.return_value = mock_protein_instance

        _ = MettaProtein(config)

        # Verify Protein was called with defaults
        mock_protein.assert_called_once()
        args, kwargs = mock_protein.call_args

        # Check that parameters were passed correctly
        protein_config = args[0]
        assert protein_config["metric"] == "accuracy"
        assert protein_config["goal"] == "minimize"
        assert protein_config["method"] == "bayes"

        # Check defaults were used
        assert kwargs["max_suggestion_cost"] == 3600
        assert kwargs["num_random_samples"] == 50
        assert kwargs["global_search_scale"] == 1

    @patch("metta.sweep.protein_metta.Protein")
    def test_suggest_method(self, mock_protein, base_config):
        """Test the suggest method returns cleaned numpy types."""
        config = OmegaConf.create(
            {
                **base_config,
                "parameters": {
                    "learning_rate": {
                        "distribution": "log_normal",
                        "min": 1e-5,
                        "max": 1e-2,
                        "scale": "auto",
                        "mean": 3e-4,
                    }
                },
            }
        )

        mock_protein_instance = Mock()

        # Mock numpy types in protein response
        import numpy as np

        suggestion_with_numpy = {
            "learning_rate": np.float64(0.001),
            "batch_size": np.int32(64),
        }
        info = {"some": "info"}

        mock_protein_instance.suggest.return_value = (suggestion_with_numpy, info)
        mock_protein.return_value = mock_protein_instance

        metta_protein = MettaProtein(config)
        result_suggestion, result_info = metta_protein.suggest()

        # Check that numpy types were converted to native Python types
        assert isinstance(result_suggestion["learning_rate"], float)
        assert isinstance(result_suggestion["batch_size"], int)
        assert result_info == info

    @patch("metta.sweep.protein_metta.Protein")
    def test_observe_method(self, mock_protein, base_config):
        """Test the observe method passes through to protein."""
        config = OmegaConf.create(base_config)

        mock_protein_instance = Mock()
        mock_protein.return_value = mock_protein_instance

        metta_protein = MettaProtein(config)

        suggestion = {"learning_rate": 0.001}
        objective = 0.95
        cost = 120.0
        is_failure = False

        metta_protein.observe(suggestion, objective, cost, is_failure)

        mock_protein_instance.observe.assert_called_once_with(suggestion, objective, cost, is_failure)

    @patch("metta.sweep.protein_metta.Protein")
    def test_observe_failure_method(self, mock_protein, base_config):
        """Test the observe_failure method calls observe with failure parameters."""
        config = OmegaConf.create(base_config)

        mock_protein_instance = Mock()
        mock_protein.return_value = mock_protein_instance

        metta_protein = MettaProtein(config)

        suggestion = {"learning_rate": 0.001}
        metta_protein.observe_failure(suggestion)

        mock_protein_instance.observe.assert_called_once_with(suggestion, 0, 0.01, True)

    @patch("metta.sweep.protein_metta.Protein")
    def test_num_observations_property(self, mock_protein, base_config):
        """Test the num_observations property."""
        config = OmegaConf.create(base_config)

        mock_protein_instance = Mock()
        mock_protein_instance.success_observations = [1, 2, 3]
        mock_protein_instance.failure_observations = [4, 5]
        mock_protein.return_value = mock_protein_instance

        metta_protein = MettaProtein(config)

        assert metta_protein.num_observations == 5
