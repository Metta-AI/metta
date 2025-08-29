"""Tests for MettaProtein class."""

from unittest.mock import Mock, patch

import pytest

from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings
from metta.sweep.protein_metta import MettaProtein


@pytest.fixture
def base_protein_config():
    """Base ProteinConfig for MettaProtein tests."""
    return ProteinConfig(
        metric="reward",
        goal="maximize",
        method="bayes",
        parameters={},
        settings=ProteinSettings(
            max_suggestion_cost=3600,
            resample_frequency=0,
            num_random_samples=50,
            global_search_scale=1,
            random_suggestions=1024,
            suggestions_per_pareto=256,
        ),
    )


class TestMettaProtein:
    """Test cases for MettaProtein class."""

    @patch("metta.sweep.protein_metta.Protein")
    def test_metta_protein_init_with_full_config(self, mock_protein, base_protein_config):
        """Test MettaProtein initialization with complete config."""
        config = ProteinConfig(
            metric="reward",
            goal="maximize",
            method="bayes",
            parameters={
                "trainer.optimizer.learning_rate": ParameterConfig(
                    distribution="log_normal",
                    min=1e-5,
                    max=1e-2,
                    scale="auto",
                    mean=3e-4,
                )
            },
            settings=ProteinSettings(
                max_suggestion_cost=7200,
                resample_frequency=0,
                num_random_samples=100,
                global_search_scale=2,
                random_suggestions=1024,
                suggestions_per_pareto=256,
            ),
        )

        mock_protein_instance = Mock()
        mock_protein.return_value = mock_protein_instance

        _ = MettaProtein(config)

        # Verify Protein was called with correct parameters
        mock_protein.assert_called_once()
        args, kwargs = mock_protein.call_args

        # Check that parameters were passed correctly
        protein_dict = args[0]
        assert protein_dict["metric"] == "reward"
        assert protein_dict["goal"] == "maximize"
        assert protein_dict["method"] == "bayes"
        assert "trainer.optimizer.learning_rate" in protein_dict

        # Check protein-specific parameters were passed as kwargs
        assert kwargs["max_suggestion_cost"] == 7200
        assert kwargs["num_random_samples"] == 100
        assert kwargs["global_search_scale"] == 2

    @patch("metta.sweep.protein_metta.Protein")
    def test_metta_protein_init_with_defaults(self, mock_protein):
        """Test MettaProtein initialization with minimal config."""
        config = ProteinConfig(
            metric="accuracy",
            goal="minimize",
            method="bayes",
            parameters={
                "batch_size": ParameterConfig(distribution="uniform", min=16, max=128, scale="auto", mean=64),
            },
        )

        mock_protein_instance = Mock()
        mock_protein.return_value = mock_protein_instance

        _ = MettaProtein(config)

        # Verify Protein was called
        mock_protein.assert_called_once()
        args, kwargs = mock_protein.call_args

        # Check basic parameters
        protein_dict = args[0]
        assert protein_dict["metric"] == "accuracy"
        assert protein_dict["goal"] == "minimize"

        # Check default settings were used
        assert kwargs["max_suggestion_cost"] == 10800  # Default value

    @patch("metta.sweep.protein_metta.Random")
    def test_metta_protein_random_method(self, mock_random):
        """Test MettaProtein initialization with random method."""
        config = ProteinConfig(
            metric="loss",
            goal="minimize",
            method="random",
            parameters={
                "learning_rate": ParameterConfig(
                    distribution="log_normal",
                    min=1e-4,
                    max=1e-1,
                    scale="auto",
                    mean=1e-2,
                )
            },
        )

        mock_random_instance = Mock()
        mock_random.return_value = mock_random_instance

        _ = MettaProtein(config)

        # Verify Random was called instead of Protein
        mock_random.assert_called_once()
        args, _ = mock_random.call_args
        protein_dict = args[0]
        assert protein_dict["method"] == "random"

    @patch("metta.sweep.protein_metta.ParetoGenetic")
    def test_metta_protein_genetic_method(self, mock_genetic):
        """Test MettaProtein initialization with genetic method."""
        config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="genetic",
            parameters={
                "param1": ParameterConfig(
                    distribution="uniform",
                    min=0,
                    max=1,
                    scale="auto",
                    mean=0.5,
                )
            },
            settings=ProteinSettings(
                bias_cost=False,
                log_bias=True,
            ),
        )

        mock_genetic_instance = Mock()
        mock_genetic.return_value = mock_genetic_instance

        _ = MettaProtein(config)

        # Verify ParetoGenetic was called
        mock_genetic.assert_called_once()
        _, kwargs = mock_genetic.call_args
        assert kwargs["bias_cost"] is False
        assert kwargs["log_bias"] is True

    def test_suggest_method(self):
        """Test the suggest method of MettaProtein."""
        config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="random",  # Use random for predictability in tests
            parameters={
                "param1": ParameterConfig(
                    distribution="uniform",
                    min=0,
                    max=1,
                    scale="auto",
                    mean=0.5,
                )
            },
        )

        protein = MettaProtein(config)

        # Test suggest
        suggestion, info = protein.suggest()

        # Check suggestion format
        assert isinstance(suggestion, dict)
        assert "param1" in suggestion
        assert 0 <= suggestion["param1"] <= 1

        # Check info format
        assert isinstance(info, dict)

    def test_observe_method(self):
        """Test the observe method of MettaProtein."""
        config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="random",
            parameters={
                "param1": ParameterConfig(
                    distribution="uniform",
                    min=0,
                    max=1,
                    scale="auto",
                    mean=0.5,
                )
            },
        )

        protein = MettaProtein(config)

        # Get a suggestion
        suggestion, _ = protein.suggest()

        # Observe the result
        protein.observe(suggestion, objective=0.8, cost=100.0, is_failure=False)

        # Check that observation was recorded
        assert protein.num_observations == 1

    def test_observe_failure_method(self):
        """Test observing a failure in MettaProtein."""
        config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="random",
            parameters={
                "param1": ParameterConfig(
                    distribution="uniform",
                    min=0,
                    max=1,
                    scale="auto",
                    mean=0.5,
                )
            },
        )

        protein = MettaProtein(config)

        # Get a suggestion
        suggestion, _ = protein.suggest()

        # Observe a failure
        protein.observe(suggestion, objective=None, cost=50.0, is_failure=True)

        # The observation should still be recorded
        assert protein.num_observations == 1

    def test_num_observations_property(self):
        """Test the num_observations property."""
        config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="random",
            parameters={
                "param1": ParameterConfig(
                    distribution="uniform",
                    min=0,
                    max=1,
                    scale="auto",
                    mean=0.5,
                )
            },
        )

        protein = MettaProtein(config)

        # Initially should have 0 observations
        assert protein.num_observations == 0

        # Add some observations
        for i in range(3):
            suggestion, _ = protein.suggest()
            protein.observe(suggestion, objective=0.5 + i * 0.1, cost=100.0, is_failure=False)

        # Should now have 3 observations
        assert protein.num_observations == 3
