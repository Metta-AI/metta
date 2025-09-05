"""Tests for ProteinOptimizer class."""


import pytest

from metta.sweep.models import Observation
from metta.sweep.optimizer.protein import ProteinOptimizer
from metta.sweep.protein_config import ParameterConfig, ProteinConfig, ProteinSettings


@pytest.fixture
def base_protein_config():
    """Base ProteinConfig for ProteinOptimizer tests."""
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


class TestProteinOptimizer:
    """Test cases for ProteinOptimizer class."""

    def test_unsupported_method_error(self):
        """Test that ProteinOptimizer raises error for unsupported methods."""
        with pytest.raises(ValueError, match="Unsupported optimization method"):
            # This should fail since we only allow 'bayes' now
            config = ProteinConfig(
                metric="loss",
                goal="minimize",
                method="bayes",  # This is supported, but let's create an invalid one
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
            # Manually set an invalid method to test error handling
            config.method = "invalid_method"
            ProteinOptimizer(config)

    def test_suggest_method_single(self):
        """Test the suggest method of ProteinOptimizer with single suggestion."""
        config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(config)

        # Test suggest with no observations
        suggestions = optimizer.suggest(observations=[], n_suggestions=1)

        # Check suggestion format
        assert isinstance(suggestions, list)
        assert len(suggestions) == 1
        suggestion = suggestions[0]
        assert isinstance(suggestion, dict)
        assert "param1" in suggestion
        assert 0 <= suggestion["param1"] <= 1

    def test_suggest_method_multiple(self):
        """Test the suggest method of ProteinOptimizer with multiple suggestions."""
        config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(config)

        # Test suggest with multiple suggestions
        suggestions = optimizer.suggest(observations=[], n_suggestions=3)

        # Check suggestion format
        assert isinstance(suggestions, list)
        assert len(suggestions) == 3
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert "param1" in suggestion
            assert 0 <= suggestion["param1"] <= 1

    def test_suggest_with_observations(self):
        """Test the suggest method with previous observations."""
        config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(config)

        # Create some observations
        observations = [
            Observation(score=0.5, cost=100, suggestion={"param1": 0.3}),
            Observation(score=0.8, cost=100, suggestion={"param1": 0.7}),
        ]

        # Get suggestions based on observations
        suggestions = optimizer.suggest(observations=observations, n_suggestions=2)

        # Check that we got suggestions back
        assert isinstance(suggestions, list)
        assert len(suggestions) == 2
        for suggestion in suggestions:
            assert isinstance(suggestion, dict)
            assert "param1" in suggestion
            assert 0 <= suggestion["param1"] <= 1

    def test_suggest_stateless_behavior(self):
        """Test that ProteinOptimizer is stateless - each call creates fresh instance."""
        config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(config)

        # Create an observation
        observation = Observation(score=0.8, cost=100, suggestion={"param1": 0.7})

        # First call with the observation
        suggestions1 = optimizer.suggest(observations=[observation], n_suggestions=1)

        # Second call with no observations - should not remember the previous observation
        suggestions2 = optimizer.suggest(observations=[], n_suggestions=1)

        # Both should return valid suggestions
        assert len(suggestions1) == 1
        assert len(suggestions2) == 1
        assert isinstance(suggestions1[0], dict)
        assert isinstance(suggestions2[0], dict)

    def test_suggest_with_failure_observations(self):
        """Test suggest method with failure observations."""
        config = ProteinConfig(
            metric="test_metric",
            goal="maximize",
            method="bayes",
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

        optimizer = ProteinOptimizer(config)

        # Create observations with one failure (score should be ignored for failures)
        observations = [
            Observation(score=0.5, cost=100, suggestion={"param1": 0.3}),
            Observation(score=0.0, cost=50, suggestion={"param1": 0.1}),  # This would be marked as failure
        ]

        # Should still work fine
        suggestions = optimizer.suggest(observations=observations, n_suggestions=1)

        assert isinstance(suggestions, list)
        assert len(suggestions) == 1
        assert isinstance(suggestions[0], dict)
        assert "param1" in suggestions[0]
