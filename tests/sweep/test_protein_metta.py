"""Tests for ProteinOptimizer class."""

import copy

import pytest

from metta.sweep.optimizer.constraints import ConstraintSpec, DivisionConstraint
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

    def test_unsupported_method_validation(self):
        """Test that ProteinConfig only accepts 'bayes' as method."""
        # Try to create a config with an unsupported method
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProteinConfig(
                metric="loss",
                goal="minimize",
                method="grid",  # Grid search is not supported
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

    def test_optimizer_initialization(self):
        """Test that ProteinOptimizer initializes correctly with bayes method."""
        config = ProteinConfig(
            metric="accuracy",
            goal="maximize",
            method="bayes",  # Bayesian optimization is supported
            parameters={
                "learning_rate": ParameterConfig(
                    distribution="log_normal",
                    min=1e-4,
                    max=1e-1,
                    scale="auto",
                    mean=1e-2,
                ),
                "batch_size": ParameterConfig(
                    distribution="int_uniform",
                    min=16,
                    max=128,
                    mean=64,
                    scale="auto",
                ),
            },
        )

        # Should not raise error
        optimizer = ProteinOptimizer(config)
        assert optimizer.config == config

    def test_suggest_with_no_observations(self, base_protein_config):
        """Test that suggest method works with no observations."""
        base_protein_config.parameters = {
            "lr": ParameterConfig(
                distribution="log_normal",
                min=1e-4,
                max=1e-1,
                scale="auto",
                mean=1e-2,
            )
        }

        optimizer = ProteinOptimizer(base_protein_config)

        # Request suggestions with no observations
        suggestions = optimizer.suggest([], n_suggestions=3)

        assert len(suggestions) == 3
        assert all("lr" in s for s in suggestions)
        assert all(1e-4 <= s["lr"] <= 1e-1 for s in suggestions)

    def test_suggest_with_observations(self):
        """Test that suggest method works with observations."""
        config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="bayes",
            parameters={
                "lr": ParameterConfig(
                    distribution="log_normal",
                    min=1e-4,
                    max=1e-1,
                    scale="auto",
                    mean=1e-2,
                )
            },
        )

        optimizer = ProteinOptimizer(config)

        # Create observations as dictionaries (not Observation objects)
        observations = [
            {"score": 0.5, "cost": 100, "suggestion": {"lr": 0.001}},
            {"score": 0.7, "cost": 120, "suggestion": {"lr": 0.01}},
            {"score": 0.6, "cost": 110, "suggestion": {"lr": 0.005}},
        ]

        # Request suggestions with observations
        suggestions = optimizer.suggest(observations, n_suggestions=2)

        assert len(suggestions) == 2
        assert all("lr" in s for s in suggestions)
        assert all(1e-4 <= s["lr"] <= 1e-1 for s in suggestions)

    def test_suggest_respects_parameter_bounds(self):
        """Test that suggestions respect parameter bounds."""
        config = ProteinConfig(
            metric="score",
            goal="minimize",
            method="bayes",
            parameters={
                "lr": ParameterConfig(
                    distribution="uniform",
                    min=0.1,
                    max=0.5,
                    mean=0.3,
                    scale="auto",
                ),
                "momentum": ParameterConfig(
                    distribution="uniform",
                    min=0.8,
                    max=0.99,
                    mean=0.9,
                    scale="auto",
                ),
            },
        )

        optimizer = ProteinOptimizer(config)

        # Request multiple suggestions
        suggestions = optimizer.suggest([], n_suggestions=10)

        assert len(suggestions) == 10
        for suggestion in suggestions:
            assert 0.1 <= suggestion["lr"] <= 0.5
            assert 0.8 <= suggestion["momentum"] <= 0.99

    def test_suggest_handles_integer_parameters(self):
        """Test that optimizer handles integer parameters."""
        config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="bayes",
            parameters={
                "num_layers": ParameterConfig(
                    distribution="int_uniform",
                    min=1,
                    max=10,
                    mean=5,
                    scale="auto",
                ),
                "hidden_size": ParameterConfig(
                    distribution="uniform_pow2",  # Powers of 2
                    min=32,
                    max=512,
                    mean=128,
                    scale="auto",
                ),
            },
        )

        optimizer = ProteinOptimizer(config)

        suggestions = optimizer.suggest([], n_suggestions=5)

        assert len(suggestions) == 5
        for suggestion in suggestions:
            assert 1 <= suggestion["num_layers"] <= 10
            assert 32 <= suggestion["hidden_size"] <= 512

    def test_empty_suggestions_request(self):
        """Test requesting zero suggestions."""
        config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="bayes",
            parameters={
                "lr": ParameterConfig(
                    distribution="uniform",
                    min=0.1,
                    max=0.5,
                    mean=0.3,
                    scale="auto",
                ),
            },
        )

        optimizer = ProteinOptimizer(config)
        suggestions = optimizer.suggest([], n_suggestions=0)

        assert suggestions == []

    def test_observations_with_missing_fields(self):
        """Test that optimizer handles observations with missing fields gracefully."""
        config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="bayes",
            parameters={
                "lr": ParameterConfig(
                    distribution="log_normal",
                    min=1e-4,
                    max=1e-1,
                    scale="auto",
                    mean=1e-2,
                )
            },
        )

        optimizer = ProteinOptimizer(config)

        # Observations with missing cost field (should default to 0)
        observations = [
            {"score": 0.5, "suggestion": {"lr": 0.001}},
            {"score": 0.7, "cost": None, "suggestion": {"lr": 0.01}},
        ]

        # Should handle gracefully
        suggestions = optimizer.suggest(observations, n_suggestions=1)
        assert len(suggestions) == 1

    def test_division_constraint_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Division constraint clamps divisor to a valid factor."""

        class DummyProtein:
            def __init__(self, *_, **__):
                pass

            def observe(self, hypers, score, cost, is_failure):
                pass

            def suggest(self, n_suggestions=1):
                base = {"trainer": {"batch_size": 64, "minibatch_size": 20}}
                info = {}
                if n_suggestions == 1:
                    return copy.deepcopy(base), info
                return [(copy.deepcopy(base), info) for _ in range(n_suggestions)]

        monkeypatch.setattr("metta.sweep.optimizer.protein.Protein", DummyProtein)

        config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="bayes",
            parameters={
                "trainer": {
                    "batch_size": ParameterConfig(distribution="int_uniform", min=32, max=128, mean=64, scale="auto"),
                    "minibatch_size": ParameterConfig(
                        distribution="int_uniform", min=8, max=64, mean=32, scale="auto"
                    ),
                }
            },
            constraints=[ConstraintSpec(divisor_key="trainer.minibatch_size", dividend_key="trainer.batch_size")],
        )

        optimizer = ProteinOptimizer(config)
        suggestion = optimizer.suggest([], n_suggestions=1)[0]

        assert suggestion["trainer"]["batch_size"] == 64
        assert suggestion["trainer"]["minibatch_size"] == 16  # Largest divisor <= 20

    def test_constraints_apply_to_multiple_suggestions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Constraints are enforced for batched suggestion requests."""

        class DummyProtein:
            def __init__(self, *_, **__):
                pass

            def observe(self, hypers, score, cost, is_failure):
                pass

            def suggest(self, n_suggestions=1):
                base = {"trainer": {"batch_size": 90, "minibatch_size": 17}}
                info = {}
                if n_suggestions == 1:
                    return copy.deepcopy(base), info
                return [(copy.deepcopy(base), info) for _ in range(n_suggestions)]

        monkeypatch.setattr("metta.sweep.optimizer.protein.Protein", DummyProtein)

        config = ProteinConfig(
            metric="score",
            goal="maximize",
            method="bayes",
            parameters={
                "trainer": {
                    "batch_size": ParameterConfig(distribution="int_uniform", min=32, max=128, mean=64, scale="auto"),
                    "minibatch_size": ParameterConfig(
                        distribution="int_uniform", min=8, max=64, mean=32, scale="auto"
                    ),
                }
            },
            constraints=[ConstraintSpec(divisor_key="trainer.minibatch_size", dividend_key="trainer.batch_size")],
        )

        optimizer = ProteinOptimizer(config)
        suggestions = optimizer.suggest([], n_suggestions=3)

        assert len(suggestions) == 3
        assert all(s["trainer"]["batch_size"] == 90 for s in suggestions)
        assert all(s["trainer"]["minibatch_size"] == 15 for s in suggestions)
        assert suggestions[0] is not suggestions[1]

    def test_division_constraint_noop_when_missing_keys(self) -> None:
        """Division constraint is vacuously satisfied when keys are absent."""
        constraint = DivisionConstraint(divisor_key="trainer.minibatch_size", dividend_key="trainer.batch_size")
        suggestion = {"trainer": {"batch_size": 64}}
        applied = constraint.apply(copy.deepcopy(suggestion))
        assert applied == suggestion
        assert constraint.is_satisfied(suggestion)

    def test_division_constraint_adjusts_dividend(self) -> None:
        """Constraint can adjust dividend instead of divisor."""
        constraint = DivisionConstraint(
            divisor_key="trainer.minibatch_size",
            dividend_key="trainer.batch_size",
            adjust_target="dividend",
        )
        suggestion = {"trainer": {"batch_size": 65, "minibatch_size": 16}}
        applied = constraint.apply(copy.deepcopy(suggestion))
        assert applied["trainer"]["batch_size"] == 64  # largest multiple of 16 <= 65
        assert applied["trainer"]["minibatch_size"] == 16
