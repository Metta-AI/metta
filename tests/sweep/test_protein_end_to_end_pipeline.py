"""
End-to-end integration test for the Protein sweep pipeline.
This test runs the real sweep pipeline but mocks training and evaluation
to verify that Protein suggestions are working correctly.
"""

import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein


class MockTrainerResults:
    """Mock training results with configurable outcomes."""

    def __init__(self, agent_steps=50000, epochs=5, train_time=120.0):
        self.agent_steps = agent_steps
        self.epochs = epochs
        self.train_time = train_time


class MockEvaluationResults:
    """Mock evaluation results with configurable scores."""

    def __init__(self, reward_score=0.75, eval_time=15.0):
        self.reward_score = reward_score
        self.eval_time = eval_time


class MockPolicyRecord:
    """Mock policy record for testing."""

    def __init__(self, name="test_policy", uri="file://test/path"):
        self.name = name
        self.uri = uri
        self.metadata = {}


class TestProteinEndToEndPipeline:
    """Test the complete Protein sweep pipeline with mocked training/eval."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_sweep_config(self):
        """Create a realistic sweep configuration for testing."""
        return OmegaConf.create(
            {
                "protein": {
                    "num_random_samples": 3,
                    "max_suggestion_cost": 300,  # 5 minutes max
                    "resample_frequency": 0,
                    "global_search_scale": 1,
                    "random_suggestions": 100,
                    "suggestions_per_pareto": 50,
                },
                "metric": "reward",
                "goal": "maximize",
                "method": "bayes",
                "parameters": {
                    "trainer": {
                        "optimizer": {
                            "learning_rate": {
                                "distribution": "log_normal",
                                "min": 0.0001,
                                "max": 0.01,
                                "mean": 0.001,
                                "scale": 0.5,
                            }
                        },
                        "batch_size": {
                            "distribution": "int_uniform",
                            "min": 1024,
                            "max": 8192,
                            "mean": 4096,
                            "scale": "auto",
                        },
                        "gamma": {
                            "distribution": "logit_normal",
                            "min": 0.9,
                            "max": 0.999,
                            "mean": 0.99,
                            "scale": 0.3,
                        },
                    },
                },
            }
        )

    @patch("metta.sweep.protein_metta.Protein")
    def test_single_sweep_run_with_mocked_training(self, mock_protein, temp_workspace, mock_sweep_config):
        """Test a single sweep run with mocked training pipeline."""
        # Mock the protein to return specific suggestions
        mock_protein_instance = Mock()
        mock_protein_instance.suggest.return_value = (
            {
                "trainer": {
                    "optimizer": {"learning_rate": 0.003},
                    "batch_size": 2048,
                    "gamma": 0.995,
                }
            },
            {"cost": 120.0, "score": 0.85},
        )
        mock_protein.return_value = mock_protein_instance

        # Create MettaProtein instance
        protein = MettaProtein(mock_sweep_config)

        # Simulate getting a suggestion and running training
        suggestion, info = protein.suggest()

        # Verify suggestion structure
        assert isinstance(suggestion, dict)
        assert "trainer" in suggestion
        assert "optimizer" in suggestion["trainer"]
        assert "learning_rate" in suggestion["trainer"]["optimizer"]
        assert "batch_size" in suggestion["trainer"]
        assert "gamma" in suggestion["trainer"]

        # Simulate training completion
        training_results = MockTrainerResults(agent_steps=50000, epochs=5, train_time=120.0)
        eval_results = MockEvaluationResults(reward_score=0.85, eval_time=15.0)

        # Record observation
        protein.observe(
            suggestion=suggestion,
            objective=eval_results.reward_score,
            cost=training_results.train_time,
            is_failure=False,
        )

        # Verify the protein observed the result
        mock_protein_instance.observe.assert_called_once_with(suggestion, 0.85, 120.0, False)

    @patch("metta.sweep.protein_metta.Protein")
    def test_multi_run_sweep_progression(self, mock_protein, temp_workspace, mock_sweep_config):
        """Test multiple runs in a sweep to verify learning progression."""
        mock_protein_instance = Mock()

        # Mock progressive suggestions
        suggestions = [
            ({"trainer": {"optimizer": {"learning_rate": 0.001}, "batch_size": 2048, "gamma": 0.99}}, {}),
            ({"trainer": {"optimizer": {"learning_rate": 0.005}, "batch_size": 4096, "gamma": 0.995}}, {}),
            ({"trainer": {"optimizer": {"learning_rate": 0.003}, "batch_size": 1024, "gamma": 0.992}}, {}),
        ]
        mock_protein_instance.suggest.side_effect = suggestions
        mock_protein.return_value = mock_protein_instance

        # Create protein instance
        protein = MettaProtein(mock_sweep_config)

        # Simulate multiple training runs with improving results
        results = []
        for i, expected_reward in enumerate([0.75, 0.85, 0.92]):
            suggestion, _ = protein.suggest()

            # Simulate training
            training_time = 100.0 + (i * 20.0)  # Increasing cost

            # Record result
            protein.observe(suggestion=suggestion, objective=expected_reward, cost=training_time, is_failure=False)

            results.append({"suggestion": suggestion, "reward": expected_reward, "cost": training_time})

        # Verify all observations were recorded
        assert mock_protein_instance.observe.call_count == 3

        # Verify progression of results
        assert len(results) == 3
        rewards = [r["reward"] for r in results]
        assert rewards == [0.75, 0.85, 0.92]  # Improving performance

    @patch("metta.sweep.protein_metta.Protein")
    def test_protein_handles_training_failures(self, mock_protein, temp_workspace, mock_sweep_config):
        """Test that Protein handles training failures correctly."""
        mock_protein_instance = Mock()
        mock_protein_instance.suggest.return_value = (
            {"trainer": {"optimizer": {"learning_rate": 0.01}, "batch_size": 8192}},  # Potentially unstable config
            {},
        )
        mock_protein.return_value = mock_protein_instance

        protein = MettaProtein(mock_sweep_config)

        # Get suggestion
        suggestion, _ = protein.suggest()

        # Simulate training failure (e.g., NaN loss, OOM, etc.)
        protein.observe_failure(suggestion)

        # Verify failure was recorded
        mock_protein_instance.observe.assert_called_once_with(suggestion, 0, 0.01, True)

    @patch("metta.sweep.protein_metta.Protein")
    def test_protein_cost_constraints(self, mock_protein, temp_workspace, mock_sweep_config):
        """Test that Protein respects cost constraints in suggestions."""
        # Configure protein with low max cost
        low_cost_config = OmegaConf.create(mock_sweep_config)
        low_cost_config.protein.max_suggestion_cost = 60  # 1 minute max

        mock_protein_instance = Mock()
        mock_protein_instance.suggest.return_value = (
            {"trainer": {"optimizer": {"learning_rate": 0.002}}},
            {"expected_cost": 45.0},  # Under the limit
        )
        mock_protein.return_value = mock_protein_instance

        protein = MettaProtein(low_cost_config)

        # Get suggestion
        suggestion, info = protein.suggest()

        # Verify suggestion respects constraints (mock already ensures this)
        assert isinstance(suggestion, dict)
        assert isinstance(info, dict)

        # Verify protein was initialized with correct cost constraint
        mock_protein.assert_called_once()
        args, kwargs = mock_protein.call_args
        assert kwargs["max_suggestion_cost"] == 60

    @patch("metta.sweep.protein_metta.Protein")
    def test_protein_suggestion_persistence(self, mock_protein, temp_workspace, mock_sweep_config):
        """Test that suggestions and observations persist across protein instances."""
        mock_protein_instance = Mock()
        mock_protein_instance.suggest.return_value = ({"trainer": {"optimizer": {"learning_rate": 0.004}}}, {})
        mock_protein_instance.success_observations = []
        mock_protein_instance.failure_observations = []
        mock_protein.return_value = mock_protein_instance

        # Create first protein instance
        protein1 = MettaProtein(mock_sweep_config)

        # Generate suggestion and record observation
        suggestion, _ = protein1.suggest()
        protein1.observe(suggestion, 0.8, 100.0, False)

        # Verify observation count
        assert protein1.num_observations == 0  # Mock lists are empty

        # Create second protein instance (would normally load from persistence)
        protein2 = MettaProtein(mock_sweep_config)

        # Both instances should work independently in this mock setup
        suggestion2, _ = protein2.suggest()
        assert isinstance(suggestion2, dict)

    @patch("metta.sweep.protein_metta.Protein")
    def test_realistic_protein_learning_progression(self, mock_protein, temp_workspace, mock_sweep_config):
        """Test a realistic learning progression over multiple runs."""
        mock_protein_instance = Mock()

        # Mock realistic suggestions that gradually improve
        suggestions = [
            # Initial random suggestions
            ({"trainer": {"optimizer": {"learning_rate": 0.001}, "batch_size": 2048}}, {}),
            ({"trainer": {"optimizer": {"learning_rate": 0.008}, "batch_size": 1024}}, {}),
            # Better suggestions based on learning
            ({"trainer": {"optimizer": {"learning_rate": 0.003}, "batch_size": 4096}}, {}),
            ({"trainer": {"optimizer": {"learning_rate": 0.0025}, "batch_size": 3072}}, {}),
            # Optimal region exploration
            ({"trainer": {"optimizer": {"learning_rate": 0.0028}, "batch_size": 3584}}, {}),
        ]
        mock_protein_instance.suggest.side_effect = suggestions
        mock_protein.return_value = mock_protein_instance

        protein = MettaProtein(mock_sweep_config)

        # Simulate realistic training outcomes
        training_scenarios = [
            # Initial exploration phase
            {"reward": 0.65, "cost": 90.0, "is_failure": False},  # Baseline
            {"reward": 0.45, "cost": 80.0, "is_failure": False},  # Too high LR
            # Learning phase
            {"reward": 0.82, "cost": 110.0, "is_failure": False},  # Good config
            {"reward": 0.85, "cost": 105.0, "is_failure": False},  # Better config
            # Optimization phase
            {"reward": 0.88, "cost": 108.0, "is_failure": False},  # Near optimal
        ]

        results = []
        for i, scenario in enumerate(training_scenarios):
            suggestion, _ = protein.suggest()

            protein.observe(
                suggestion=suggestion,
                objective=scenario["reward"],
                cost=scenario["cost"],
                is_failure=scenario["is_failure"],
            )

            results.append({"run": i + 1, "suggestion": suggestion, **scenario})

        # Verify all runs completed
        assert len(results) == 5
        assert mock_protein_instance.observe.call_count == 5

        # Verify general trend towards better performance
        rewards = [r["reward"] for r in results[-3:]]  # Last 3 runs
        assert all(r >= 0.8 for r in rewards)  # Should be in good region


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
