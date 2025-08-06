"""Integration tests for the complete sweep pipeline.

These tests cover the end-to-end workflow:
1. Sweep initialization (MettaProtein)
2. Parameter suggestions
3. Configuration application
4. Multi-run sweep progression
"""

import os
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein
from tools.sweep_prepare_run import apply_protein_suggestion


class TestSweepPipelineIntegration:
    """Integration tests for the complete sweep pipeline."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self):
        """Setup test environment with temporary directories."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, "data")
        self.sweep_dir = os.path.join(self.data_dir, "sweep", "test_sweep")
        self.runs_dir = os.path.join(self.sweep_dir, "runs")

        os.makedirs(self.runs_dir, exist_ok=True)

        yield

        # Cleanup
        if hasattr(self, "temp_dir"):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.fixture
    def base_sweep_config(self):
        """Create a basic sweep configuration for testing."""
        return OmegaConf.create(
            {
                "protein": {
                    "num_random_samples": 3,
                    "max_suggestion_cost": 300,
                    "resample_frequency": 0,
                    "global_search_scale": 1,
                    "random_suggestions": 10,
                    "suggestions_per_pareto": 5,
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
                            "min": 32,
                            "max": 128,
                            "mean": 64,
                            "scale": "auto",
                        },
                    },
                },
            }
        )

    @pytest.fixture
    def base_train_config(self):
        """Create a basic training configuration for testing."""
        return OmegaConf.create(
            {
                "run": "test_run",
                "run_dir": os.path.join(self.runs_dir, "test_run"),
                "device": "cpu",
                "trainer": {
                    "_target_": "metta.rl.trainer.MettaTrainer",
                    "total_timesteps": 100,
                    "evaluate_interval": 50,
                    "optimizer": {
                        "learning_rate": 0.001,
                    },
                    "batch_size": 64,
                },
            }
        )

    @patch("metta.sweep.protein_metta.Protein")
    def test_protein_suggestion_generation(self, mock_protein, base_sweep_config):
        """Test basic protein suggestion generation."""
        # Mock the protein to return a specific suggestion
        mock_protein_instance = Mock()
        mock_protein_instance.suggest.return_value = (
            {
                "trainer": {
                    "optimizer": {"learning_rate": 0.003},
                    "batch_size": 64,
                }
            },
            {"cost": 100.0},
        )
        mock_protein.return_value = mock_protein_instance

        # Create MettaProtein instance
        metta_protein = MettaProtein(base_sweep_config)

        # Generate suggestion and verify it's valid
        suggestion, info = metta_protein.suggest()

        # Verify suggestion structure
        assert isinstance(suggestion, dict)
        assert isinstance(info, dict)
        assert "trainer" in suggestion
        assert "optimizer" in suggestion["trainer"]
        assert "learning_rate" in suggestion["trainer"]["optimizer"]

    @patch("metta.sweep.protein_metta.Protein")
    def test_config_suggestion_application(self, mock_protein, base_sweep_config, base_train_config):
        """Test application of Protein suggestions to training config."""
        # Mock protein suggestion
        mock_protein_instance = Mock()
        mock_protein_instance.suggest.return_value = (
            {
                "trainer": {
                    "optimizer": {"learning_rate": 0.005},
                    "batch_size": 96,
                }
            },
            {},
        )
        mock_protein.return_value = mock_protein_instance

        # Create MettaProtein and get suggestion
        metta_protein = MettaProtein(base_sweep_config)
        suggestion, _ = metta_protein.suggest()

        # Apply suggestion to config
        apply_protein_suggestion(base_train_config, suggestion)

        # Verify suggestion was applied correctly
        assert base_train_config.trainer.optimizer.learning_rate == 0.005
        assert base_train_config.trainer.batch_size == 96

    @patch("metta.sweep.protein_metta.Protein")
    def test_observation_recording(self, mock_protein, base_sweep_config):
        """Test that observations can be recorded to protein."""
        mock_protein_instance = Mock()
        mock_protein_instance.suggest.return_value = (
            {"trainer": {"optimizer": {"learning_rate": 0.002}}},
            {},
        )
        mock_protein.return_value = mock_protein_instance

        # Create MettaProtein
        metta_protein = MettaProtein(base_sweep_config)

        # Get a suggestion first
        suggestion, _ = metta_protein.suggest()

        # Record an observation
        metta_protein.observe(suggestion=suggestion, objective=0.85, cost=120.0, is_failure=False)

        # Verify the observe method was called on the underlying protein
        mock_protein_instance.observe.assert_called_once_with(suggestion, 0.85, 120.0, False)

    @patch("metta.sweep.protein_metta.Protein")
    def test_failure_recording(self, mock_protein, base_sweep_config):
        """Test that failures can be recorded to protein."""
        mock_protein_instance = Mock()
        mock_protein_instance.suggest.return_value = (
            {"trainer": {"optimizer": {"learning_rate": 0.002}}},
            {},
        )
        mock_protein.return_value = mock_protein_instance

        # Create MettaProtein
        metta_protein = MettaProtein(base_sweep_config)

        # Get a suggestion first
        suggestion, _ = metta_protein.suggest()

        # Record a failure
        metta_protein.observe_failure(suggestion)

        # Verify the observe method was called with failure parameters
        mock_protein_instance.observe.assert_called_once_with(suggestion, 0, 0.01, True)

    @patch("metta.sweep.protein_metta.Protein")
    def test_multi_run_progression(self, mock_protein, base_sweep_config):
        """Test multiple runs with different suggestions."""
        mock_protein_instance = Mock()

        # Mock multiple suggestions
        suggestions = [
            ({"trainer": {"optimizer": {"learning_rate": 0.001}}}, {}),
            ({"trainer": {"optimizer": {"learning_rate": 0.003}}}, {}),
            ({"trainer": {"optimizer": {"learning_rate": 0.007}}}, {}),
        ]
        mock_protein_instance.suggest.side_effect = suggestions
        mock_protein.return_value = mock_protein_instance

        # Create MettaProtein
        metta_protein = MettaProtein(base_sweep_config)

        # Simulate multiple runs
        results = []
        for i in range(3):
            suggestion, _ = metta_protein.suggest()
            results.append(suggestion)

            # Simulate recording observation
            objective = 0.8 + (i * 0.05)  # Increasing performance
            metta_protein.observe(suggestion, objective, 100.0, False)

        # Verify we got different suggestions
        assert len(results) == 3
        learning_rates = [r["trainer"]["optimizer"]["learning_rate"] for r in results]
        assert len(set(learning_rates)) == 3  # All different

        # Verify all observations were recorded
        assert mock_protein_instance.observe.call_count == 3

    @patch("metta.sweep.protein_metta.Protein")
    def test_num_observations_property(self, mock_protein, base_sweep_config):
        """Test the num_observations property."""
        mock_protein_instance = Mock()
        mock_protein_instance.success_observations = [1, 2, 3]
        mock_protein_instance.failure_observations = [4, 5]
        mock_protein.return_value = mock_protein_instance

        metta_protein = MettaProtein(base_sweep_config)

        assert metta_protein.num_observations == 5
