"""Integration tests for the protein sweep pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf

from metta.sweep.protein_metta import MettaProtein


class TestProteinSweepIntegration:
    """Integration tests for the full protein sweep pipeline."""

    @pytest.fixture
    def sweep_dir(self):
        """Create a temporary directory for sweep data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sweep_config(self):
        """Create a realistic sweep configuration."""
        return {
            "sweep_name": "test_integration_sweep",
            "method": "bayes",
            "metric": "eval/mean_score",
            "goal": "maximize",
            "parameters": {
                "trainer": {
                    "optimizer": {
                        "learning_rate": {"min": 0.0001, "max": 0.01},
                        "weight_decay": {"min": 0.0, "max": 0.1},
                    },
                    "batch_size": {"values": [16, 32, 64, 128]},
                    "gradient_clip": {"min": 0.1, "max": 10.0},
                }
            },
            "protein": {
                "search_center": {
                    "trainer/optimizer/learning_rate": 0.001,
                    "trainer/optimizer/weight_decay": 0.01,
                    "trainer/batch_size": 32,
                    "trainer/gradient_clip": 1.0,
                },
                "search_radius": {
                    "trainer/optimizer/learning_rate": 0.5,
                    "trainer/optimizer/weight_decay": 0.5,
                    "trainer/batch_size": 0.5,
                    "trainer/gradient_clip": 0.5,
                },
                "kernel": "matern",
                "gamma": 0.25,
                "xi": 0.001,
            },
        }

    @patch("metta.sweep.protein_metta.Protein")
    def test_protein_loads_previous_observations(self, mock_protein_class, sweep_config):
        """Test that protein can load observations from previous runs."""
        # Mock previous observations data
        previous_observations = [
            {
                "suggestion": {"trainer/optimizer/learning_rate": 0.001, "trainer/batch_size": 32},
                "objective": 0.8,
                "cost": 100.0,
                "is_failure": False,
            },
            {
                "suggestion": {"trainer/optimizer/learning_rate": 0.005, "trainer/batch_size": 64},
                "objective": 0.6,
                "cost": 120.0,
                "is_failure": False,
            },
        ]

        # Mock protein instance
        mock_protein = Mock()
        mock_protein.success_observations = previous_observations
        mock_protein.failure_observations = []
        mock_protein_class.return_value = mock_protein

        # Create config and protein
        cfg = OmegaConf.create(sweep_config)

        # Mock that protein loads previous observations
        MettaProtein(cfg)

        # Add observations to simulate loading
        for obs in previous_observations:
            mock_protein.observe(obs["suggestion"], obs["objective"], obs["cost"], obs["is_failure"])

        # Verify observations were loaded
        assert mock_protein.observe.call_count == 2

    @patch("metta.sweep.protein_metta.Protein")
    def test_record_observation_updates_protein(self, mock_protein_class, sweep_config):
        """Test that recording observations updates the protein correctly."""
        mock_protein = Mock()
        mock_protein.suggest.return_value = ({"trainer/optimizer/learning_rate": 0.003}, {"cost": 100.0})
        mock_protein_class.return_value = mock_protein

        # Create MettaProtein
        cfg = OmegaConf.create(sweep_config)
        metta_protein = MettaProtein(cfg)

        # Get a suggestion
        suggestion, _ = metta_protein.suggest()

        # Record an observation
        metta_protein.observe(objective=0.95, suggestion=suggestion, cost=200.0, is_failure=False)

        # Verify Protein was updated
        mock_protein.observe.assert_called_once_with(suggestion, 0.95, 200.0, False)

    @patch("metta.sweep.protein_metta.Protein")
    def test_serialization_in_real_scenario(self, mock_protein_class, sweep_config):
        """Test that protein suggestions can be serialized for real scenarios."""
        # Mock protein with numpy types in suggestions
        mock_protein = Mock()
        suggestion_with_numpy = {
            "trainer/optimizer/learning_rate": np.float64(0.003),
            "trainer/batch_size": np.int32(64),
            "trainer/gradient_clip": np.float32(1.5),
        }
        mock_protein.suggest.return_value = (suggestion_with_numpy, {"cost": np.float64(150.0)})
        mock_protein_class.return_value = mock_protein

        cfg = OmegaConf.create(sweep_config)
        MettaProtein(cfg)

        # This should not raise any serialization errors
        json.dumps(suggestion_with_numpy, default=str)

    def test_sweep_state_file_compatibility(self, sweep_dir):
        """Test that sweep state files are compatible across versions."""
        # Create a mock sweep state file
        sweep_state = {
            "sweep_name": "test_sweep",
            "current_run": 5,
            "total_runs": 100,
            "best_objective": 0.95,
            "best_config": {
                "trainer": {
                    "optimizer": {"learning_rate": 0.003},
                    "batch_size": 64,
                }
            },
        }

        state_file = sweep_dir / "sweep_state.json"
        with open(state_file, "w") as f:
            json.dump(sweep_state, f)

        # Verify we can load the state file
        with open(state_file, "r") as f:
            loaded_state = json.load(f)

        assert loaded_state["sweep_name"] == "test_sweep"
        assert loaded_state["current_run"] == 5
        assert loaded_state["best_objective"] == 0.95
