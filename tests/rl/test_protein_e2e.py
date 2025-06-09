#!/usr/bin/env python3
"""
End-to-end integration test for Protein optimization workflow.
Tests the complete flow from sweep config to parameter optimization.
"""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from metta.rl.protein_opt.metta_protein import MettaProtein


def test_protein_e2e_workflow():
    """Test complete end-to-end Protein workflow."""

    # Use our actual sweep configs
    sweep_config = OmegaConf.create(
        {
            "parameters": {
                "trainer.learning_rate": {
                    "min": 0.0001,
                    "max": 0.01,
                    "mean": 0.001,
                    "scale": 1,
                    "distribution": "log_normal",
                },
                "trainer.batch_size": {"min": 32, "max": 256, "mean": 64, "scale": 1, "distribution": "int_uniform"},
                "trainer.gamma": {"min": 0.95, "max": 0.999, "mean": 0.99, "scale": 1, "distribution": "uniform"},
                "trainer.clip_param": {"min": 0.1, "max": 0.3, "mean": 0.2, "scale": 1, "distribution": "uniform"},
            },
            "metric": "reward",
            "goal": "maximize",
        }
    )

    # Mock WandB components
    mock_run = MagicMock()
    mock_run.id = "test_run"
    mock_run.summary = {}
    mock_run.config = MockConfig()

    mock_api = MagicMock()
    mock_api.runs.return_value.__iter__ = lambda self: iter([])

    with patch("wandb.Api", return_value=mock_api):
        # Step 1: Initialize Protein
        protein = MettaProtein(sweep_config, wandb_run=mock_run)

        # Step 2: Generate first suggestion
        suggestion1, info1 = protein.suggest()

        # Verify suggestion has all expected parameters
        assert "trainer.learning_rate" in suggestion1
        assert "trainer.batch_size" in suggestion1
        assert "trainer.gamma" in suggestion1
        assert "trainer.clip_param" in suggestion1

        # Verify WandB config was updated with Protein's suggestions
        assert mock_run.config.get("trainer.learning_rate") == suggestion1["trainer.learning_rate"]
        assert mock_run.config.get("trainer.batch_size") == suggestion1["trainer.batch_size"]

        # Step 3: Simulate training with good result
        protein.record_observation(0.85, 100.0)
        assert mock_run.summary.get("protein.state") == "success"
        assert mock_run.summary.get("protein.objective") == 0.85

        # Step 4: Generate second suggestion (should be valid)
        suggestion2, info2 = protein.suggest()

        # Should generate a valid suggestion (may be same if algorithm is deterministic)
        assert isinstance(suggestion2, dict)
        assert "trainer.learning_rate" in suggestion2

        # Step 5: Simulate training with poor result
        protein_run2 = MagicMock()
        protein_run2.summary = {}
        protein.record_failure()

        # Verify failure recorded
        assert mock_run.summary.get("protein.state") == "failure"


class MockConfig:
    """Mock WandB config for e2e testing."""

    def __init__(self):
        self._data = {}
        self.__dict__["_locked"] = {}

    def update(self, data, allow_val_change=False):
        self._data.update(data)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def items(self):
        return self._data.items()


def test_protein_config_compatibility():
    """Test that Protein works with our actual sweep configs."""

    # Test with protein_simple.yaml equivalent
    simple_config = OmegaConf.create(
        {
            "parameters": {
                "trainer.learning_rate": {
                    "min": 0.0001,
                    "max": 0.01,
                    "mean": 0.001,
                    "scale": 1,
                    "distribution": "log_normal",
                },
                "trainer.batch_size": {"min": 32, "max": 256, "mean": 64, "scale": 1, "distribution": "int_uniform"},
                "trainer.gamma": {"min": 0.95, "max": 0.999, "mean": 0.99, "scale": 1, "distribution": "uniform"},
                "trainer.clip_param": {"min": 0.1, "max": 0.3, "mean": 0.2, "scale": 1, "distribution": "uniform"},
            }
        }
    )

    mock_run = MagicMock()
    mock_run.summary = {}
    mock_run.config = MockConfig()

    mock_api = MagicMock()
    mock_api.runs.return_value.__iter__ = lambda self: iter([])

    with patch("wandb.Api", return_value=mock_api):
        protein = MettaProtein(simple_config, wandb_run=mock_run)
        suggestion, info = protein.suggest()

        # Verify nested parameter names work
        assert "trainer.learning_rate" in suggestion
        assert "trainer.batch_size" in suggestion

        # Verify parameter bounds
        assert 0.0001 <= suggestion["trainer.learning_rate"] <= 0.01
        assert 32 <= suggestion["trainer.batch_size"] <= 256
        assert 0.95 <= suggestion["trainer.gamma"] <= 0.999
        assert 0.1 <= suggestion["trainer.clip_param"] <= 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
