#!/usr/bin/env python3
"""
Comprehensive test suite for Protein optimization implementation.
Tests all critical functionality needed for production use.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from metta.rl.protein_opt.metta_protein import MettaProtein


class MockWandbRun:
    """Mock WandB run that behaves like the real thing."""

    def __init__(self, run_id="test_run", sweep_id="test_sweep"):
        self.id = run_id
        self.name = f"run_{run_id}"
        self.sweep_id = sweep_id
        self.entity = "test_entity"
        self.project = "test_project"
        self.summary = MockSummary()
        self.config = MockConfig()
        self.heartbeat_at = datetime.now(timezone.utc)


class MockConfig:
    """Mock WandB config with proper dictionary behavior."""

    def __init__(self):
        self._data = {}
        self.__dict__["_locked"] = {}

    def update(self, data, allow_val_change=False):
        self._data.update(data)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def items(self):
        return self._data.items()

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


class MockSummary:
    """Mock WandB summary with proper state tracking."""

    def __init__(self):
        self._data = {}

    def get(self, key, default=None):
        return self._data.get(key, default)

    def update(self, data):
        self._data.update(data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


@pytest.fixture
def mock_wandb_api():
    """Mock WandB API with no previous runs."""
    api = MagicMock()
    runs_collection = MagicMock()
    runs_collection.__iter__ = lambda self: iter([])
    api.runs.return_value = runs_collection
    return api


@pytest.fixture
def basic_sweep_config():
    """Basic sweep configuration for testing."""
    return OmegaConf.create(
        {
            "parameters": {
                "learning_rate": {"min": 1e-5, "max": 1e-1, "mean": 1e-3, "scale": 1, "distribution": "log_normal"},
                "batch_size": {"min": 16, "max": 128, "mean": 64, "scale": 1, "distribution": "int_uniform"},
            },
            "metric": "reward",
            "goal": "maximize",
        }
    )


class TestMettaProteinCore:
    """Test core MettaProtein functionality."""

    def test_initialization_basic(self, mock_wandb_api, basic_sweep_config):
        """Test basic MettaProtein initialization."""
        mock_run = MockWandbRun()

        with patch("wandb.Api", return_value=mock_wandb_api):
            protein = MettaProtein(basic_sweep_config, wandb_run=mock_run)

            # Verify initialization
            assert protein._wandb_run == mock_run
            assert mock_run.summary.get("protein.state") == "running"

    def test_suggest_basic(self, mock_wandb_api, basic_sweep_config):
        """Test basic parameter suggestion."""
        mock_run = MockWandbRun()

        with patch("wandb.Api", return_value=mock_wandb_api):
            protein = MettaProtein(basic_sweep_config, wandb_run=mock_run)

            suggestion, info = protein.suggest()

            # Verify suggestion format
            assert isinstance(suggestion, dict)
            assert "learning_rate" in suggestion
            assert "batch_size" in suggestion
            assert isinstance(suggestion["learning_rate"], float)
            assert isinstance(suggestion["batch_size"], int)

            # Verify parameter bounds
            assert 1e-5 <= suggestion["learning_rate"] <= 1e-1
            assert 16 <= suggestion["batch_size"] <= 128

    def test_wandb_config_overwrite(self, mock_wandb_api, basic_sweep_config):
        """Test that Protein overwrites WandB agent suggestions."""
        mock_run = MockWandbRun()

        # Set some initial config (simulating WandB agent suggestion)
        mock_run.config.update(
            {
                "learning_rate": 0.999,  # Bad value
                "batch_size": 9999,  # Bad value
            }
        )

        with patch("wandb.Api", return_value=mock_wandb_api):
            protein = MettaProtein(basic_sweep_config, wandb_run=mock_run)

            suggestion, info = protein.suggest()

            # Verify Protein overwrote the bad values
            assert mock_run.config.get("learning_rate") != 0.999
            assert mock_run.config.get("batch_size") != 9999
            assert mock_run.config.get("learning_rate") == suggestion["learning_rate"]
            assert mock_run.config.get("batch_size") == suggestion["batch_size"]

    def test_record_observation(self, mock_wandb_api, basic_sweep_config):
        """Test recording successful observations."""
        mock_run = MockWandbRun()

        with patch("wandb.Api", return_value=mock_wandb_api):
            protein = MettaProtein(basic_sweep_config, wandb_run=mock_run)

            # Record an observation
            protein.record_observation(0.85, 120.5)

            # Verify WandB summary was updated
            assert mock_run.summary.get("protein.state") == "success"
            assert mock_run.summary.get("protein.objective") == 0.85
            assert mock_run.summary.get("protein.cost") == 120.5

    def test_record_failure(self, mock_wandb_api, basic_sweep_config):
        """Test recording failures."""
        mock_run = MockWandbRun()

        with patch("wandb.Api", return_value=mock_wandb_api):
            protein = MettaProtein(basic_sweep_config, wandb_run=mock_run)

            # Record a failure
            protein.record_failure()

            # Verify WandB summary was updated
            assert mock_run.summary.get("protein.state") == "failure"


class TestWandbIntegration:
    """Test WandB-specific integration features."""

    def test_static_record_observation(self):
        """Test static observation recording method."""
        mock_run = MockWandbRun()

        # Set initial state
        mock_run.summary.update({"protein.state": "running"})

        # Record observation using static method
        MettaProtein._record_observation(mock_run, 0.75, 95.0)

        # Verify updates
        assert mock_run.summary.get("protein.state") == "success"
        assert mock_run.summary.get("protein.objective") == 0.75
        assert mock_run.summary.get("protein.cost") == 95.0

    def test_load_previous_runs_empty(self, mock_wandb_api, basic_sweep_config):
        """Test behavior with no previous runs."""
        mock_run = MockWandbRun()

        with patch("wandb.Api", return_value=mock_wandb_api):
            protein = MettaProtein(basic_sweep_config, wandb_run=mock_run)

            # Should initialize with zero observations
            assert protein._num_observations == 0
            assert protein._num_failures == 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_initialization_no_wandb(self, mock_wandb_api, basic_sweep_config):
        """Test initialization without WandB run."""
        with patch("wandb.Api", return_value=mock_wandb_api):
            # Should work with None wandb_run and mock the internal wandb.run
            with patch("wandb.run", None):
                with pytest.raises(AssertionError, match="No active wandb run found"):
                    MettaProtein(basic_sweep_config, wandb_run=None)


class TestConfigParsing:
    """Test configuration parsing variations."""

    def test_config_parsing_nested(self, mock_wandb_api):
        """Test parsing of nested sweep configurations."""
        nested_config = OmegaConf.create(
            {
                "sweep": {
                    "parameters": {
                        "learning_rate": {
                            "min": 1e-5,
                            "max": 1e-1,
                            "mean": 1e-3,
                            "scale": 1,
                            "distribution": "log_normal",
                        }
                    },
                    "metric": "reward",
                    "goal": "maximize",
                }
            }
        )

        mock_run = MockWandbRun()
        with patch("wandb.Api", return_value=mock_wandb_api):
            protein = MettaProtein(nested_config, wandb_run=mock_run)

            suggestion, info = protein.suggest()
            assert "learning_rate" in suggestion

    def test_config_parsing_flat(self, mock_wandb_api, basic_sweep_config):
        """Test parsing of flat configurations."""
        mock_run = MockWandbRun()
        with patch("wandb.Api", return_value=mock_wandb_api):
            protein = MettaProtein(basic_sweep_config, wandb_run=mock_run)

            suggestion, info = protein.suggest()
            assert "learning_rate" in suggestion
            assert "batch_size" in suggestion
