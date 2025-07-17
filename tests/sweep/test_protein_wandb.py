"""Tests for WandbProtein class with real wandb integration."""

import os
import tempfile
from unittest.mock import Mock

import pytest
import wandb

from metta.sweep.protein_wandb import WandbProtein


class TestWandbProtein:
    """Test cases for WandbProtein class with real wandb integration."""

    @pytest.fixture(autouse=True)
    def setup_wandb(self):
        """Setup wandb for testing with offline mode."""
        # Use a temporary directory for wandb
        self.temp_dir = tempfile.mkdtemp()
        os.environ["WANDB_DIR"] = self.temp_dir
        os.environ["WANDB_MODE"] = "offline"  # Prevent actual API calls
        os.environ["WANDB_SILENT"] = "true"  # Reduce logging noise

        yield

        # Cleanup
        if hasattr(self, "temp_dir"):
            import shutil

            shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_protein(self):
        """Create a mock Protein instance with proper behavior."""
        protein = Mock()
        protein.hyperparameters = Mock()
        protein.hyperparameters.flat_spaces = {"learning_rate": Mock(), "batch_size": Mock()}
        return protein

    def test_wandb_protein_init_with_real_run(self, mock_protein):
        """Test WandbProtein initialization with a real wandb run."""
        # Initialize a real wandb run with sweep
        wandb.init(project="test_protein", job_type="test", mode="offline", config={"test_param": 1})

        try:
            # Mock the protein to return a proper suggestion
            mock_protein.suggest.return_value = (
                {"learning_rate": 0.001, "batch_size": 32},
                {"cost": 100.0, "score": 0.95, "rating": 0.8},
            )

            # Create WandbProtein - this should work with real wandb
            wandb_protein = WandbProtein(mock_protein)

            # Verify basic attributes are set
            assert wandb_protein._protein == mock_protein
            assert wandb_protein._wandb_run is not None
            assert wandb_protein._suggestion_info is not None

            # Verify wandb run has protein state
            assert wandb_protein._wandb_run.summary.get("protein.state") is not None

        finally:
            wandb.finish()

    def test_suggestion_generation_and_storage(self, mock_protein):
        """Test that suggestions are generated and stored in wandb."""
        wandb.init(project="test_protein", job_type="test", mode="offline")

        try:
            # Mock protein to return specific suggestion and info
            expected_suggestion = {"learning_rate": 0.002, "batch_size": 64}
            expected_info = {"cost": 75.0, "score": 0.88, "rating": 0.7}
            mock_protein.suggest.return_value = (expected_suggestion, expected_info)

            # Create WandbProtein
            wandb_protein = WandbProtein(mock_protein)

            # Test suggest() method returns the stored info
            suggestion, info = wandb_protein.suggest()

            # Verify the suggestion and info
            assert suggestion == expected_suggestion
            assert info["cost"] == 75.0
            assert info["score"] == 0.88
            assert info["rating"] == 0.7
            # Note: suggestion_uuid was removed during cleanup

            # Verify data was stored in wandb summary
            assert wandb_protein._wandb_run.summary.get("protein.suggestion") == expected_suggestion
            stored_info = wandb_protein._wandb_run.summary.get("protein.suggestion_info")
            assert stored_info["cost"] == 75.0
            assert stored_info["score"] == 0.88

        finally:
            wandb.finish()

    def test_observe_functionality(self, mock_protein):
        """Test the record_observation functionality with real wandb integration."""
        wandb.init(project="test_protein", job_type="test", mode="offline")

        try:
            # Setup mock protein
            mock_protein.suggest.return_value = ({"learning_rate": 0.001}, {"cost": 50.0})

            wandb_protein = WandbProtein(mock_protein)

            # Test record_observation method
            wandb_protein.record_observation(objective=0.95, cost=120.0)

            # Verify the wandb summary was updated
            assert wandb_protein._wandb_run.summary.get("protein.objective") == 0.95
            assert wandb_protein._wandb_run.summary.get("protein.cost") == 120.0
            assert wandb_protein._wandb_run.summary.get("protein.state") == "success"

        finally:
            wandb.finish()

    def test_transform_suggestion_with_numpy(self, mock_protein):
        """Test _transform_suggestion with numpy types."""
        wandb.init(project="test_protein", job_type="test", mode="offline")

        try:
            mock_protein.suggest.return_value = ({}, {"cost": 0.0})

            wandb_protein = WandbProtein(mock_protein)

            # Test numpy type conversion
            import numpy as np

            suggestion_with_numpy = {
                "learning_rate": np.float64(0.001),
                "batch_size": np.int32(64),
                "regularization": np.float32(0.01),
                "nested": {"param": np.array([1, 2, 3])},
            }

            result = wandb_protein._transform_suggestion(suggestion_with_numpy)

            # WandbProtein._transform_suggestion DOES convert numpy types
            assert isinstance(result["learning_rate"], float)
            assert result["learning_rate"] == pytest.approx(0.001)
            assert isinstance(result["batch_size"], int)
            assert result["batch_size"] == 64
            assert isinstance(result["regularization"], float)
            assert result["regularization"] == pytest.approx(0.01)
            assert isinstance(result["nested"]["param"], list)
            assert result["nested"]["param"] == [1, 2, 3]

        finally:
            wandb.finish()

    def test_wandb_config_integration(self, mock_protein):
        """Test that wandb config is properly updated with suggestions."""
        wandb.init(project="test_protein", job_type="test", mode="offline", config={"initial_param": "value"})

        try:
            # Mock protein to return specific suggestion
            expected_suggestion = {"learning_rate": 0.003, "batch_size": 128, "optimizer": "adam"}
            mock_protein.suggest.return_value = (expected_suggestion, {"cost": 25.0})

            # Create WandbProtein - this will update wandb config
            wandb_protein = WandbProtein(mock_protein)

            # Verify wandb config was updated with suggestion
            config = wandb_protein._wandb_run.config
            assert config["learning_rate"] == 0.003
            assert config["batch_size"] == 128
            assert config["optimizer"] == "adam"

            # Verify original config is preserved
            assert config["initial_param"] == "value"

            # Note: separate parameters section was removed during cleanup -
            # suggestions are now applied directly to the main config

        finally:
            wandb.finish()

    def test_protein_state_management(self, mock_protein):
        """Test that protein state is properly managed in wandb."""
        wandb.init(project="test_protein", job_type="test", mode="offline")

        try:
            mock_protein.suggest.return_value = ({"param": 1}, {"cost": 10.0})

            assert wandb.run is not None

            # Initially, there should be no protein state
            assert wandb.run.summary.get("protein.state") is None

            # Create WandbProtein
            _ = WandbProtein(mock_protein)

            # After initialization, protein state should be "running"
            assert wandb.run.summary.get("protein.state") == "running"

        finally:
            wandb.finish()

    def test_multiple_suggestions(self, mock_protein):
        """Test generating multiple suggestions in sequence."""
        wandb.init(project="test_protein", job_type="test", mode="offline")

        try:
            # First suggestion during initialization
            mock_protein.suggest.return_value = ({"learning_rate": 0.001}, {"cost": 100.0})

            wandb_protein = WandbProtein(mock_protein)

            # Get initial suggestion
            suggestion1, info1 = wandb_protein.suggest()
            assert suggestion1["learning_rate"] == 0.001
            assert info1["cost"] == 100.0

            # Now change what protein.suggest returns for the next call
            mock_protein.suggest.return_value = ({"learning_rate": 0.002}, {"cost": 50.0})

            # Generate new suggestion
            wandb_protein._generate_protein_suggestion()
            suggestion2, info2 = wandb_protein.suggest()
            assert suggestion2["learning_rate"] == 0.002
            assert info2["cost"] == 50.0

            # Verify that wandb summary was updated with latest suggestion
            latest_suggestion = wandb_protein._wandb_run.summary.get("protein.suggestion")
            assert latest_suggestion["learning_rate"] == 0.002

        finally:
            wandb.finish()

    def test_error_handling_with_real_wandb(self, mock_protein):
        """Test error handling when protein.suggest() fails with real wandb."""
        wandb.init(project="test_protein", job_type="test", mode="offline")

        try:
            # Make protein.suggest() raise an exception
            mock_protein.suggest.side_effect = RuntimeError("GP optimization failed")

            # This should raise the exception during initialization
            with pytest.raises(RuntimeError, match="GP optimization failed"):
                WandbProtein(mock_protein)

        finally:
            wandb.finish()

    def test_suggestion_from_run_functionality(self, mock_protein):
        """Test loading suggestions from wandb run data."""
        wandb.init(project="test_protein", job_type="test", mode="offline")

        try:
            # Create a WandbProtein instance first
            mock_protein.suggest.return_value = ({"param": 1}, {"cost": 10.0})
            wandb_protein = WandbProtein(mock_protein)

            # Create a mock run with stored data (simulating historical run)
            mock_run = Mock()
            mock_run.id = "historical_run_123"
            mock_run.summary = {
                "protein.suggestion": {"learning_rate": 0.005, "batch_size": 256},
                "protein.suggestion_info": {"cost": 15.0, "score": 0.99, "rating": 0.95},
            }

            # Test loading suggestion from run
            suggestion, info = wandb_protein._suggestion_from_run(mock_run)

            # Verify loaded data
            assert suggestion["learning_rate"] == 0.005
            assert suggestion["batch_size"] == 256
            assert info["cost"] == 15.0
            assert info["score"] == 0.99
            assert info["rating"] == 0.95

        finally:
            wandb.finish()
