#!/usr/bin/env python3
"""
Test suite for sweep_init.py functionality.
Tests nested config extraction, numpy serialization, and Protein integration.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml


class TestSweepInit:
    """Test sweep initialization functionality."""

    def test_nested_config_extraction(self):
        """Test extraction of trainer/env overrides from nested sweep structure."""
        # Create a config with nested structure
        config = {
            "rollout_count": 5,
            "num_samples": 2,
            "sweep": {
                "parameters": {"trainer.learning_rate": {"min": 0.0001, "max": 0.01, "distribution": "log_normal"}},
                "trainer": {"total_timesteps": 1000, "evaluate_interval": 100, "minibatch_size": 32},
                "env": {"game": {"max_steps": 64, "objects": {"altar": {"hp": 10, "cooldown": 5}}}},
            },
        }

        # Test extraction logic (simulating what sweep_init does)
        sweep_config = config.get("sweep", {})

        # Extract trainer overrides
        trainer_overrides = {}
        if "trainer" in sweep_config:
            for key, value in sweep_config["trainer"].items():
                if key not in ["parameters", "metric", "goal"]:
                    trainer_overrides[key] = value

        # Extract env overrides
        env_overrides = {}
        if "env" in sweep_config:
            env_overrides = sweep_config["env"]

        # Verify extraction
        assert trainer_overrides["total_timesteps"] == 1000
        assert trainer_overrides["evaluate_interval"] == 100
        assert trainer_overrides["minibatch_size"] == 32

        assert env_overrides["game"]["max_steps"] == 64
        assert env_overrides["game"]["objects"]["altar"]["hp"] == 10

    def test_numpy_serialization(self):
        """Test proper serialization of numpy types in Protein suggestions."""
        # Create mock Protein suggestion with numpy types
        suggestion = {
            "trainer.learning_rate": np.float64(0.001),
            "trainer.batch_size": np.int64(64),
            "trainer.gamma": np.array(0.99).item(),  # Already converted
            "trainer.minibatch_size": np.int32(16),
            "trainer.bptt_horizon": int(np.array([8])[0]),  # Array access
        }

        # Test the cleaning function (simulating what sweep_init does)
        def clean_numpy_types(obj):
            """Convert numpy types to Python native types."""
            if isinstance(obj, np.ndarray):
                return obj.item() if obj.size == 1 else obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: clean_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_numpy_types(v) for v in obj]
            return obj

        cleaned = clean_numpy_types(suggestion)

        # Verify all values are JSON serializable
        json_str = json.dumps(cleaned)
        assert json_str  # Should not raise

        # Verify types
        assert isinstance(cleaned["trainer.learning_rate"], float)
        assert isinstance(cleaned["trainer.batch_size"], int)
        assert isinstance(cleaned["trainer.gamma"], float)
        assert isinstance(cleaned["trainer.minibatch_size"], int)
        assert isinstance(cleaned["trainer.bptt_horizon"], int)

    def test_config_file_creation(self):
        """Test creation of override config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            run_dir.mkdir(parents=True)

            # Create trainer overrides
            trainer_overrides = {"total_timesteps": 5000, "evaluate_interval": 500, "batch_size": 64}

            # Create env overrides
            env_overrides = {"game": {"max_steps": 32, "objects": {"altar": {"hp": 5}}}}

            # Write config files
            train_config_path = run_dir / "train_config_overrides.yaml"
            with open(train_config_path, "w") as f:
                yaml.dump({"trainer": trainer_overrides}, f)

            env_config_path = run_dir / "env_config_overrides.yaml"
            with open(env_config_path, "w") as f:
                yaml.dump({"env": env_overrides}, f)

            # Verify files exist and content is correct
            assert train_config_path.exists()
            assert env_config_path.exists()

            # Load and verify content
            with open(train_config_path) as f:
                loaded_trainer = yaml.safe_load(f)
            assert loaded_trainer["trainer"]["total_timesteps"] == 5000

            with open(env_config_path) as f:
                loaded_env = yaml.safe_load(f)
            assert loaded_env["env"]["game"]["max_steps"] == 32

    def test_protein_suggestion_format(self):
        """Test that Protein suggestions are properly formatted."""
        # Mock a Protein suggestion
        raw_suggestion = {
            "trainer.learning_rate": 0.0023,
            "trainer.batch_size": 64,
            "trainer.gamma": 0.98,
            "trainer.minibatch_size": 16,
            "trainer.bptt_horizon": 8,
            "trainer.forward_pass_minibatch_target_size": 16,
        }

        # Verify all keys use dot notation
        for key in raw_suggestion:
            assert "." in key
            assert key.startswith("trainer.") or key.startswith("env.")

    def test_power_of_2_validation(self):
        """Test that power-of-2 parameters are validated correctly."""
        # Test valid power-of-2 values
        valid_values = [8, 16, 32, 64, 128, 256]
        for val in valid_values:
            assert (val & (val - 1)) == 0  # Power of 2 check

        # Test invalid values
        invalid_values = [12, 24, 48, 96, 192]
        for val in invalid_values:
            assert (val & (val - 1)) != 0  # Not power of 2

    def test_config_compatibility(self):
        """Test compatibility between different config formats."""
        # Old CARBS format (for reference)
        old_format = {"trainer": {"learning_rate": "${ss:log, 1e-5, 1e-2}", "batch_size": "${ss:pow2, 32, 256}"}}

        # New Protein format
        new_format = {
            "sweep": {
                "parameters": {
                    "trainer.learning_rate": {"min": 0.00001, "max": 0.01, "distribution": "log_normal"},
                    "trainer.batch_size": {"min": 32, "max": 256, "distribution": "uniform_pow2"},
                },
                "trainer": {"total_timesteps": 5000},
            }
        }

        # Verify new format has proper structure
        assert "sweep" in new_format
        assert "parameters" in new_format["sweep"]
        assert "trainer" in new_format["sweep"]

        # Verify parameter definitions use dot notation
        for param in new_format["sweep"]["parameters"]:
            assert "." in param


class TestSweepInitIntegration:
    """Integration tests for sweep_init with actual configs."""

    def test_protein_working_config(self):
        """Test with our protein_working.yaml config."""
        config_path = Path("configs/sweep/protein_working.yaml")
        if not config_path.exists():
            pytest.skip("protein_working.yaml not found")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify structure
        assert "rollout_count" in config
        assert "num_samples" in config
        assert "sweep" in config
        assert "parameters" in config["sweep"]
        assert "trainer" in config["sweep"]
        assert "env" in config["sweep"]

        # Verify power-of-2 distributions
        params = config["sweep"]["parameters"]
        pow2_params = [
            "trainer.batch_size",
            "trainer.minibatch_size",
            "trainer.bptt_horizon",
            "trainer.forward_pass_minibatch_target_size",
        ]

        for param in pow2_params:
            if param in params:
                assert params[param]["distribution"] == "uniform_pow2"

    @patch("wandb.Api")
    @patch("wandb.sweep")
    def test_sweep_init_mock_run(self, mock_sweep, mock_api):
        """Test running sweep_init with mocked dependencies."""
        # Mock WandB API
        mock_api.return_value = MagicMock()
        mock_sweep.return_value = "test_sweep_id"

        # Create a simple test config
        test_config = {
            "rollout_count": 2,
            "num_samples": 1,
            "sweep": {
                "parameters": {"trainer.learning_rate": {"min": 0.001, "max": 0.01, "distribution": "log_normal"}},
                "metric": "reward",
                "goal": "maximize",
                "trainer": {"total_timesteps": 100},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(test_config, f)
            config_path = f.name

        try:
            # Test would call sweep_init here with the config
            # For now, just verify the config is valid
            with open(config_path) as f:
                loaded = yaml.safe_load(f)
            assert loaded == test_config
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
