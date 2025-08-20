import json
import os
import tempfile
from pathlib import Path

from metta.rl.trainer_config import TrainerConfig
from tools.sweep_config_utils import (
    load_train_job_config_with_overrides,
    merge_train_job_config_overrides,
    save_train_job_override_config,
    validate_train_job_config,
)


def test_load_train_job_config_with_overrides():
    """Test loading train job config with overrides from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a base config dict
        base_cfg = {
            "run_dir": tmpdir,
            "device": "cpu",
            "some_value": 42,
        }

        # Create an override file
        override_path = Path(tmpdir) / "train_config_overrides.json"
        overrides = {
            "device": "cuda",
            "new_value": "test",
        }
        with open(override_path, "w") as f:
            json.dump(overrides, f)

        # Apply overrides
        result_cfg = load_train_job_config_with_overrides(base_cfg)

        # Check that overrides were applied
        assert result_cfg["device"] == "cuda"
        assert result_cfg["some_value"] == 42  # Original value preserved
        assert result_cfg["new_value"] == "test"


def test_load_train_job_config_with_overrides_no_file():
    """Test that config is unchanged when no override file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_cfg = {
            "run_dir": tmpdir,
            "device": "cpu",
            "some_value": 99,
        }

        # No override file exists
        result_cfg = load_train_job_config_with_overrides(base_cfg)

        # Config should be unchanged
        assert result_cfg["device"] == "cpu"
        assert result_cfg["some_value"] == 99


def test_save_train_job_override_config():
    """Test saving train job override config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create configs
        cfg = {
            "run_dir": tmpdir,
        }

        train_cfg = {
            "total_timesteps": 1000000,
            "batch_size": 256,
        }

        # Save overrides
        save_path = save_train_job_override_config(cfg, train_cfg)

        # Check file was created (now JSON instead of YAML)
        assert save_path == os.path.join(tmpdir, "train_config_overrides.json")
        assert os.path.exists(save_path)

        # Load and verify contents
        with open(save_path, "r") as f:
            loaded = json.load(f)
        assert loaded["total_timesteps"] == 1000000
        assert loaded["batch_size"] == 256


def test_merge_train_job_config_overrides():
    """Test merging train job configs with deep merge logic."""
    base_cfg = {
        "device": "cpu",
        "trainer": {
            "total_timesteps": 50000000,
            "batch_size": 512,
            "optimizer": {
                "learning_rate": 0.001,
                "beta1": 0.9,
            },
        },
        "some_value": 42,
    }

    overrides = {
        "device": "cuda",
        "trainer": {
            "batch_size": 256,  # Override existing
            "checkpoint": {  # Add new nested key
                "checkpoint_interval": 100
            },
        },
        "new_value": "test",  # Add completely new key
    }

    result = merge_train_job_config_overrides(base_cfg, overrides)

    # Check overridden values
    assert result["device"] == "cuda"
    assert result["trainer"]["batch_size"] == 256

    # Check preserved values
    assert result["some_value"] == 42
    assert result["trainer"]["total_timesteps"] == 50000000
    assert result["trainer"]["optimizer"]["learning_rate"] == 0.001
    assert result["trainer"]["optimizer"]["beta1"] == 0.9

    # Check new values
    assert result["new_value"] == "test"
    assert result["trainer"]["checkpoint"]["checkpoint_interval"] == 100


def test_validate_train_job_config():
    """Test validation of TrainerConfig."""
    # Create a valid trainer config
    trainer_cfg = TrainerConfig()

    # This should not raise an exception
    validated = validate_train_job_config(trainer_cfg)
    assert validated == trainer_cfg
    assert validated.total_timesteps > 0  # Should have default value
