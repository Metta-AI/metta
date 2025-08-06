import os
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from tools.sweep_config_utils import (
    load_train_job_config_with_overrides,
    save_train_job_override_config,
)


def test_load_train_job_config_with_overrides():
    """Test loading train job config with overrides from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a base config without trainer to avoid validation
        base_cfg = OmegaConf.create(
            {
                "run_dir": tmpdir,
                "device": "cpu",
                "some_value": 42,
            }
        )

        # Create an override file
        override_path = Path(tmpdir) / "train_config_overrides.yaml"
        overrides = {
            "device": "cuda",
            "new_value": "test",
        }
        OmegaConf.save(overrides, override_path)

        # Apply overrides
        result_cfg = load_train_job_config_with_overrides(base_cfg)

        # Check that overrides were applied
        assert result_cfg.device == "cuda"
        assert result_cfg.some_value == 42  # Original value preserved
        assert result_cfg.new_value == "test"


def test_load_train_job_config_with_overrides_no_file():
    """Test that config is unchanged when no override file exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_cfg = OmegaConf.create(
            {
                "run_dir": tmpdir,
                "device": "cpu",
                "some_value": 99,
            }
        )

        # No override file exists
        result_cfg = load_train_job_config_with_overrides(base_cfg)

        # Config should be unchanged
        assert result_cfg.device == "cpu"
        assert result_cfg.some_value == 99


def test_save_train_job_override_config():
    """Test saving train job override config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create configs
        cfg = OmegaConf.create(
            {
                "run_dir": tmpdir,
            }
        )

        train_cfg = OmegaConf.create(
            {
                "trainer": {
                    "gamma": 0.95,
                    "batch_size": 256,
                }
            }
        )

        # Save overrides
        save_path = save_train_job_override_config(cfg, train_cfg)

        # Check file was created
        assert save_path == os.path.join(tmpdir, "train_config_overrides.yaml")
        assert os.path.exists(save_path)

        # Load and verify contents
        loaded = OmegaConf.load(save_path)
        assert loaded.trainer.gamma == 0.95
        assert loaded.trainer.batch_size == 256
