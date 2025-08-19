import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from metta.common.util.fs import cd_repo_root


def get_test_basic_cfg() -> DictConfig:
    return get_cfg("test_basic")


def get_cfg(config_name: str) -> DictConfig:
    # Save current directory to restore later if needed
    original_cwd = Path.cwd()

    try:
        # Change to repository root
        cd_repo_root()

        # Now we can use a consistent path from repo root
        mettagrid_configs_root = Path("mettagrid/configs")

        config_path = mettagrid_configs_root / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        cfg = OmegaConf.load(config_path)
        assert isinstance(cfg, DictConfig), f"Config {config_name} did not load into a DictConfig"

        return cfg

    finally:
        # Optionally restore original directory
        os.chdir(original_cwd)
