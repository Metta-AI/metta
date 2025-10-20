"""Game management and discovery for CoGames."""

import importlib.util
import json
import sys
from pathlib import Path

import yaml

from mettagrid.config.mettagrid_config import MettaGridConfig

_SUPPORTED_MISSION_EXTENSIONS = [".yaml", ".yml", ".json", ".py"]


def load_mission_config_from_python(path: Path) -> MettaGridConfig:
    """Load a mission configuration from a Python file.

    The Python file should define a function called 'get_config()' that returns a MettaGridConfig.
    Alternatively, it can define a variable named 'config' that is a MettaGridConfig.

    Args:
        path: Path to the Python file

    Returns:
        The loaded mission configuration

    Raises:
        ValueError: If the Python file doesn't contain the required function or variable
    """
    # Load the Python module dynamically
    spec = importlib.util.spec_from_file_location("game_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to load Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["game_config"] = module
    spec.loader.exec_module(module)

    # Try to get config from get_config() function or config variable
    if hasattr(module, "get_config") and callable(module.get_config):
        config = module.get_config()
    elif hasattr(module, "config"):
        config = module.config
    else:
        raise ValueError(
            f"Python file {path} must define either a 'get_config()' function "
            "or a 'config' variable that returns/contains a MettaGridConfig"
        )

    if not isinstance(config, MettaGridConfig):
        raise ValueError(f"Python file {path} must return a MettaGridConfig instance")

    # Clean up the temporary module
    del sys.modules["game_config"]

    return config


def save_mission_config(config: MettaGridConfig, output_path: Path) -> None:
    """Save a mission configuration to file.

    Args:
        config: The mission configuration
        output_path: Path to save the configuration

    Raises:
        ValueError: If file extension is not supported
    """
    if output_path.suffix in [".yaml", ".yml"]:
        with open(output_path, "w") as f:
            yaml.dump(config.model_dump(mode="yaml"), f, default_flow_style=False, sort_keys=False)
    elif output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2)
    else:
        raise ValueError(
            f"Unsupported file format: {output_path.suffix}. Supported: {', '.join(_SUPPORTED_MISSION_EXTENSIONS)}"
        )


def load_mission_config(path: Path) -> MettaGridConfig:
    """Load a mission configuration from file.

    Args:
        path: Path to the configuration file

    Returns:
        The loaded game configuration

    Raises:
        ValueError: If file extension is not supported
    """
    if path.suffix in [".yaml", ".yml"]:
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            config_dict = json.load(f)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Supported: {', '.join(_SUPPORTED_MISSION_EXTENSIONS)}"
        )

    return MettaGridConfig(**config_dict)
