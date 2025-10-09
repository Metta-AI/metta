"""Game management and discovery for CoGames."""

import importlib.util
import json
import sys
from pathlib import Path

import yaml

from cogames.mission_aliases import (
    MAP_MISSION_DELIMITER,
    generate_env,
    list_registered_missions,
    resolve_map_and_mission,
)
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


def get_all_missions() -> list[str]:
    """Return the list of registered missions, including aliases."""
    return list_registered_missions()


def get_mission(map_identifier: str, mission_name: str | None = None) -> tuple[MettaGridConfig, str | None, str | None]:
    """Resolve a mission by name, alias, or file path."""
    if any(map_identifier.endswith(ext) for ext in _SUPPORTED_MISSION_EXTENSIONS):
        path = Path(map_identifier)
        if not path.exists() or not path.is_file():
            raise ValueError(f"File not found: {map_identifier}")
        if path.suffix == ".py":
            return load_mission_config_from_python(path), None, None
        if path.suffix in [".yaml", ".yml", ".json"]:
            return load_mission_config(path), None, None
        raise ValueError(f"Unsupported file format: {path.suffix}")

    normalized_map, normalized_mission = resolve_map_and_mission(map_identifier, mission_name)
    config, resolved_map, resolved_mission = generate_env(normalized_map, normalized_mission)
    return config, resolved_map, resolved_mission


__all__ = [
    "MAP_MISSION_DELIMITER",
    "get_all_missions",
    "get_mission",
    "load_mission_config",
    "load_mission_config_from_python",
    "save_mission_config",
]
