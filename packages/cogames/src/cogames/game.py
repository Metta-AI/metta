"""Game management and discovery for CoGames."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import yaml

from cogames.cogs_vs_clips.missions import USER_MAP_CATALOG, UserMap
from mettagrid.config.mettagrid_config import MettaGridConfig

SUPPORTED_MISSION_EXTENSIONS = (".yaml", ".yml", ".json", ".py")
MAP_MISSION_DELIMITER = ":"
_SUFFIX_TO_VARIANT = (
    ("_easy_shaped", "easy_shaped"),
    ("_easy", "easy"),
    ("_shaped", "shaped"),
)


def load_mission_config_from_python(path: Path) -> MettaGridConfig:
    """Load a mission configuration from a Python file."""
    spec = importlib.util.spec_from_file_location("game_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to load Python module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["game_config"] = module
    spec.loader.exec_module(module)

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

    del sys.modules["game_config"]
    return config


def save_mission_config(config: MettaGridConfig, output_path: Path) -> None:
    """Save a mission configuration to YAML or JSON."""
    if output_path.suffix in [".yaml", ".yml"]:
        with open(output_path, "w") as f:
            yaml.dump(config.model_dump(mode="yaml"), f, default_flow_style=False, sort_keys=False)
    elif output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2)
    else:
        raise ValueError(
            f"Unsupported file format: {output_path.suffix}. Supported: {', '.join(SUPPORTED_MISSION_EXTENSIONS)}"
        )


def load_mission_config(path: Path) -> MettaGridConfig:
    """Load a mission configuration from YAML or JSON."""
    if path.suffix in [".yaml", ".yml"]:
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
    elif path.suffix == ".json":
        with open(path, "r") as f:
            config_dict = json.load(f)
    else:
        raise ValueError(
            f"Unsupported file format: {path.suffix}. Supported: {', '.join(SUPPORTED_MISSION_EXTENSIONS)}"
        )

    return MettaGridConfig(**config_dict)


def get_all_missions() -> list[str]:
    """Return the full set of registered missions."""
    missions: list[str] = []
    for user_map in USER_MAP_CATALOG:
        for mission in user_map.available_missions:
            if mission == user_map.default_mission:
                missions.append(user_map.name)
            else:
                missions.append(f"{user_map.name}{MAP_MISSION_DELIMITER}{mission}")
    return missions


def get_user_map(map_name: str) -> Optional[UserMap]:
    """Return the registered user map for a given name, if present."""
    for user_map in USER_MAP_CATALOG:
        if user_map.name == map_name:
            return user_map
    return None


def get_mission(
    map_identifier: str,
    mission_name: Optional[str] = None,
) -> tuple[MettaGridConfig, Optional[str], Optional[str]]:
    """Resolve a mission by file, map name, or alias."""
    if _looks_like_supported_file(map_identifier):
        path = Path(map_identifier)
        if not path.exists():
            raise ValueError(f"File not found: {map_identifier}")
        if not path.is_file():
            raise ValueError(f"Not a file: {map_identifier}")
        if path.suffix == ".py":
            return load_mission_config_from_python(path), None, None
        return load_mission_config(path), None, None

    candidate_map = map_identifier
    candidate_mission = mission_name

    if candidate_mission is None and MAP_MISSION_DELIMITER in candidate_map:
        candidate_map, candidate_mission = candidate_map.split(MAP_MISSION_DELIMITER, 1)
        if candidate_mission == "":
            candidate_mission = None

    user_map = get_user_map(candidate_map)
    if user_map is None:
        suffix_match = _match_suffix_alias(candidate_map)
        if suffix_match is not None and candidate_mission is None:
            base_name, derived_mission = suffix_match
            return get_mission(base_name, derived_mission)
        raise ValueError(
            f"Map '{candidate_map}' not found. "
            f"Available maps: {', '.join(user_map.name for user_map in USER_MAP_CATALOG)}"
        )

    effective_mission = candidate_mission or user_map.default_mission
    if effective_mission not in user_map.available_missions:
        raise ValueError(
            f"Mission '{effective_mission}' not found for map '{user_map.name}'. "
            f"Available missions: {', '.join(user_map.available_missions)}"
        )

    return user_map.generate_env(effective_mission), user_map.name, effective_mission


def _looks_like_supported_file(identifier: str) -> bool:
    return any(identifier.endswith(ext) for ext in SUPPORTED_MISSION_EXTENSIONS)


def _match_suffix_alias(map_name: str) -> Optional[tuple[str, str]]:
    for suffix, variant in _SUFFIX_TO_VARIANT:
        if map_name.endswith(suffix):
            return map_name[: -len(suffix)], variant
    return None
