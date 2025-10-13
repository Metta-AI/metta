"""Helpers for resolving mission aliases and map lookups."""

from __future__ import annotations

from typing import Optional, Tuple

from cogames.cogs_vs_clips.missions import USER_MAP_CATALOG, UserMap
from mettagrid.config.mettagrid_config import MettaGridConfig

MAP_MISSION_DELIMITER = ":"
_SUFFIX_TO_VARIANT: tuple[tuple[str, str], ...] = (
    ("_easy_shaped", "easy_shaped"),
    ("_easy", "easy"),
    ("_shaped", "shaped"),
)


def get_user_map(map_name: str) -> Optional[UserMap]:
    for user_map in USER_MAP_CATALOG:
        if user_map.name == map_name:
            return user_map
    return None


def list_registered_missions() -> list[str]:
    missions: list[str] = []
    for user_map in USER_MAP_CATALOG:
        for mission in user_map.available_missions:
            if mission == user_map.default_mission:
                missions.append(user_map.name)
            else:
                missions.append(f"{user_map.name}{MAP_MISSION_DELIMITER}{mission}")
    return missions


def resolve_map_and_mission(identifier: str, mission_name: Optional[str] = None) -> tuple[str, Optional[str]]:
    map_name, mission = _split_identifier(identifier, mission_name)
    if get_user_map(map_name) is not None:
        return map_name, mission
    normalized_map, normalized_mission = _normalize_suffix(map_name, mission)
    return normalized_map, normalized_mission


def generate_env(map_name: str, mission_name: Optional[str]) -> Tuple[MettaGridConfig, str, str]:
    user_map = get_user_map(map_name)
    if user_map is None:
        available = ", ".join(user_map.name for user_map in USER_MAP_CATALOG)
        raise ValueError(f"Map '{map_name}' not found. Available: {available}")

    effective_mission = mission_name or user_map.default_mission
    if effective_mission not in user_map.available_missions:
        raise ValueError(
            f"Mission '{effective_mission}' not found for map '{map_name}'. "
            f"Available missions: {', '.join(user_map.available_missions)}"
        )
    config = user_map.generate_env(effective_mission)
    return config, user_map.name, effective_mission


def _split_identifier(identifier: str, mission_name: Optional[str]) -> tuple[str, Optional[str]]:
    if MAP_MISSION_DELIMITER in identifier:
        before, after = identifier.split(MAP_MISSION_DELIMITER, 1)
        if mission_name is not None:
            raise ValueError("Mission argument already includes ':' delimiter and mission_name parameter")
        if not before:
            raise ValueError("Mission identifier must include a map name before ':'")
        return before, after or None
    return identifier, mission_name


def _normalize_suffix(map_name: str, mission_name: Optional[str]) -> tuple[str, Optional[str]]:
    for suffix, variant in _SUFFIX_TO_VARIANT:
        if map_name.endswith(suffix):
            base = map_name[: -len(suffix)]
            if not base:
                break
            if mission_name is None:
                return base, variant
            return base, mission_name
    return map_name, mission_name


__all__ = [
    "MAP_MISSION_DELIMITER",
    "generate_env",
    "get_user_map",
    "list_registered_missions",
    "resolve_map_and_mission",
]
