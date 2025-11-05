"""Utility functions for mission configuration."""

from __future__ import annotations

from pathlib import Path
from types import MethodType
from typing import Callable, List

from cogames.cogs_vs_clips.mission import Mission
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.map_builder.map_builder import MapBuilderConfig


def get_map(map_name: str) -> MapBuilderConfig:
    """Load a map by name from the maps directory."""
    maps_dir = Path(__file__).parent.parent / "maps"
    map_path = maps_dir / map_name
    return MapBuilderConfig.from_uri(str(map_path))


def _add_make_env_modifier(mission: Mission, modifier: Callable[[MettaGridConfig], None]) -> Mission:
    """Add a modifier function to a mission's make_env chain."""
    modifiers: List[Callable[[MettaGridConfig], None]] = getattr(mission, "__env_modifiers__", None)

    if modifiers is None:
        original_make_env = mission.make_env.__func__
        original_instantiate = mission.instantiate.__func__

        def wrapped_make_env(self, *args, **kwargs):
            cfg = original_make_env(self, *args, **kwargs)
            for fn in getattr(self, "__env_modifiers__", []):
                fn(cfg)
            return cfg

        def wrapped_instantiate(self, *args, **kwargs):
            instantiated = original_instantiate(self, *args, **kwargs)
            parent_mods = getattr(self, "__env_modifiers__", [])
            if parent_mods:
                object.__setattr__(instantiated, "__env_modifiers__", list(parent_mods))
                object.__setattr__(instantiated, "make_env", MethodType(wrapped_make_env, instantiated))
                object.__setattr__(instantiated, "instantiate", MethodType(wrapped_instantiate, instantiated))
            return instantiated

        object.__setattr__(mission, "__env_modifiers__", [])
        object.__setattr__(mission, "make_env", MethodType(wrapped_make_env, mission))
        object.__setattr__(mission, "instantiate", MethodType(wrapped_instantiate, mission))
        modifiers = mission.__env_modifiers__

    modifiers.append(modifier)
    return mission
