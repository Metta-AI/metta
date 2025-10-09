"""Backward-compatible utilities for the CoGames CLI."""

from __future__ import annotations

from typing import Optional, Tuple

import typer
from rich.console import Console

from cogames.cogs_vs_clips.missions import UserMap
from cogames.device import resolve_training_device as _resolve_training_device
from cogames.game import get_mission
from cogames.mission_aliases import MAP_MISSION_DELIMITER, get_user_map
from cogames.policy.utils import initialize_or_load_policy as _initialize_or_load_policy
from mettagrid.config.mettagrid_config import MettaGridConfig


def get_mission_config(
    console: Console,
    mission_arg: str,
) -> Tuple[str, MettaGridConfig, Optional[UserMap]]:
    """Resolve mission arguments while retaining backward-compatible metadata."""
    try:
        config, map_name, mission_name = get_mission(mission_arg)
    except ValueError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        raise typer.Exit(1) from exc

    resolved_name = _format_resolved_name(mission_arg, map_name, mission_name)
    user_map = get_user_map(map_name) if map_name is not None else None
    return resolved_name, config, user_map


resolve_training_device = _resolve_training_device
initialize_or_load_policy = _initialize_or_load_policy

__all__ = ["get_mission_config", "resolve_training_device", "initialize_or_load_policy"]


def _format_resolved_name(
    requested: str,
    map_name: Optional[str],
    mission_name: Optional[str],
) -> str:
    if map_name is None:
        return requested
    if mission_name and mission_name != "default":
        return f"{map_name}{MAP_MISSION_DELIMITER}{mission_name}"
    return map_name
