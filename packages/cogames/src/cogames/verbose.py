"""Utilities for verbose output and config printing."""

import json

from rich.console import Console

from cogames.cogs_vs_clips.mission import Mission
from mettagrid.config.mettagrid_config import MettaGridConfig


def print_configs(
    console: Console,
    env_cfg: MettaGridConfig,
    mission_cfg: Mission | None = None,
    print_cvc_config: bool = False,
    print_mg_config: bool = False,
) -> None:
    """Print mission and/or mettagrid configurations.

    Args:
        console: Rich console for output
        env_cfg: MettaGrid environment configuration
        mission_cfg: Optional mission configuration (CVC config)
        print_cvc_config: Whether to print Mission config
        print_mg_config: Whether to print MettaGridConfig

    Raises:
        Exception: If there's an error printing the configs
    """
    if not (print_cvc_config or print_mg_config):
        return

    if print_cvc_config:
        console.print("[bold cyan]Mission (CVC Config):[/bold cyan]")
        if mission_cfg:
            cvc_data = mission_cfg.model_dump(mode="json")
            # Replace map_data with simplified ascii_map representation
            if "map" in cvc_data and cvc_data["map"] and "map_data" in cvc_data["map"]:
                # Try to get width/height from the actual config object if available
                try:
                    width = mission_cfg.map.width
                    height = mission_cfg.map.height
                except (AttributeError, KeyError):
                    width = cvc_data["map"].get("width", "?")
                    height = cvc_data["map"].get("height", "?")
                cvc_data["map"]["map_data"] = f"ascii_map({width}, {height})"
            console.print(json.dumps(cvc_data, indent=2))
        else:
            console.print("[yellow]Mission config not available for this mission type[/yellow]")
        if print_mg_config:
            console.print()

    if print_mg_config:
        console.print("[bold cyan]MettaGridConfig:[/bold cyan]")
        mg_data = env_cfg.model_dump(mode="json")
        # Replace map_data with simplified ascii_map representation
        if "game" in mg_data and "map_builder" in mg_data["game"] and "map_data" in mg_data["game"]["map_builder"]:
            # Try to get width/height from the actual config object if available
            try:
                width = env_cfg.game.map_builder.width
                height = env_cfg.game.map_builder.height
            except (AttributeError, KeyError):
                width = mg_data["game"]["map_builder"].get("width", "?")
                height = mg_data["game"]["map_builder"].get("height", "?")
            mg_data["game"]["map_builder"]["map_data"] = f"ascii_map({width}, {height})"
        console.print(json.dumps(mg_data, indent=2))
    console.print()  # Add separator line after configs
