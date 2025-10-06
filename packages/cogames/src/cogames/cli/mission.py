from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from cogames.cli.base import console
from cogames.cogs_vs_clips.missions import USER_MAP_CATALOG
from cogames.game import load_mission_config, load_mission_config_from_python
from mettagrid import MettaGridConfig
from mettagrid.config.mettagrid_config import AssemblerConfig

MAP_MISSION_DELIMITER = ":"


def get_mission_name_and_config(ctx: typer.Context, mission_arg: Optional[str]) -> tuple[str, MettaGridConfig]:
    if mission_arg is None:
        console.print(ctx.get_help())
    else:
        try:
            return get_mission(mission_arg)
        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]")
            console.print()
    list_missions()

    if mission_arg is not None:
        console.print()
        console.print(ctx.get_usage())
    console.print()
    raise typer.Exit(0)


def get_mission(
    mission_arg: str,
) -> tuple[str, MettaGridConfig]:
    """Get a specific mission configuration by name or file path.

    Args:
        map_name: Name of the map or path to config file (.yaml, .json, or .py)
        mission_name: Name of the mission. If unspecified, will use the default mission for the map.

    Returns:
        Environment configuration, map name, mission name

    Raises:
        ValueError: If mission not found or file cannot be loaded
    """
    # Check if it's a file path
    if any(mission_arg.endswith(ext) for ext in [".yaml", ".yml", ".json", ".py"]):
        path = Path(mission_arg)
        if not path.exists():
            raise ValueError(f"File not found: {mission_arg}")
        if not path.is_file():
            raise ValueError(f"Not a file: {mission_arg}")

        # Load config based on file extension
        if path.suffix == ".py":
            return mission_arg, load_mission_config_from_python(path)
        elif path.suffix in [".yaml", ".yml", ".json"]:
            return mission_arg, load_mission_config(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    # Otherwise, treat it as a mission name
    if (delim_count := mission_arg.count(MAP_MISSION_DELIMITER)) == 0:
        map_name, mission_name = mission_arg, None
    elif delim_count > 1:
        raise ValueError(f"Mission name can contain at most one `{MAP_MISSION_DELIMITER}` delimiter")
    else:
        map_name, mission_name = mission_arg.split(MAP_MISSION_DELIMITER)
    matching_maps = [user_map for user_map in USER_MAP_CATALOG if user_map.name == map_name]
    if not matching_maps:
        raise ValueError(f"Could not find map {map_name}")
    elif len(matching_maps) > 1:
        raise ValueError(f"Invalid map catalog: more than one map named {map_name}")
    matching_map = matching_maps[0]
    effective_mission = mission_name if mission_name is not None else matching_map.default_mission
    if effective_mission not in matching_map.available_missions:
        raise ValueError(f"Mission {effective_mission} not available on map {map_name}")
    return f"{matching_map.name}{MAP_MISSION_DELIMITER}{effective_mission}", matching_map.generate_env(
        effective_mission
    )


def list_missions() -> None:
    """Print a table listing all available missions."""

    if not USER_MAP_CATALOG:
        console.print("No maps found")
        return

    table = Table(title="Available Missions", show_header=True, header_style="bold magenta")
    table.add_column("Mission", style="cyan", no_wrap=True)
    table.add_column("Agents", style="yellow", justify="center")
    table.add_column("Map Size", style="green", justify="center")

    for user_map in USER_MAP_CATALOG:
        for mission_name in user_map.available_missions:
            game_config = user_map.generate_env(mission_name)
            num_agents = game_config.game.num_agents

            # Try to get map size if available
            map_builder = game_config.game.map_builder
            if hasattr(map_builder, "width") and hasattr(map_builder, "height"):
                map_size = f"{map_builder.width}x{map_builder.height}"  # type: ignore[attr-defined]
            else:
                map_size = "N/A"

            if mission_name == user_map.default_mission:
                table.add_row(user_map.name, str(num_agents), map_size)
            else:
                table.add_row(
                    f"{user_map.name}[gray]{MAP_MISSION_DELIMITER}[/gray][cyan]{mission_name}[/cyan]",
                    str(num_agents),
                    map_size,
                )
    console.print(table)
    console.print()
    console.print("To specify a [[bold cyan]MISSION[/bold cyan]], you can:")
    console.print("  • Use a mission name from above")
    console.print("  • Use a path to a mission configuration file, e.g. path/to/mission.yaml")


def describe_mission(mission_name: str, game_config: MettaGridConfig) -> None:
    """Print detailed information about a specific mission.

    Args:
        mission_name: Name of the mission
        env_cfg: Environment configuration
    """

    console.print(f"\n[bold cyan]{mission_name}[/bold cyan]\n")

    # Display mission configuration
    console.print("[bold]Mission Configuration:[/bold]")
    console.print(f"  • Number of agents: {game_config.game.num_agents}")
    console.print(f"  • Map size: {game_config.game.map_builder.width}x{game_config.game.map_builder.height}")  # type: ignore[attr-defined]

    # Display available actions
    console.print("\n[bold]Available Actions:[/bold]")
    for n, a in game_config.game.actions.model_dump().items():
        if a["enabled"]:
            console.print(f"  • {n}: {a['consumed_resources']}")

    # Display objects
    console.print("\n[bold]Stations:[/bold]")
    for obj_name, obj_config in game_config.game.objects.items():
        console.print(f"  • {obj_name}")
        if isinstance(obj_config, AssemblerConfig):
            for _, recipe in obj_config.recipes:
                if recipe.input_resources:
                    inputs = ", ".join(f"{k}:{v}" for k, v in recipe.input_resources.items())
                    outputs = ", ".join(f"{k}:{v}" for k, v in recipe.output_resources.items())
                    console.print(f"    {inputs} → {outputs} (cooldown: {recipe.cooldown})")

    # Display agent configuration
    console.print("\n[bold]Agent Configuration:[/bold]")
    console.print(f"  • Default resource limit: {game_config.game.agent.default_resource_limit}")
    if game_config.game.agent.resource_limits:
        console.print(f"  • Resource limits: {game_config.game.agent.resource_limits}")
