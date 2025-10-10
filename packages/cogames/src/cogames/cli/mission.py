import re
from typing import Optional

import typer
from rich.table import Table

from cogames import game
from cogames.cli.base import console
from cogames.cogs_vs_clips.missions import USER_MAP_CATALOG
from cogames.mission_aliases import MAP_MISSION_DELIMITER, list_registered_missions
from mettagrid import MettaGridConfig
from mettagrid.config.mettagrid_config import AssemblerConfig


def get_all_missions() -> list[str]:
    return list_registered_missions()


def get_mission_name_and_config(ctx: typer.Context, mission_arg: Optional[str]) -> tuple[str, MettaGridConfig]:
    if not mission_arg:
        console.print(ctx.get_help())
        console.print("[yellow]Missing: --mission / -m[/yellow]\n")
    else:
        try:
            return get_mission(mission_arg)
        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]\n")
    list_missions()

    if mission_arg is not None:
        console.print("\n" + ctx.get_usage())
    console.print("\n")
    raise typer.Exit(0)


def get_mission_names_and_configs(
    ctx: typer.Context, missions_arg: Optional[list[str]]
) -> list[tuple[str, MettaGridConfig]]:
    if not missions_arg:
        console.print(ctx.get_help())
        console.print("[yellow]Supply at least one: --mission / -m[/yellow]\n")
    else:
        try:
            not_deduped = [mission for missions in missions_arg for mission in get_missions(missions)]
            name_set: set[str] = set()
            deduped = []
            for m, c in not_deduped:
                if m not in name_set:
                    name_set.add(m)
                    deduped.append((m, c))
            return deduped
        except ValueError as e:
            console.print(f"[yellow]{e}[/yellow]\n")
    list_missions()

    if missions_arg is not None:
        console.print("\n" + ctx.get_usage())
    console.print("\n")
    raise typer.Exit(0)


def get_missions(mission_arg: str) -> list[tuple[str, MettaGridConfig]]:
    if "*" in mission_arg:
        # Convert shell-style wildcard to regex pattern
        regex_pattern = mission_arg.replace(".", "\\.").replace("*", ".*")
        missions = [m for m in get_all_missions() if re.search(regex_pattern, m)]
        return [get_mission(m) for m in missions]
    return [get_mission(mission_arg)]


def get_mission(mission_arg: str) -> tuple[str, MettaGridConfig]:
    """Resolve a mission argument into a canonical mission name and configuration."""
    config, resolved_map, resolved_mission = game.get_mission(mission_arg)
    if resolved_map is None:
        return mission_arg, config
    if resolved_mission in (None, "default"):
        return resolved_map, config
    return f"{resolved_map}{MAP_MISSION_DELIMITER}{resolved_mission}", config


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
    console.print("\n")
    console.print("To specify a [bold cyan] -m [MISSION][/bold cyan], you can:")
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
