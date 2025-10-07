"""Game management and discovery for CoGames."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

from cogames.cogs_vs_clips.missions import USER_MAP_CATALOG
from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig

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


def get_all_missions() -> list[str]:
    """Get all available missions."""
    return [
        f"{user_map.name}:{mission}" if mission != user_map.default_mission else user_map.name
        for user_map in USER_MAP_CATALOG
        for mission in user_map.available_missions
    ]


def get_mission(
    map_name: str, mission_name: Optional[str] = None
) -> tuple[MettaGridConfig, Optional[str], Optional[str]]:
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
    if any(map_name.endswith(ext) for ext in [".yaml", ".yml", ".json", ".py"]):
        path = Path(map_name)
        if not path.exists():
            raise ValueError(f"File not found: {map_name}")
        if not path.is_file():
            raise ValueError(f"Not a file: {map_name}")

        # Load config based on file extension
        if path.suffix == ".py":
            return load_mission_config_from_python(path), None, None
        elif path.suffix in [".yaml", ".yml", ".json"]:
            return load_mission_config(path), None, None
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    # Otherwise, treat it as a mission name
    matching_maps = [um for um in USER_MAP_CATALOG if um.name == map_name]
    if not matching_maps:
        raise ValueError(
            f"Map '{map_name}' not found. Available maps: {', '.join(user_map.name for user_map in USER_MAP_CATALOG)}"
        )
    user_map = matching_maps[0]
    effective_mission = mission_name or user_map.default_mission
    return user_map.generate_env(effective_mission), user_map.name, effective_mission


def list_missions(console: Console) -> None:
    """Create a table listing all available missions.

    Args:
        console: Rich console for rendering

    Returns:
        Rich Table with mission information
    """

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
                table.add_row(f"{user_map.name}[gray]:[/gray][cyan]{mission_name}[/cyan]", str(num_agents), map_size)
    console.print(table)
    console.print()
    console.print("To specify a <[bold cyan]mission[/bold cyan]>, you can:")
    console.print("  • Use a mission name from above")
    console.print("  • Use a path to a mission configuration file, e.g. path/to/mission.yaml")


def describe_mission(mission_name: str, game_config: MettaGridConfig, console: Console) -> None:
    """Print detailed information about a specific mission.

    Args:
        mission_name: Name of the mission
        env_cfg: Environment configuration
        console: Rich console for output
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


def require_mission_argument(ctx: typer.Context, value: Optional[str], console: Console) -> str:
    if value is not None:
        return value

    list_missions(console)
    console.print(f"\n[dim]Usage: {ctx.command_path} <mission>[/dim]")
    console.print()
    raise typer.Exit(0)
