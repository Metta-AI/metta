"""Game management and discovery for CoGames."""

import json
from pathlib import Path
from typing import Dict, Optional

import yaml
from rich.console import Console
from rich.table import Table

from cogames.cogs_vs_clips.scenarios import games as cogs_vs_clips_games
from mettagrid.config.mettagrid_config import AssemblerConfig, MettaGridConfig


def get_all_games() -> Dict[str, MettaGridConfig]:
    """Get all available games (scenarios).

    Returns:
        Dictionary of game names to their configurations
    """
    # Flatten all scenarios into a single dictionary
    all_games = {}

    # Add cogs_vs_clips games
    for game_name, game_config in cogs_vs_clips_games().items():
        all_games[game_name] = game_config

    # Future games can be added here
    # for scenario_name, scenario_config in other_game_scenarios().items():
    #     all_games[scenario_name] = scenario_config

    return all_games


def get_game(game_name: str) -> MettaGridConfig:
    """Get a specific game configuration by name.

    Args:
        game_name: Name of the game

    Returns:
        Game configuration

    Raises:
        ValueError: If game not found
    """
    all_games = get_all_games()
    if game_name not in all_games:
        # Try partial match
        matches = [name for name in all_games if game_name.lower() in name.lower()]
        if len(matches) == 1:
            return all_games[matches[0]]
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous game name '{game_name}'. Matches: {', '.join(matches)}")
        else:
            raise ValueError(f"Game '{game_name}' not found. Available games: {', '.join(all_games.keys())}")
    return all_games[game_name]


def resolve_game_name(game_arg: Optional[str]) -> Optional[str]:
    """Resolve a game name from user input.

    Args:
        game_arg: User input for game name

    Returns:
        Resolved game name or None if not found
    """
    if not game_arg:
        return None

    all_games = get_all_games()

    # Exact match
    if game_arg in all_games:
        return game_arg

    # Case-insensitive match
    lower_map = {name.lower(): name for name in all_games}
    if game_arg.lower() in lower_map:
        return lower_map[game_arg.lower()]

    # Partial match
    matches = [name for name in all_games if game_arg.lower() in name.lower()]
    if len(matches) == 1:
        return matches[0]

    return None


def list_games(console: Console) -> Table:
    """Create a table listing all available games.

    Args:
        console: Rich console for rendering

    Returns:
        Rich Table with game information
    """

    all_games = get_all_games()

    table = Table(title="Available Games", show_header=True, header_style="bold magenta")
    table.add_column("Game", style="cyan", no_wrap=True)
    table.add_column("Agents", style="yellow", justify="center")
    table.add_column("Map Size", style="green", justify="center")

    for game_name, game_config in all_games.items():
        num_agents = game_config.game.num_agents
        map_size = f"{game_config.game.map_builder.width}x{game_config.game.map_builder.height}"

        table.add_row(game_name, str(num_agents), map_size)

    return table


def describe_game(game_name: str, console: Console) -> None:
    """Print detailed information about a specific game.

    Args:
        game_name: Name of the game
        console: Rich console for output
    """

    try:
        game_config = get_game(game_name)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    console.print(f"\n[bold cyan]{game_name}[/bold cyan]\n")

    # Display game configuration
    console.print("[bold]Game Configuration:[/bold]")
    console.print(f"  • Number of agents: {game_config.game.num_agents}")
    console.print(f"  • Map size: {game_config.game.map_builder.width}x{game_config.game.map_builder.height}")
    console.print(f"  • Number of agents on map: {game_config.game.map_builder.agents}")

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


def save_game_config(config: MettaGridConfig, output_path: Path) -> None:
    """Save a game configuration to file.

    Args:
        config: The game configuration
        output_path: Path to save the configuration

    Raises:
        ValueError: If file extension is not supported
    """
    config_dict = config.model_dump()

    if output_path.suffix in [".yaml", ".yml"]:
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}. Use .yaml, .yml, or .json")


def load_game_config(path: Path) -> MettaGridConfig:
    """Load a game configuration from file.

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
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json")

    return MettaGridConfig(**config_dict)
