"""Game management and discovery for CoGames."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import yaml
from rich.console import Console
from rich.table import Table

from cogames.cogs_vs_clips.scenarios import games as cogs_vs_clips_games
from cogames.scalable_astroid import make_scalable_arena
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

    all_games["scalable_astroid"] = make_scalable_arena()

    # Future games can be added here
    # for scenario_name, scenario_config in other_game_scenarios().items():
    #     all_games[scenario_name] = scenario_config

    return all_games


def load_game_config_from_python(path: Path) -> MettaGridConfig:
    """Load a game configuration from a Python file.

    The Python file should define a function called 'get_config()' that returns a MettaGridConfig.
    Alternatively, it can define a variable named 'config' that is a MettaGridConfig.

    Args:
        path: Path to the Python file

    Returns:
        The loaded game configuration

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


def get_game(game_name: str) -> MettaGridConfig:
    """Get a specific game configuration by name or file path.

    Args:
        game_name: Name of the game or path to config file (.yaml, .json, or .py)

    Returns:
        Game configuration

    Raises:
        ValueError: If game not found or file cannot be loaded
    """
    # Check if it's a file path
    if any(game_name.endswith(ext) for ext in [".yaml", ".yml", ".json", ".py"]):
        path = Path(game_name)
        if not path.exists():
            raise ValueError(f"File not found: {game_name}")
        if not path.is_file():
            raise ValueError(f"Not a file: {game_name}")

        # Load config based on file extension
        if path.suffix == ".py":
            return load_game_config_from_python(path)
        elif path.suffix in [".yaml", ".yml", ".json"]:
            return load_game_config(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    # Otherwise, treat it as a game name
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


def _convert_tuples_to_lists(obj):
    """Recursively convert tuples to lists in a nested data structure."""
    if isinstance(obj, dict):
        return {k: _convert_tuples_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_tuples_to_lists(item) for item in obj]
    else:
        return obj


def save_game_config(config: MettaGridConfig, output_path: Path) -> None:
    """Save a game configuration to file.

    Args:
        config: The game configuration
        output_path: Path to save the configuration

    Raises:
        ValueError: If file extension is not supported
    """
    config_dict = config.model_dump()

    # Convert tuples to lists for better serialization compatibility
    config_dict = _convert_tuples_to_lists(config_dict)

    if output_path.suffix in [".yaml", ".yml"]:
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
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
