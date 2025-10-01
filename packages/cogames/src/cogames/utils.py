"""Utility functions for CoGames CLI."""

from pathlib import Path
from typing import Optional, Tuple

from cogames import game as game_module


def resolve_game(game_arg: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a game name from user input.

    Args:
        game_arg: User input for game name or path to config file

    Returns:
        Tuple of (resolved_game_name, error_message)
    """
    if not game_arg:
        return None, None

    # Check if it's a file path
    if any(game_arg.endswith(ext) for ext in [".yaml", ".yml", ".json", ".py"]):
        path = Path(game_arg)
        if not path.exists():
            return None, f"File not found: {game_arg}"
        if not path.is_file():
            return None, f"Not a file: {game_arg}"
        # Return the path as the resolved name
        return game_arg, None

    # Otherwise, treat it as a game name
    resolved = game_module.resolve_game_name(game_arg)
    if resolved:
        return resolved, None

    # Check for partial matches
    all_games = game_module.get_all_games()
    matches = [name for name in all_games if game_arg.lower() in name.lower()]

    if len(matches) > 1:
        return None, f"Ambiguous game name '{game_arg}'. Matches: {', '.join(matches)}"
    else:
        return None, f"Game '{game_arg}' not found. Use 'cogames games' to list available games."
