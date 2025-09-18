"""Utility functions for CoGames CLI."""

from typing import Optional, Tuple

from cogames import game as game_module


def resolve_game(game_arg: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a game name from user input.

    Args:
        game_arg: User input for game name

    Returns:
        Tuple of (resolved_game_name, error_message)
    """
    if not game_arg:
        return None, None

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
