"""Utility functions for CoGames CLI."""

import contextlib
import logging
import signal
from pathlib import Path
from typing import Iterator, Optional, Tuple

from cogames import game as game_module

logger = logging.getLogger("cogames.utils")


class TimeoutExpired(RuntimeError):
    """Raised when a CLI command exceeds the configured timeout."""


@contextlib.contextmanager
def cli_timeout(timeout_seconds: Optional[int]) -> Iterator[None]:
    """Enforce a wall-clock timeout for CLI commands.

    Args:
        timeout_seconds: Timeout in seconds. If None or non-positive, no timeout is enforced.
    """

    if timeout_seconds is None or timeout_seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        logger.warning("Timeout option is not supported on this platform.")
        yield
        return

    def _handle_timeout(signum, frame):
        raise TimeoutExpired(f"Command timed out after {timeout_seconds} seconds.")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)

    if hasattr(signal, "setitimer"):
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    else:  # pragma: no cover - fallback path
        signal.alarm(int(timeout_seconds))

    try:
        yield
    finally:
        if hasattr(signal, "setitimer"):
            signal.setitimer(signal.ITIMER_REAL, 0)
        else:  # pragma: no cover - fallback path
            signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


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
