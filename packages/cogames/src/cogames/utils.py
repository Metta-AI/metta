"""Utility functions for CoGames CLI."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

from cogames import game as game_module
from cogames.policy import Policy, TrainablePolicy
from mettagrid.util.module import load_symbol

if TYPE_CHECKING:
    import torch
    from rich.console import Console

    from mettagrid.config.mettagrid_config import MettaGridConfig


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


def get_game_config(console: "Console", game_arg: str) -> tuple[str, "MettaGridConfig"]:
    """Return a resolved game name and configuration for cli usage."""
    import typer

    resolved_game, error = resolve_game(game_arg)
    if error or resolved_game is None:
        console.print(f"[red]Error: {error or 'Unknown game'}[/red]")
        raise typer.Exit(1) from ValueError(error)
    return resolved_game, game_module.get_game(resolved_game)


def resolve_training_device(console: "Console", requested: str) -> "torch.device":
    import torch

    normalized = requested.strip().lower()

    def cuda_usable() -> bool:
        cuda_backend = getattr(torch.backends, "cuda", None)
        if cuda_backend is None or not cuda_backend.is_built():
            return False
        if not hasattr(torch._C, "_cuda_getDeviceCount"):
            return False
        return torch.cuda.is_available()

    if normalized == "auto":
        if cuda_usable():
            return torch.device("cuda")
        console.print("[yellow]CUDA not available; falling back to CPU for training.[/yellow]")
        return torch.device("cpu")

    try:
        candidate = torch.device(requested)
    except (RuntimeError, ValueError):
        console.print(f"[yellow]Warning: Unknown device '{requested}'. Falling back to CPU.[/yellow]")
        return torch.device("cpu")

    if candidate.type == "cuda" and not cuda_usable():
        console.print("[yellow]CUDA requested but unavailable. Training will run on CPU instead.[/yellow]")
        return torch.device("cpu")

    return candidate


def _resolve_checkpoint_path(path: Path) -> Path:
    """Resolve a checkpoint path, descending into directories if needed."""
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidate_files = sorted((p for p in path.rglob("*.pt")), key=lambda target: target.stat().st_mtime)
    if not candidate_files:
        raise FileNotFoundError(f"No checkpoint files (*.pt) found in directory: {path}")
    return candidate_files[-1]


def instantiate_or_load_policy(
    policy_class_path: str, policy_data_path: Optional[str], env: Any, device: "torch.device | None" = None
) -> Policy:
    import torch

    policy_class = load_symbol(policy_class_path)
    policy = policy_class(env, device or torch.device("cpu"))

    if policy_data_path:
        resolved = _resolve_checkpoint_path(Path(policy_data_path))
        if not isinstance(policy, TrainablePolicy):
            raise TypeError("Policy data provided, but the selected policy does not support loading checkpoints.")

        policy.load_policy_data(str(resolved))
    return policy
