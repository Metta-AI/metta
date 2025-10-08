"""Utility functions for CoGames CLI."""

from typing import Any, Optional

import torch
import typer
from rich.console import Console

from cogames import game as game_module
from cogames.cogs_vs_clips.missions import UserMap
from cogames.policy import Policy, TrainablePolicy
from mettagrid.config.mettagrid_config import MettaGridConfig
from mettagrid.util.module import load_symbol


def get_mission_config(console: Console, mission_arg: str) -> tuple[str, MettaGridConfig, Optional[UserMap]]:
    """Return a resolved mission name, configuration, and matching UserMap (if registered)."""

    requested_mission: Optional[str] = None
    if ":" in mission_arg:
        map_name, requested_mission = mission_arg.split(":")
    else:
        map_name = mission_arg

    config, registered_map_name, mission_name = game_module.get_mission(map_name, requested_mission)
    user_map: Optional[UserMap] = None
    if registered_map_name is not None:
        user_map = game_module.get_user_map(registered_map_name)
    try:
        if registered_map_name and mission_name and mission_name != "default":
            full_mission_name = f"{registered_map_name}:{mission_name}"
        else:
            full_mission_name = registered_map_name or map_name
        return full_mission_name, config, user_map
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


def resolve_training_device(console: Console, requested: str) -> torch.device:
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


def initialize_or_load_policy(
    policy_class_path: str, policy_data_path: Optional[str], env: Any, device: "torch.device | None" = None
) -> Policy:
    policy_class = load_symbol(policy_class_path)
    policy = policy_class(env, device or torch.device("cpu"))

    if policy_data_path:
        if not isinstance(policy, TrainablePolicy):
            raise TypeError("Policy data provided, but the selected policy does not support loading checkpoints.")

        policy.load_policy_data(policy_data_path)
    return policy
