"""Utility functions for CoGames CLI."""

import contextlib
import logging
import os
import re
import signal
from datetime import date
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Tuple

from cogames import game as game_module
from mettagrid import MettaGridConfig

logger = logging.getLogger("cogames.utils")

CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".py"}


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

    sigalrm = getattr(signal, "SIGALRM", None)
    if sigalrm is None:
        logger.warning("Timeout option is not supported on this platform.")
        yield
        return

    def _handle_timeout(signum, frame):
        raise TimeoutExpired(f"Command timed out after {timeout_seconds} seconds.")

    previous_handler = signal.getsignal(sigalrm)
    signal.signal(sigalrm, _handle_timeout)

    setitimer = getattr(signal, "setitimer", None)
    if setitimer is not None:
        setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    else:  # pragma: no cover - fallback path
        signal.alarm(int(timeout_seconds))

    try:
        yield
    finally:
        if setitimer is not None:
            setitimer(signal.ITIMER_REAL, 0)
        else:  # pragma: no cover - fallback path
            signal.alarm(0)
        signal.signal(sigalrm, previous_handler)


def resolve_game(game_arg: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a game name from user input.

    Args:
        game_arg: User input for game name or path to config file

    Returns:
        Tuple of (resolved_game_name, error_message)
    """
    if not game_arg:
        return None, None

    candidate = Path(game_arg)
    if candidate.suffix in CONFIG_EXTENSIONS:
        if not candidate.exists():
            return None, f"File not found: {game_arg}"
        if not candidate.is_file():
            return None, f"Not a file: {game_arg}"
        return game_arg, None

    resolved = game_module.resolve_game_name(game_arg)
    if resolved:
        return resolved, None

    game_names = game_module.get_all_games().keys()
    matches = [name for name in game_names if game_arg.lower() in name.lower()]

    if len(matches) > 1:
        return None, f"Ambiguous game name '{game_arg}'. Matches: {', '.join(matches)}"

    return None, "Game '{game}' not found. Use 'cogames games' to list available games.".format(game=game_arg)


def ensure_config(value: Any) -> MettaGridConfig:
    if isinstance(value, MettaGridConfig):
        return value
    if isinstance(value, (str, Path)):
        return game_module.get_game(str(value))
    if isinstance(value, dict):
        return MettaGridConfig.model_validate(value)
    raise ValueError(f"Unsupported curriculum item type: {type(value)!r}.")


def load_curriculum_items(source: Any, max_items: int) -> list[MettaGridConfig]:
    queue: list[Any] = [source]
    configs: list[MettaGridConfig] = []

    while queue and len(configs) < max_items:
        item = queue.pop(0)
        if item is None:
            continue
        if isinstance(item, (MettaGridConfig, str, Path, dict)):
            configs.append(ensure_config(item))
            continue
        if callable(item):
            queue.append(item())
            continue
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            queue.extend(item)
            continue
        raise ValueError(f"Curriculum source produced unsupported type: {type(item)!r}.")

    if not configs:
        raise ValueError("Curriculum did not yield any MettaGridConfig instances.")

    return configs


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return sanitized or "map"


def dump_game_configs(
    configs: Sequence[MettaGridConfig],
    names: Sequence[str],
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for existing in destination.glob("*.yaml"):
        existing.unlink()

    name_list = list(names)
    if len(name_list) < len(configs):
        name_list.extend(f"map_{idx:03d}" for idx in range(len(name_list), len(configs)))

    for index, (config_obj, raw_name) in enumerate(zip(configs, name_list, strict=False)):
        base_name = raw_name or f"map_{index:03d}"
        file_stem = sanitize_filename(base_name)
        target = destination / f"{file_stem}.yaml"
        if target.exists():
            target = destination / f"{file_stem}_{index:03d}.yaml"
        game_module.save_game_config(config_obj, target)


def resolve_run_dir(base_runs_dir: Path, run_dir: Optional[Path]) -> Path:
    if run_dir:
        return run_dir.expanduser().resolve()
    default_name = date.today().isoformat()
    return (base_runs_dir / default_name).resolve()


def suggest_parallelism(
    device: str,
    requested_envs: Optional[int],
    requested_workers: Optional[int],
) -> Tuple[int, int]:
    if requested_envs is not None and requested_workers is not None:
        return requested_envs, requested_workers

    cpu_count = os.cpu_count() or 4

    base_envs = cpu_count * 2 if device == "cuda" else cpu_count
    envs = (
        max(1, requested_envs) if requested_envs is not None else min(max(base_envs, 2), 32 if device == "cuda" else 16)
    )
    workers = max(1, requested_workers) if requested_workers is not None else max(1, min(envs, max(1, cpu_count // 2)))

    if envs % workers != 0:
        for candidate in range(workers, 0, -1):
            if envs % candidate == 0:
                workers = candidate
                break
        else:
            workers = 1
            envs = max(envs, workers)

    return envs, workers


def filter_uniform_agent_count(
    configs: Sequence[MettaGridConfig],
    names: Sequence[str],
) -> Tuple[list[MettaGridConfig], list[str], int]:
    if not configs:
        return list(configs), list(names), 0

    counts = Counter(cfg.game.num_agents for cfg in configs)
    target_agents = counts.most_common(1)[0][0]

    name_list = list(names)
    if len(name_list) < len(configs):
        name_list.extend(f"map_{idx:03d}" for idx in range(len(name_list), len(configs)))

    filtered_pairs = [
        (cfg, name_list[index]) for index, cfg in enumerate(configs) if cfg.game.num_agents == target_agents
    ]
    dropped = len(configs) - len(filtered_pairs)

    if not filtered_pairs:
        return list(configs), name_list[: len(configs)], dropped

    filtered_cfgs, filtered_names = zip(*filtered_pairs, strict=False)
    return list(filtered_cfgs), list(filtered_names), dropped


def resolve_initial_weights(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    if path.is_file():
        return path
    if not path.exists():
        raise ValueError(f"Initial weights path not found: {path}")
    if path.is_dir():
        raise ValueError(
            "Initial weights must reference a checkpoint file. Passing a directory "
            "(automatic latest-checkpoint detection) is temporarily disabled."
        )

    raise ValueError(f"Initial weights path is not a checkpoint file: {path}")
