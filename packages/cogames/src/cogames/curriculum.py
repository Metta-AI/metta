"""Utility helpers for building CoGames curricula."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Iterable, Iterator, List, Sequence

from cogames import game
from mettagrid import MettaGridConfig

SUPPORTED_EXTENSIONS: Sequence[str] = (".yaml", ".yml", ".json", ".py")


def _discover_map_files(directory: Path) -> List[Path]:
    files = [
        path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def load_map_folder(directory: str | Path) -> Deque[MettaGridConfig]:
    """Load all map configs from *directory* and return them as a reusable queue."""
    dir_path = Path(directory)
    if not dir_path.exists():
        msg = f"Curriculum directory not found: {dir_path}"
        raise FileNotFoundError(msg)
    if not dir_path.is_dir():
        msg = f"Curriculum path is not a directory: {dir_path}"
        raise NotADirectoryError(msg)

    files = _discover_map_files(dir_path)
    if not files:
        msg = f"No supported map files found in {dir_path}"
        raise ValueError(msg)

    configs: Deque[MettaGridConfig] = deque()
    for path in files:
        cfg = game.get_game(str(path))
        configs.append(cfg)
    return configs


def cycle_maps(directory: str | Path) -> Iterator[MettaGridConfig]:
    """Yield `MettaGridConfig` objects by cycling through map files in *directory*.

    The generator loads every supported file in the directory once and then
    yields deep copies in a round-robin order so callers can safely mutate the
    configs without affecting subsequent iterations.
    """

    configs = load_map_folder(directory)
    if not configs:
        return iter(())

    while True:
        cfg = configs[0]
        configs.rotate(-1)
        yield cfg.model_copy(deep=True)


def curriculum_from_folder(directory: str | Path) -> Iterable[MettaGridConfig]:
    """Convenience wrapper returning a curriculum iterable for CLI usage."""

    return cycle_maps(directory)
