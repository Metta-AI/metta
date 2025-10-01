"""Utility helpers for building CoGames curricula."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any, Deque, Iterable, Iterator, List, Optional, Sequence, Tuple

from cogames import game, utils
from mettagrid import MettaGridConfig
from mettagrid.util.module import load_symbol

DEFAULT_BIOME_GAMES: tuple[str, ...] = (
    "machina_1",
    "machina_1_big",
    "machina_2_bigger",
    "machina_3_big",
    "machina_4_bigger",
    "machina_5_big",
    "machina_6_bigger",
    "machina_7_big",
)

SUPPORTED_EXTENSIONS: Sequence[str] = (".yaml", ".yml", ".json", ".py")


class CurriculumError(ValueError):
    """Base exception for curriculum-related errors."""


class CurriculumArgumentError(CurriculumError):
    """Error raised when user-supplied curriculum arguments are invalid."""

    def __init__(self, message: str, *, param_name: Optional[str] = None) -> None:
        super().__init__(message)
        self.param_name = param_name


def ensure_config(value: Any, *, param_name: Optional[str] = None) -> MettaGridConfig:
    """Convert supported curriculum inputs into a ``MettaGridConfig``."""

    if isinstance(value, MettaGridConfig):
        return value
    if isinstance(value, (str, Path)):
        return game.get_game(str(value))
    if isinstance(value, dict):
        return MettaGridConfig.model_validate(value)
    raise CurriculumArgumentError(
        f"Unsupported curriculum item type: {type(value)!r}.",
        param_name=param_name,
    )


def _load_curriculum_configs(
    source: Any,
    max_items: int,
    *,
    param_name: Optional[str] = None,
) -> Sequence[MettaGridConfig]:
    queue: List[Any] = [source]
    configs: List[MettaGridConfig] = []

    while queue and len(configs) < max_items:
        item = queue.pop(0)

        if isinstance(item, (MettaGridConfig, str, Path, dict)):
            configs.append(ensure_config(item, param_name=param_name))
            continue

        if callable(item):
            produced = item()
            queue.append(produced)
            continue

        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            for produced in item:
                queue.append(produced)
                if len(configs) + len(queue) >= max_items:
                    break
            continue

        raise CurriculumArgumentError(
            f"Curriculum source produced unsupported type: {type(item)!r}.",
            param_name=param_name,
        )

    if not configs:
        raise CurriculumArgumentError(
            "Curriculum did not yield any MettaGridConfig instances.",
            param_name=param_name,
        )

    return configs


def sanitize_map_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return sanitized or "map"


def dump_game_configs(configs: Sequence[MettaGridConfig], names: Sequence[str], output_dir: Path) -> None:
    """Write configs to disk, using sanitized file names derived from labels."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for existing in output_dir.glob("*.yaml"):
        existing.unlink()
    for index, (config_obj, raw_name) in enumerate(zip(configs, names, strict=False)):
        base_name = raw_name or f"map_{index:03d}"
        file_stem = sanitize_map_name(base_name)
        candidate = output_dir / f"{file_stem}.yaml"
        if candidate.exists():
            candidate = output_dir / f"{file_stem}_{index:03d}.yaml"
        game.save_game_config(config_obj, candidate)


def collect_curriculum_configs(
    game_names: Iterable[str],
    *,
    curriculum_path: Optional[str],
    max_items: int,
    fallback_folder: Optional[Path],
    game_param: str,
) -> Tuple[List[MettaGridConfig], List[str]]:
    configs: List[MettaGridConfig] = []
    names: List[str] = []

    for game_name in game_names:
        resolved_game, error = utils.resolve_game(game_name)
        if error:
            raise CurriculumArgumentError(error, param_name=game_param)
        if resolved_game is None:
            raise CurriculumArgumentError(
                f"game '{game_name}' not found",
                param_name=game_param,
            )
        configs.append(game.get_game(resolved_game))
        names.append(resolved_game)

    if curriculum_path is not None:
        try:
            curriculum_source = load_symbol(curriculum_path)
        except Exception as exc:  # pragma: no cover - import failure is user-facing
            raise CurriculumArgumentError(
                f"Failed to import curriculum '{curriculum_path}': {exc}",
                param_name="curriculum",
            ) from exc

        curriculum_cfgs = _load_curriculum_configs(
            curriculum_source,
            max_items,
            param_name="curriculum",
        )
        start_index = len(names)
        for offset, cfg in enumerate(curriculum_cfgs):
            cfg_name = getattr(getattr(cfg, "game", None), "name", None)
            label = str(cfg_name) if cfg_name else f"curriculum_{start_index + offset:03d}"
            configs.append(cfg)
            names.append(label)
    elif fallback_folder is not None and fallback_folder.exists():
        try:
            folder_cfgs, folder_names = load_map_folder_with_names(fallback_folder)
        except (FileNotFoundError, NotADirectoryError, ValueError):
            pass
        else:
            configs.extend(folder_cfgs)
            names.extend(folder_names)

    for index in range(len(names), len(configs)):
        names.append(f"map_{index:03d}")

    return configs, names


def _discover_map_files(directory: Path) -> List[Path]:
    files = [
        path for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def load_map_folder_with_names(directory: str | Path) -> Tuple[List[MettaGridConfig], List[str]]:
    """Load map configs and their corresponding filenames (without suffix)."""
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

    configs: List[MettaGridConfig] = []
    names: List[str] = []
    for path in files:
        cfg = game.get_game(str(path))
        configs.append(cfg)
        names.append(path.stem)
    return configs, names


def load_map_folder(directory: str | Path) -> Deque[MettaGridConfig]:
    """Load all map configs from *directory* and return them as a reusable queue."""

    configs, _ = load_map_folder_with_names(directory)
    return deque(configs)


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
