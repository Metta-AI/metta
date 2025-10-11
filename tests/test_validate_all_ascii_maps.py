"""
Test suite for validating ASCII map files.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pytest
import yaml

from metta.common.util.fs import cd_repo_root, get_repo_root
from mettagrid.map_builder.map_builder import MapBuilderConfig, validate_any_map_builder
from tools.map.convert_legacy_maps_to_yaml import DEFAULT_DIRECTORIES

cd_repo_root()
REPO_ROOT = get_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def find_map_files(root_dir: Path | str) -> list[Path]:
    """
    Find all .map files.

    Args:
        root_dir: Root directory to search from

    Returns:
        Sorted list of absolute paths for .map files
    """
    root_path = Path(root_dir).resolve()

    # Return empty list if directory doesn't exist
    if not root_path.exists():
        return []

    map_files = list(root_path.rglob("*.map"))

    return sorted(map_files)


def map_files() -> list[Path]:
    candidates: Iterable[Path] = (
        map_file for directory in _default_directories() for map_file in find_map_files(directory)
    )
    return sorted(candidates)


def _default_directories() -> list[Path]:
    return sorted({(REPO_ROOT / directory).resolve() for directory in DEFAULT_DIRECTORIES})


@lru_cache(maxsize=None)
def _load_map_builder_config_cached(map_file: str) -> MapBuilderConfig[Any]:
    with Path(map_file).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    config = validate_any_map_builder(data)
    assert isinstance(config, MapBuilderConfig)
    return config


def load_map_builder_config(map_file: Path) -> MapBuilderConfig[Any]:
    return _load_map_builder_config_cached(str(map_file))


MAP_FILES = map_files()
if MAP_FILES:
    pytest_parametrize = pytest.mark.parametrize("map_file", MAP_FILES, ids=[str(path) for path in MAP_FILES])
else:
    directories = ", ".join(str(directory) for directory in _default_directories())
    pytest_parametrize = pytest.mark.skip(reason=f"No map files found in {directories}")


def test_map_files_discovered():
    assert MAP_FILES, "Should discover at least one .map file"


@pytest_parametrize
def test_map_builder_can_build(map_file: Path):
    config = load_map_builder_config(map_file)
    builder = config.create()
    builder.build()
