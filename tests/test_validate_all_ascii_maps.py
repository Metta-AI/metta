"""
Test suite for validating ASCII map files.
"""

import functools
import pathlib
import sys
import typing

import pytest
import yaml

import metta.common.util.fs
import mettagrid.map_builder.map_builder

metta.common.util.fs.cd_repo_root()
REPO_ROOT = metta.common.util.fs.get_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DIRECTORIES = (
    pathlib.Path("packages/mettagrid/configs/maps"),
    pathlib.Path("packages/cogames/src/cogames/maps"),
)


def find_map_files(root_dir: pathlib.Path | str) -> list[pathlib.Path]:
    """
    Find all .map files.

    Args:
        root_dir: Root directory to search from

    Returns:
        Sorted list of absolute paths for .map files
    """
    root_path = pathlib.Path(root_dir).resolve()

    # Return empty list if directory doesn't exist
    if not root_path.exists():
        return []

    map_files = list(root_path.rglob("*.map"))

    return sorted(map_files)


def map_files() -> list[pathlib.Path]:
    candidates: typing.Iterable[pathlib.Path] = (
        map_file for directory in _default_directories() for map_file in find_map_files(directory)
    )
    return sorted(candidates)


def _default_directories() -> list[pathlib.Path]:
    return sorted({(REPO_ROOT / directory).resolve() for directory in DEFAULT_DIRECTORIES})


@functools.lru_cache(maxsize=None)
def _load_map_builder_config_cached(map_file: str) -> mettagrid.map_builder.map_builder.MapBuilderConfig[typing.Any]:
    with pathlib.Path(map_file).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    config = mettagrid.map_builder.map_builder.MapBuilderConfig.model_validate(data)
    assert isinstance(config, mettagrid.map_builder.map_builder.MapBuilderConfig)
    return config


def load_map_builder_config(map_file: pathlib.Path) -> mettagrid.map_builder.map_builder.MapBuilderConfig[typing.Any]:
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
def test_map_builder_can_build(map_file: pathlib.Path):
    config = load_map_builder_config(map_file)
    builder = config.create()
    builder.build()
