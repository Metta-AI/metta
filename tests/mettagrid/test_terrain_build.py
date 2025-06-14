import random

import numpy as np
from omegaconf import DictConfig

from mettagrid.room import terrain_utils
from mettagrid.room.terrain_from_numpy import TerrainFromNumpy


def python_valid_positions(level: np.ndarray) -> list[tuple[int, int]]:
    positions = []
    for i in range(1, level.shape[0] - 1):
        for j in range(1, level.shape[1] - 1):
            if level[i, j] == "empty":
                if (
                    level[i - 1, j] == "empty"
                    or level[i + 1, j] == "empty"
                    or level[i, j - 1] == "empty"
                    or level[i, j + 1] == "empty"
                ):
                    positions.append((i, j))
    return positions


def test_cpp_valid_positions_matches_python():
    grid = np.array(
        [
            ["wall", "wall", "wall", "wall"],
            ["wall", "empty", "empty", "wall"],
            ["wall", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall"],
        ],
        dtype=object,
    )

    expected = python_valid_positions(grid)
    actual = terrain_utils.get_valid_positions(grid)
    assert sorted(expected) == sorted(actual)


def test_build_with_cpp(tmp_path):
    grid = np.array(
        [
            ["wall", "wall", "wall", "wall"],
            ["wall", "empty", "empty", "wall"],
            ["wall", "agent.agent", "empty", "wall"],
            ["wall", "wall", "wall", "wall"],
        ],
        dtype=object,
    )

    path = tmp_path / "test.npy"
    np.save(path, grid)

    room = TerrainFromNumpy(objects=DictConfig({"rock": 1}), agents=1, dir=str(tmp_path), file="test.npy")
    random.seed(0)
    level = room.build().grid

    assert (level == "agent.agent").sum() == 1
    assert (level == "rock").sum() == 1
