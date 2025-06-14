import random
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from mettagrid.room.terrain_from_numpy import TerrainFromNumpy


def create_map(tmp_path: Path) -> np.ndarray:
    grid = np.array(
        [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "empty", "empty", "empty", "wall"],
            ["wall", "empty", "agent.agent", "empty", "wall"],
            ["wall", "empty", "empty", "empty", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ],
        dtype="<U16",
    )
    np.save(tmp_path / "test.npy", grid)
    return grid


def test_build_basic(tmp_path):
    level = create_map(tmp_path)
    objs = OmegaConf.create({"block": 1})
    env = TerrainFromNumpy(objs, agents=1, dir=str(tmp_path), file="test.npy")
    random.seed(0)
    np.random.seed(0)
    grid = env._build()
    assert grid.shape == level.shape
    assert (grid == "agent.agent").sum() == 1
    assert (grid == "block").sum() == 1


def test_build_with_border(tmp_path):
    create_map(tmp_path)
    objs = OmegaConf.create({})
    env = TerrainFromNumpy(objs, agents=0, dir=str(tmp_path), file="test.npy", border_width=1)
    random.seed(0)
    np.random.seed(0)
    level = env.build()
    assert level.grid.shape == (7, 7)


def test_agent_placement(tmp_path):
    create_map(tmp_path)
    objs = OmegaConf.create({})
    num_agents = 4
    env = TerrainFromNumpy(
        objs, agents=num_agents, dir=str(tmp_path), file="test.npy", border_width=1
    )
    random.seed(0)
    np.random.seed(0)
    level = env.build()
    assert (level.grid == "agent.agent").sum() == num_agents
