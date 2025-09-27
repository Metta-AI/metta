import numpy as np

from mettagrid.mapgen.mapgen import MapGen
from mettagrid.mapgen.scene import ChildrenAction
from mettagrid.mapgen.scenes.maze import Maze, MazeParams
from mettagrid.mapgen.scenes.quadrants import Quadrants, QuadrantsParams


def _build_map(seed: int) -> np.ndarray:
    cfg = MapGen.Config(
        width=40,
        height=40,
        seed=seed,
        root=Quadrants.factory(
            QuadrantsParams(base_size=11),
            children_actions=[
                # Put a Maze in each quadrant; rely on deterministic child seeding
                dict(
                    scene=Maze.factory(MazeParams(algorithm="dfs")),
                    where={"tags": ["quadrant.0"]},
                    order_by="first",
                ),
                dict(
                    scene=Maze.factory(MazeParams(algorithm="dfs")),
                    where={"tags": ["quadrant.1"]},
                    order_by="first",
                ),
                dict(
                    scene=Maze.factory(MazeParams(algorithm="dfs")),
                    where={"tags": ["quadrant.2"]},
                    order_by="first",
                ),
                dict(
                    scene=Maze.factory(MazeParams(algorithm="dfs")),
                    where={"tags": ["quadrant.3"]},
                    order_by="first",
                ),
            ],
        ),
    )
    level = MapGen(cfg).build()
    return level.grid.copy()


def test_same_seed_same_map():
    g1 = _build_map(1234)
    g2 = _build_map(1234)
    assert np.array_equal(g1, g2)


def test_different_seed_different_map():
    g1 = _build_map(1234)
    g2 = _build_map(1235)
    assert not np.array_equal(g1, g2)


def test_quadrants_not_identical_with_offsets():
    # Use seed offsets to ensure quadrants diverge deterministically
    cfg = MapGen.Config(
        width=40,
        height=40,
        seed=999,
        root=Quadrants.factory(
            QuadrantsParams(base_size=11),
            children_actions=[
                ChildrenAction(
                    scene=Maze.factory(MazeParams(algorithm="dfs")),
                    where={"tags": ["quadrant.0"]},
                    order_by="first",
                    seed_offset=0,
                ),
                ChildrenAction(
                    scene=Maze.factory(MazeParams(algorithm="dfs")),
                    where={"tags": ["quadrant.1"]},
                    order_by="first",
                    seed_offset=1,
                ),
                ChildrenAction(
                    scene=Maze.factory(MazeParams(algorithm="dfs")),
                    where={"tags": ["quadrant.2"]},
                    order_by="first",
                    seed_offset=2,
                ),
                ChildrenAction(
                    scene=Maze.factory(MazeParams(algorithm="dfs")),
                    where={"tags": ["quadrant.3"]},
                    order_by="first",
                    seed_offset=3,
                ),
            ],
        ),
    )
    grid = MapGen(cfg).build().grid
    h, w = grid.shape
    cx, cy = w // 2, h // 2
    q0 = grid[:cy, :cx]
    q1 = grid[:cy, cx:]
    q2 = grid[cy:, :cx]
    q3 = grid[cy:, cx:]
    # At least two quadrants must differ
    assert not (np.array_equal(q0, q1) and np.array_equal(q1, q2) and np.array_equal(q2, q3))
