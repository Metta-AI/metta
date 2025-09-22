import numpy as np

from experiments.evals.navigation_with_corridors import (
    make_corridors_env,
    make_grid_maze_env,
    make_radial_mini_env,
)
from metta.mettagrid.mapgen.mapgen import MapGen


def _build(env_func):
    cfg = env_func()
    instance_map_cfg = cfg.game.map_builder.instance_map
    # If instance_map is already a MapGen.Config, pass through
    if isinstance(instance_map_cfg, MapGen.Config):
        m = MapGen(instance_map_cfg)
    else:
        # Wrap OpsMapBuilder.Config into a MapGen.Config for building
        m = MapGen(MapGen.Config(instance_map=instance_map_cfg, border_width=0, instances=1, instance_border_width=0))
    return m.build().grid


def test_radial_mini_ops_structure():
    grid = _build(make_radial_mini_env)
    # Center cell should be empty and spokes present in 4-neighborhoods
    assert grid[11, 11] in ("empty", "agent.agent")
    # Check some spoke ends where we stamp altars
    assert grid[3, 11] == "altar"
    assert grid[11, 19] == "altar"


def test_grid_maze_ops_has_intersections():
    grid = _build(make_grid_maze_env)
    # Ensure grid-like crossing: central crossing should be empty
    assert grid[30, 30] == "empty"
    # Corners should have altars per placement
    assert grid[10, 10] == "altar"
    assert grid[50, 50] == "altar"


def test_corridors_ops_main_trunk_present():
    grid = _build(lambda: make_corridors_env(False))
    # Main horizontal trunk at row 13 should be mostly empty
    row = grid[13]
    empties = int(np.sum(row == "empty"))
    assert empties > 40
