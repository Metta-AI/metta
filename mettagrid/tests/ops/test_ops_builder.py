from metta.mettagrid.map_builder.ops_builder import OpsMapBuilder
from metta.mettagrid.mapgen.ops import DrawOp


def test_ops_map_builder_builds_with_border():
    ops = [DrawOp(start=(2, 2), end=(7, 2), thickness=1)]
    cfg = OpsMapBuilder.Config(ops=ops, width=12, height=12, initial_fill="empty", border_width=2)
    builder = OpsMapBuilder(cfg)
    game_map = builder.build()

    grid = game_map.grid
    # Outer border walls
    assert (grid[0, :] == "wall").all()
    assert (grid[-1, :] == "wall").all()
    assert (grid[:, 0] == "wall").all()
    assert (grid[:, -1] == "wall").all()

    # Inner line should be empty
    for y in range(2, 8):
        assert grid[y, 2] == "empty"




