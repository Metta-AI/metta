from metta.mettagrid.map_builder.utils import create_grid
from metta.mettagrid.mapgen.ops import DrawOp, Operation, apply_ops, surround_with_walls


def test_apply_draw_op_vertical_line_basic():
    grid = create_grid(20, 20)

    ops: list[Operation] = [DrawOp(start=(2, 5), end=(17, 5), thickness=1, material="empty")]
    apply_ops(grid, ops, initial_fill="wall")

    # Line cells are empty
    for y in range(2, 18):
        assert grid[y, 5] == "empty"

    # Sample off-line cells remain walls
    for y in (2, 10, 17):
        assert grid[y, 4] == "wall"
        assert grid[y, 6] == "wall"


def test_surround_with_walls_adds_side_walls_only():
    grid = create_grid(20, 20)

    base: list[DrawOp] = [DrawOp(start=(2, 5), end=(17, 5), thickness=1, material="empty")]
    walls = surround_with_walls(base, wall_thickness=1)
    ops: list[Operation] = [*base, *walls]

    # Start from empty to see walls introduced only by surround ops
    apply_ops(grid, ops, initial_fill="empty")

    # Corridor cells are empty; immediate side neighbors are walls; farther cells stay empty
    for y in range(2, 18):
        assert grid[y, 5] == "empty"
        assert grid[y, 4] == "wall"
        assert grid[y, 6] == "wall"

    # Corners away from the line remain empty
    assert grid[0, 0] == "empty"
    assert grid[19, 19] == "empty"
