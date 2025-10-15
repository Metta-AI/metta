import numpy as np

from mettagrid.map_builder.utils import (
    compute_positions,
    create_grid,
    draw_border,
    make_odd,
    sample_position,
    set_position,
)
from mettagrid.mapgen.types import map_grid_dtype


class TestCreateGrid:
    def test_create_grid_default(self):
        grid = create_grid(3, 4)
        expected = np.full((3, 4), "empty", dtype=map_grid_dtype)
        assert np.array_equal(grid, expected)
        assert grid.dtype == map_grid_dtype

    def test_create_grid_custom_fill(self):
        grid = create_grid(2, 3, "wall")
        expected = np.full((2, 3), "wall", dtype=map_grid_dtype)
        assert np.array_equal(grid, expected)

    def test_create_grid_single_cell(self):
        grid = create_grid(1, 1, "agent.agent")
        expected = np.array([["agent.agent"]], dtype=map_grid_dtype)
        assert np.array_equal(grid, expected)

    def test_create_grid_large(self):
        grid = create_grid(100, 200, "test")
        assert grid.shape == (100, 200)
        assert np.all(grid == "test")


class TestDrawBorder:
    def test_draw_border_width_1(self):
        grid = create_grid(5, 5, "empty")
        draw_border(grid, 1, "wall")

        expected = np.array(
            [
                ["wall", "wall", "wall", "wall", "wall"],
                ["wall", "empty", "empty", "empty", "wall"],
                ["wall", "empty", "empty", "empty", "wall"],
                ["wall", "empty", "empty", "empty", "wall"],
                ["wall", "wall", "wall", "wall", "wall"],
            ],
            dtype=map_grid_dtype,
        )

        assert np.array_equal(grid, expected)

    def test_draw_border_width_2(self):
        grid = create_grid(6, 6, "empty")
        draw_border(grid, 2, "wall")

        expected = np.array(
            [
                ["wall", "wall", "wall", "wall", "wall", "wall"],
                ["wall", "wall", "wall", "wall", "wall", "wall"],
                ["wall", "wall", "empty", "empty", "wall", "wall"],
                ["wall", "wall", "empty", "empty", "wall", "wall"],
                ["wall", "wall", "wall", "wall", "wall", "wall"],
                ["wall", "wall", "wall", "wall", "wall", "wall"],
            ],
            dtype=map_grid_dtype,
        )

        assert np.array_equal(grid, expected)

    def test_draw_border_zero_width(self):
        original_grid = create_grid(3, 3, "empty")
        grid = original_grid.copy()
        draw_border(grid, 0, "wall")

        # Should remain unchanged
        assert np.array_equal(grid, original_grid)

    def test_draw_border_fills_entire_grid(self):
        grid = create_grid(3, 3, "empty")
        draw_border(grid, 2, "wall")  # Border larger than half the grid

        # Entire grid should be walls
        expected = np.full((3, 3), "wall", dtype=map_grid_dtype)
        assert np.array_equal(grid, expected)

    def test_draw_border_modifies_in_place(self):
        grid = create_grid(3, 3, "empty")
        original_id = id(grid)
        draw_border(grid, 1, "wall")

        # Same object reference
        assert id(grid) == original_id


class TestComputePositions:
    def test_compute_positions_single_block(self):
        blocks = [("block1", 5)]
        positions = compute_positions(0, 10, blocks)
        assert positions == {"block1": 0}

    def test_compute_positions_two_blocks_equal_size(self):
        blocks = [("block1", 3), ("block2", 3)]
        positions = compute_positions(0, 10, blocks)
        # Total blocks: 6, total space: 10, gap: 4
        assert positions == {"block1": 0, "block2": 7}

    def test_compute_positions_three_blocks(self):
        blocks = [("a", 2), ("b", 2), ("c", 2)]
        positions = compute_positions(0, 12, blocks)
        # Total blocks: 6, total space: 12, gap: 6, per gap: 3
        assert positions == {"a": 0, "b": 5, "c": 10}

    def test_compute_positions_uneven_distribution(self):
        blocks = [("a", 1), ("b", 1), ("c", 1)]
        positions = compute_positions(0, 8, blocks)
        # Total blocks: 3, total space: 8, gap: 5, per gap: 2, extra: 1
        assert positions == {"a": 0, "b": 4, "c": 7}

    def test_compute_positions_no_gap_space(self):
        blocks = [("a", 3), ("b", 2)]
        positions = compute_positions(0, 5, blocks)
        assert positions == {"a": 0, "b": 3}

    def test_compute_positions_different_sizes(self):
        blocks = [("small", 1), ("medium", 3), ("large", 2)]
        positions = compute_positions(0, 10, blocks)
        # Total blocks: 6, total space: 10, gap: 4, per gap: 2
        assert positions == {"small": 0, "medium": 3, "large": 8}

    def test_compute_positions_empty_blocks(self):
        blocks = []
        positions = compute_positions(0, 10, blocks)
        assert positions == {}


class TestSamplePosition:
    def test_sample_position_no_constraints(self):
        pos = sample_position(0, 10, 0, 10, 0, [])
        assert 0 <= pos[0] <= 10
        assert 0 <= pos[1] <= 10

    def test_sample_position_with_min_distance(self):
        existing = [(5, 5)]
        pos = sample_position(0, 10, 0, 10, 3, existing, rng=np.random.default_rng(42))
        # Should be at least distance 3 from (5,5)
        distance = abs(pos[0] - 5) + abs(pos[1] - 5)
        assert distance >= 3

    def test_sample_position_with_forbidden_positions(self):
        forbidden = {(3, 3), (4, 4), (5, 5)}
        pos = sample_position(3, 5, 3, 5, 0, [], forbidden=forbidden, rng=np.random.default_rng(42))
        assert pos not in forbidden

    def test_sample_position_fallback_when_impossible(self):
        # Create impossible constraints
        existing = [(1, 1)]
        forbidden = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
        pos = sample_position(0, 2, 0, 2, 2, existing, forbidden=forbidden, attempts=5)
        # Should fallback to (x_min, y_min)
        assert pos == (0, 0)

    def test_sample_position_with_custom_rng(self):
        rng = np.random.default_rng(12345)
        pos1 = sample_position(0, 10, 0, 10, 0, [], rng=rng)

        rng = np.random.default_rng(12345)  # Same seed
        pos2 = sample_position(0, 10, 0, 10, 0, [], rng=rng)

        assert pos1 == pos2

    def test_sample_position_single_cell_region(self):
        pos = sample_position(5, 5, 7, 7, 0, [])
        assert pos == (5, 7)

    def test_sample_position_multiple_existing_constraints(self):
        existing = [(2, 2), (8, 8), (2, 8), (8, 2)]
        pos = sample_position(0, 10, 0, 10, 3, existing, rng=np.random.default_rng(42))

        # Check distance from all existing positions
        for ex, ey in existing:
            distance = abs(pos[0] - ex) + abs(pos[1] - ey)
            assert distance >= 3


class TestMakeOdd:
    def test_make_odd_even_numbers(self):
        assert make_odd(2) == 3
        assert make_odd(4) == 5
        assert make_odd(0) == 1
        assert make_odd(-2) == -1

    def test_make_odd_odd_numbers(self):
        assert make_odd(1) == 1
        assert make_odd(3) == 3
        assert make_odd(5) == 5
        assert make_odd(-1) == -1
        assert make_odd(-3) == -3

    def test_make_odd_large_numbers(self):
        assert make_odd(100) == 101
        assert make_odd(1001) == 1001


class TestSetPosition:
    def test_set_position_normal_range(self):
        assert set_position(5, 10) == 5
        assert set_position(3, 10) == 3

    def test_set_position_even_numbers_made_odd(self):
        assert set_position(4, 10) == 5
        assert set_position(6, 10) == 7

    def test_set_position_negative_numbers(self):
        assert set_position(-1, 10) == 1
        assert set_position(-5, 10) == 1

    def test_set_position_at_boundary(self):
        assert set_position(9, 10) == 9
        assert set_position(10, 10) == 8  # 10 is even, so 10-2=8

    def test_set_position_above_boundary(self):
        assert set_position(15, 10) == 8  # 15 is odd, so upper_bound-2
        assert set_position(14, 10) == 8  # 14 is even, make_odd(14)=15, so upper_bound-2

    def test_set_position_small_boundaries(self):
        assert set_position(0, 3) == 1
        assert set_position(1, 3) == 1
        assert set_position(2, 3) == 1  # 2 is even, becomes 3, but 3>=3, so fallback
        assert set_position(3, 3) == 1  # 3>=3, so upper_bound-2=1

    def test_set_position_edge_cases(self):
        assert set_position(0, 1) == -1  # make_odd(0)=1, 1>=1, so upper_bound-2=-1
        assert set_position(1, 2) == 1
        assert set_position(2, 2) == 0  # make_odd(2)=3, 3>=2, so upper_bound-2=0
