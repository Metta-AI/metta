"""
Renderer buffer and symbol tests.

These tests validate emoji rendering, bounds, and symbol mapping using the
MapBuffer and symbol helpers without requiring the interactive GUI.
"""

import io
import sys
from contextlib import contextmanager

import pytest

from mettagrid.config.mettagrid_config import GameConfig, WallConfig
from mettagrid.renderer.miniscope.buffer import MapBuffer, compute_bounds
from mettagrid.renderer.miniscope.symbol import DEFAULT_SYMBOL_MAP, get_symbol_for_object


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout during tests."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


def make_grid_object(type_id: int, r: int, c: int, **kwargs) -> dict:
    """Factory to create object dicts matching the C++ API contract for tests."""
    obj = {
        "type": type_id,
        "r": r,
        "c": c,
        "locations": [(c, r, 0)],  # (c, r, layer)
    }
    obj.update(kwargs)
    return obj


@pytest.fixture
def object_type_names():
    return [
        "agent",
        "wall",
        "altar",
        "mine_red",
        "generator_red",
        "lasery",
        "marker",
        "block",
    ]


@pytest.fixture
def game_config():
    return GameConfig(
        resource_names=[],
        num_agents=1,
        max_steps=100,
        obs_width=7,
        obs_height=7,
        num_observation_tokens=50,
        objects={
            "altar": WallConfig(name="altar", type_id=2, map_char="_", render_symbol="🎯"),
            "mine_red": WallConfig(name="mine_red", type_id=3, map_char="r", render_symbol="🔺"),
            "generator_red": WallConfig(name="generator_red", type_id=4, map_char="g", render_symbol="🔋"),
            "lasery": WallConfig(name="lasery", type_id=5, map_char="L", render_symbol="🟥"),
            "marker": WallConfig(name="marker", type_id=6, map_char="m", render_symbol="🟠"),
            "block": WallConfig(name="block", type_id=7, map_char="s", render_symbol="📦"),
        },
    )


@pytest.fixture
def symbol_map(game_config):
    m = DEFAULT_SYMBOL_MAP.copy()
    for name, cfg in game_config.objects.items():
        m[name] = cfg.render_symbol
    return m


class TestMapBufferAndSymbols:
    def test_buffer_initialization(self, object_type_names, symbol_map):
        buf = MapBuffer(object_type_names, symbol_map, initial_height=10, initial_width=10)
        assert buf.get_bounds() == (0, 0, 10, 10)

    def test_symbol_mapping(self, object_type_names, symbol_map):
        # Basic types
        assert get_symbol_for_object({"type": 0}, object_type_names, symbol_map) in {"🤖", "🟦"}
        assert get_symbol_for_object({"type": 1}, object_type_names, symbol_map) == symbol_map["wall"]
        assert get_symbol_for_object({"type": 2}, object_type_names, symbol_map) == symbol_map["altar"]
        # Agent numbering 0-9
        for i, expected in enumerate(["🟦", "🟧", "🟩", "🟨", "🟪", "🟥", "🟫", "⬛", "🟦", "🟧"]):
            sym = get_symbol_for_object({"type": 0, "agent_id": i}, object_type_names, symbol_map)
            assert sym == expected
        # Agent >=10 fallback to robot
        assert get_symbol_for_object({"type": 0, "agent_id": 10}, object_type_names, symbol_map) == "🤖"

    def test_compute_bounds_with_walls(self, object_type_names):
        grid_objects = {
            0: make_grid_object(1, 0, 0),
            1: make_grid_object(1, 0, 5),
            2: make_grid_object(1, 3, 0),
            3: make_grid_object(1, 3, 5),
            4: make_grid_object(0, 1, 2),
        }
        min_row, min_col, height, width = compute_bounds(grid_objects, object_type_names)
        assert min_row == 0
        assert min_col == 0
        assert height == 4
        assert width == 6

    def test_compute_bounds_handles_locations(self, object_type_names):
        grid_objects = {
            0: {"type": 1, "location": (3, 2, 0), "locations": [(3, 2, 0)]},
            1: {"type": 1, "location": (7, 5, 0), "locations": [(7, 5, 0)]},
        }
        min_row, min_col, height, width = compute_bounds(grid_objects, object_type_names)
        assert min_row == 2
        assert min_col == 3
        assert height == 4
        assert width == 5

    def test_render_simple_grid(self, object_type_names, symbol_map):
        buf = MapBuffer(object_type_names, symbol_map, initial_height=10, initial_width=10)
        grid_objects = {
            0: make_grid_object(1, 0, 0),
            1: make_grid_object(1, 0, 2),
            2: make_grid_object(0, 1, 1, agent_id=0),
            3: make_grid_object(2, 2, 1),
            4: make_grid_object(1, 2, 0),
            5: make_grid_object(1, 2, 2),
        }
        with suppress_stdout():
            output = buf.render_full_map(grid_objects)
        assert symbol_map["wall"] in output
        assert "🟦" in output
        assert symbol_map["altar"] in output
        assert symbol_map["empty"] in output

    def test_render_with_special_objects(self, object_type_names, symbol_map):
        buf = MapBuffer(object_type_names, symbol_map, initial_height=10, initial_width=10)
        grid_objects = {
            0: make_grid_object(1, 0, 0),
            1: make_grid_object(0, 1, 1, agent_id=0),
            2: make_grid_object(5, 1, 2),  # lasery
            3: make_grid_object(6, 2, 1),  # marker
            4: make_grid_object(7, 2, 2),  # block
            5: make_grid_object(1, 3, 3),
        }
        with suppress_stdout():
            output = buf.render_full_map(grid_objects)
        assert "🟥" in output
        assert "🟠" in output
        assert "📦" in output

    def test_viewport_arrows_at_edges(self, object_type_names, symbol_map):
        buf = MapBuffer(object_type_names, symbol_map, initial_height=10, initial_width=10)
        grid_objects = {}
        obj_id = 0
        for r in range(10):
            for c in range(10):
                grid_objects[obj_id] = make_grid_object(1, r, c)
                obj_id += 1
        # Center viewport at (5,5) with 3x3
        buf.set_viewport(center_row=5, center_col=5, height=3, width=3)
        rendered = buf.render(grid_objects, use_viewport=True)
        lines = rendered.split("\n")
        assert len(lines) == 3
        assert "▲" in lines[0]
        assert "▼" in lines[2]
        assert "◀" in lines[1][0:2]
        assert "▶" in lines[1][-2:]

    def test_viewport_no_arrows_when_full_map(self, object_type_names, symbol_map):
        buf = MapBuffer(object_type_names, symbol_map, initial_height=3, initial_width=3)
        grid_objects = {0: make_grid_object(1, 0, 0), 1: make_grid_object(1, 2, 2)}
        rendered = buf.render_full_map(grid_objects)
        assert "▲" not in rendered
        assert "▼" not in rendered
        assert "◀" not in rendered
        assert "▶" not in rendered

    def test_bounds_checking_skips_out_of_range_objects(self, object_type_names, symbol_map):
        buf = MapBuffer(object_type_names, symbol_map, initial_height=10, initial_width=10)
        grid_objects = {
            0: make_grid_object(0, 0, 0, agent_id=0),
            1: make_grid_object(1, 5, 5),
            2: make_grid_object(1, 9, 9),
            3: make_grid_object(1, -1, 0),
            4: make_grid_object(1, 15, 15),
        }
        buf.set_viewport(center_row=5, center_col=5, height=3, width=3)
        rendered = buf.render(grid_objects, use_viewport=True)
        assert isinstance(rendered, str)
        assert len(rendered) > 0

    def test_unknown_object_fallback(self, object_type_names, symbol_map):
        # Append an unknown type name and use it
        object_type_names = object_type_names + ["unknown_type"]
        sym = get_symbol_for_object({"type": len(object_type_names) - 1}, object_type_names, symbol_map)
        assert sym == "❓"
