"""
Tests for the MiniscopeRenderer class.

This module contains unit tests for the MiniscopeRenderer, ensuring proper
emoji rendering, alignment, and functionality for MettaGrid environments.
"""

import io
import sys
from contextlib import contextmanager

import numpy as np
import pytest
from rich.console import Console

from mettagrid.config.mettagrid_config import GameConfig, WallConfig
from mettagrid.renderer.miniscope import MiniscopeRenderer


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout during tests."""
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout


class TestMiniscopeRenderer:
    """Test suite for MiniscopeRenderer functionality."""

    @pytest.fixture
    def object_type_names(self):
        """Provide standard object type names for testing."""
        return ["agent", "wall", "altar", "mine_red", "generator_red", "lasery", "marker", "block"]

    @pytest.fixture
    def game_config(self):
        """Provide a minimal GameConfig for testing."""

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
    def renderer(self, object_type_names, game_config):
        return MiniscopeRenderer(object_type_names, game_config, map_height=10, map_width=10)

    def test_initialization(self, renderer):
        """Test that MiniscopeRenderer initializes correctly."""
        assert renderer._bounds_set is True  # True when map dimensions are provided
        assert renderer._min_row == 0
        assert renderer._min_col == 0
        assert renderer._height == 10
        assert renderer._width == 10
        assert renderer._last_buffer is None

    def test_symbol_mapping(self, renderer):
        """Test that objects map to correct emoji symbols."""
        # Test basic object types
        test_cases = [
            ({"type": 0, "r": 0, "c": 0}, "🤖"),  # agent
            ({"type": 1, "r": 0, "c": 0}, "⬛"),  # wall
            ({"type": 2, "r": 0, "c": 0}, "🎯"),  # altar
            ({"type": 3, "r": 0, "c": 0}, "🔺"),  # mine_red
            ({"type": 4, "r": 0, "c": 0}, "🔋"),  # generator_red
            ({"type": 5, "r": 0, "c": 0}, "🟥"),  # lasery
            ({"type": 6, "r": 0, "c": 0}, "🟠"),  # marker
            ({"type": 7, "r": 0, "c": 0}, "📦"),  # block
        ]

        for obj, expected_symbol in test_cases:
            symbol = renderer._symbol_for(obj)
            assert symbol == expected_symbol

    def test_agent_numbering(self, renderer):
        """Test that agents are numbered with colored square emojis."""
        # Test agents 0-9 get colored squares
        agent_squares = ["🟦", "🟧", "🟩", "🟨", "🟪", "🟥", "🟫", "⬛", "🟦", "🟧"]

        for i in range(10):
            obj = {"type": 0, "agent_id": i, "r": 0, "c": 0}
            symbol = renderer._symbol_for(obj)
            assert symbol == agent_squares[i]

        # Test agents >= 10 use default robot emoji
        obj = {"type": 0, "agent_id": 10, "r": 0, "c": 0}
        symbol = renderer._symbol_for(obj)
        assert symbol == "🤖"

    def test_compute_bounds_with_walls(self, renderer):
        """Test bounds computation based on wall positions."""
        grid_objects = {
            0: {"type": 1, "r": 0, "c": 0},  # wall at (0, 0)
            1: {"type": 1, "r": 0, "c": 5},  # wall at (0, 5)
            2: {"type": 1, "r": 3, "c": 0},  # wall at (3, 0)
            3: {"type": 1, "r": 3, "c": 5},  # wall at (3, 5)
            4: {"type": 0, "r": 1, "c": 2},  # agent at (1, 2)
        }

        renderer._compute_bounds(grid_objects)

        assert renderer._bounds_set is True
        assert renderer._min_row == 0
        assert renderer._min_col == 0
        assert renderer._height == 4  # rows 0-3
        assert renderer._width == 6  # cols 0-5

    def test_render_simple_grid(self, renderer):
        """Test rendering a simple grid with various objects."""
        grid_objects = {
            0: {"type": 1, "r": 0, "c": 0},  # wall
            1: {"type": 1, "r": 0, "c": 2},  # wall
            2: {"type": 0, "r": 1, "c": 1, "agent_id": 0},  # agent 0
            3: {"type": 2, "r": 2, "c": 1},  # altar
            4: {"type": 1, "r": 2, "c": 0},  # wall
            5: {"type": 1, "r": 2, "c": 2},  # wall
        }

        # Render without terminal output
        with suppress_stdout():
            output = renderer.render(step=10, grid_objects=grid_objects)

        # Check that output contains expected emojis
        assert "⬛" in output  # walls
        assert "🟦" in output  # agent 0 (blue square)
        assert "🎯" in output  # altar (target)
        assert "⬜" in output  # empty spaces

    def test_render_with_special_objects(self, game_config):
        """Test rendering with the special objects from debug maps."""
        # Update object type names to include special types
        object_type_names = ["agent", "wall", "altar", "lasery", "marker", "block"]
        renderer = MiniscopeRenderer(object_type_names, game_config, map_height=10, map_width=10)

        grid_objects = {
            0: {"type": 1, "r": 0, "c": 0},  # wall
            1: {"type": 0, "r": 1, "c": 1, "agent_id": 0},  # agent
            2: {"type": 3, "r": 1, "c": 2},  # lasery (L)
            3: {"type": 4, "r": 2, "c": 1},  # marker (m)
            4: {"type": 5, "r": 2, "c": 2},  # block (s)
            5: {"type": 1, "r": 3, "c": 3},  # wall (for bounds)
        }

        with suppress_stdout():
            output = renderer.render(step=5, grid_objects=grid_objects)

        # Check for special emoji
        assert "🟥" in output  # lasery
        assert "🟠" in output  # marker
        assert "📦" in output  # block

    def test_render_caching(self, renderer):
        """Test that renderer caches output when grid doesn't change."""
        grid_objects = {
            0: {"type": 1, "r": 0, "c": 0},  # wall
            1: {"type": 1, "r": 0, "c": 2},  # wall
            2: {"type": 1, "r": 2, "c": 0},  # wall
            3: {"type": 1, "r": 2, "c": 2},  # wall
            4: {"type": 0, "r": 1, "c": 1, "agent_id": 0},  # agent
        }

        # First render
        with suppress_stdout():
            renderer.render(step=1, grid_objects=grid_objects)
        first_buffer = renderer._last_buffer

        # Second render with same grid
        with suppress_stdout():
            renderer.render(step=2, grid_objects=grid_objects)
        second_buffer = renderer._last_buffer

        # Buffer should be the same since grid hasn't changed
        assert first_buffer == second_buffer

        # Third render with different grid - move agent within bounds
        grid_objects[4]["c"] = 0  # Move agent to different column
        with suppress_stdout():
            renderer.render(step=3, grid_objects=grid_objects)
        third_buffer = renderer._last_buffer

        # Buffer should be different since grid changed
        assert second_buffer != third_buffer

    def test_get_buffer_path(self, renderer):
        """Exercise get_buffer to cover non-printing code path."""
        grid_objects = {
            0: {"type": 1, "r": 0, "c": 0},  # wall
            1: {"type": 0, "r": 1, "c": 1, "agent_id": 0},  # agent 0
        }

        # Ensure bounds are computed and buffer is returned
        buf = renderer.get_buffer(grid_objects)
        assert isinstance(buf, str)
        assert "⬛" in buf or "🤖" in buf or "⬜" in buf

    def test_empty_grid(self, renderer):
        """Test rendering an empty grid."""
        grid_objects = {}

        # Should not crash
        renderer.render(step=0, grid_objects=grid_objects)

        # Should have the dimensions passed to __init__
        assert renderer._height == 10
        assert renderer._width == 10

    def test_unknown_object_fallback(self, renderer):
        """Test that unknown objects render with the fallback symbol."""
        # Mock an object with an unknown type name
        renderer._object_type_names.append("unknown_type")

        grid_objects = {
            0: {"type": len(renderer._object_type_names) - 1, "r": 0, "c": 0},  # Use the unknown type
        }

        symbol = renderer._symbol_for(grid_objects[0])
        assert symbol == "❓"  # Fallback symbol

    def test_viewport_arrows_at_edges(self, renderer):
        """Test that viewport arrows appear when there's more content beyond viewport."""
        # Create a large grid (10x10)
        grid_objects = {}
        obj_id = 0
        for r in range(10):
            for c in range(10):
                grid_objects[obj_id] = {"type": 1, "r": r, "c": c}  # walls everywhere
                obj_id += 1

        # Render with a small viewport centered at (5, 5) showing only 3x3
        with suppress_stdout():
            buffer = renderer._build_buffer(
                grid_objects, viewport_center_row=5, viewport_center_col=5, viewport_height=3, viewport_width=3
            )

        # Should have arrows on all edges since viewport is in the middle
        lines = buffer.split("\n")
        assert len(lines) == 3

        # Top row should have up arrows
        assert "🔼" in lines[0]

        # Bottom row should have down arrows
        assert "🔽" in lines[2]

        # Left edge should have left arrows
        assert "◀" in lines[1][0:2]  # Check first 2 chars for arrow (may have variation selector)

        # Right edge should have right arrows
        assert "▶" in lines[1][-2:]  # Check last 2 chars for arrow (may have variation selector)

    def test_viewport_no_arrows_when_at_edges(self, renderer):
        """Test that no arrows appear when viewport shows the entire map."""
        grid_objects = {
            0: {"type": 1, "r": 0, "c": 0},
            1: {"type": 1, "r": 2, "c": 2},
        }

        # Render without viewport constraints (full map)
        with suppress_stdout():
            buffer = renderer._build_buffer(grid_objects)

        # Should not have arrows since we're showing everything
        assert "🔼" not in buffer
        assert "🔽" not in buffer
        assert "◀" not in buffer
        assert "▶" not in buffer

    def test_info_panel_no_agent_selected(self, renderer):
        """Test info panel when no agent is selected."""

        grid_objects = {}
        panel = renderer._build_info_panel(grid_objects, None, [], 10, np.array([]))

        # Panel is now a rich.Table, so render it to text to check content
        console = Console()
        with console.capture() as capture:
            console.print(panel)
        panel_text = capture.get()

        assert "No agent" in panel_text
        assert "selected" in panel_text

    def test_info_panel_with_agent(self, renderer):
        """Test info panel with an agent selected."""

        grid_objects = {
            0: {"type": 0, "r": 0, "c": 0, "agent_id": 1, "inventory": {0: 10, 1: 5}},
        }
        resource_names = ["energy", "hearts"]
        total_rewards = np.array([0.0, 42.5])

        panel = renderer._build_info_panel(grid_objects, 1, resource_names, 15, total_rewards)

        # Panel is now a rich.Table, so render it to text to check content
        console = Console()
        with console.capture() as capture:
            console.print(panel)
        panel_text = capture.get()

        assert "1" in panel_text  # Agent ID
        assert "42.5" in panel_text  # Reward
        assert "energy" in panel_text
        assert "10" in panel_text  # energy amount
        assert "hearts" in panel_text
        assert "5" in panel_text  # hearts amount

    def test_info_panel_with_empty_inventory(self, renderer):
        """Test info panel with an agent that has no inventory."""

        grid_objects = {
            0: {"type": 0, "r": 0, "c": 0, "agent_id": 0, "inventory": {}},
        }
        total_rewards = np.array([10.0])

        panel = renderer._build_info_panel(grid_objects, 0, [], 10, total_rewards)

        # Panel is now a rich.Table, so render it to text to check content
        console = Console()
        with console.capture() as capture:
            console.print(panel)
        panel_text = capture.get()

        assert "(empty)" in panel_text

    def test_bounds_checking_skips_out_of_range_objects(self, renderer):
        """Test that objects outside viewport bounds are skipped during rendering."""
        # Create objects at various positions
        grid_objects = {
            0: {"type": 0, "r": 0, "c": 0, "agent_id": 0},  # Top-left corner
            1: {"type": 1, "r": 5, "c": 5},  # Center
            2: {"type": 1, "r": 9, "c": 9},  # Bottom-right corner
            3: {"type": 1, "r": -1, "c": 0},  # Out of bounds (negative)
            4: {"type": 1, "r": 15, "c": 15},  # Out of bounds (too large)
        }

        # Render with viewport showing only center 3x3 area
        with suppress_stdout():
            buffer = renderer._build_buffer(
                grid_objects, viewport_center_row=5, viewport_center_col=5, viewport_height=3, viewport_width=3
            )

        # Should not crash and should render something
        assert buffer is not None
        assert len(buffer) > 0
