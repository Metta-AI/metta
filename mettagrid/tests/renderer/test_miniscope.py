"""
Tests for the MiniscopeRenderer class.

This module contains unit tests for the MiniscopeRenderer, ensuring proper
emoji rendering, alignment, and functionality for MettaGrid environments.
"""

import io
import sys
from contextlib import contextmanager

import pytest

from metta.mettagrid.renderer.miniscope import MiniscopeRenderer


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
    def renderer(self, object_type_names):
        return MiniscopeRenderer(object_type_names)

    def test_initialization(self, renderer):
        """Test that MiniscopeRenderer initializes correctly."""
        assert renderer._bounds_set is False
        assert renderer._min_row == 0
        assert renderer._min_col == 0
        assert renderer._height == 0
        assert renderer._width == 0
        assert renderer._last_buffer is None

    def test_symbol_mapping(self, renderer):
        """Test that objects map to correct emoji symbols."""
        # Test basic object types
        test_cases = [
            ({"type": 0, "r": 0, "c": 0}, "🤖"),  # agent
            ({"type": 1, "r": 0, "c": 0}, "🧱"),  # wall
            ({"type": 2, "r": 0, "c": 0}, "🎯"),  # altar
            ({"type": 3, "r": 0, "c": 0}, "🔺"),  # mine
            ({"type": 4, "r": 0, "c": 0}, "🔋"),  # generator
            ({"type": 5, "r": 0, "c": 0}, "🔫"),  # lasery
            ({"type": 6, "r": 0, "c": 0}, "📍"),  # marker
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
        assert "🧱" in output  # walls
        assert "🟦" in output  # agent 0 (blue square)
        assert "🎯" in output  # altar (target)
        assert "⬜" in output  # empty spaces

    def test_render_with_special_objects(self, object_type_names):
        """Test rendering with the special objects from debug maps."""
        # Update object type names to include special types
        object_type_names = ["agent", "wall", "altar", "lasery", "marker", "block"]
        renderer = MiniscopeRenderer(object_type_names)

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
        assert "🔫" in output  # lasery
        assert "📍" in output  # marker
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

    def test_empty_grid(self, renderer):
        """Test rendering an empty grid."""
        grid_objects = {}

        # Should not crash
        renderer.render(step=0, grid_objects=grid_objects)

        # Should have minimal dimensions
        assert renderer._height == 1
        assert renderer._width == 1

    def test_unknown_object_fallback(self, renderer):
        """Test that unknown objects render with the fallback symbol."""
        # Mock an object with an unknown type name
        renderer._object_type_names.append("unknown_type")

        grid_objects = {
            0: {"type": len(renderer._object_type_names) - 1, "r": 0, "c": 0},  # Use the unknown type
        }

        symbol = renderer._symbol_for(grid_objects[0])
        assert symbol == "❓"  # Fallback symbol
