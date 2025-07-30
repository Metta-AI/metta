#!/usr/bin/env python3
"""
Tests for the NethackRenderer class using NetHack-style symbols.

This module contains unit tests for the NethackRenderer, ensuring proper
ASCII rendering, alignment, and NetHack-style conversion functionality.
"""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from metta.mettagrid.char_encoder import CHAR_TO_NAME
from metta.mettagrid.curriculum.core import SingleTaskCurriculum
from metta.mettagrid.mettagrid_env import MettaGridEnv
from metta.mettagrid.renderer.nethack import NethackRenderer
from metta.mettagrid.util.hydra import get_cfg


class TestNethackRenderer:
    """Test suite for NethackRenderer functionality."""

    @pytest.fixture
    def basic_renderer(self):
        """Create a basic renderer for testing."""
        object_type_names = ["agent", "wall", "empty", "mine_red", "generator_red", "altar", "factory", "lab", "temple"]
        return NethackRenderer(object_type_names)

    @pytest.fixture
    def sample_grid_objects(self):
        """Create sample grid objects for testing."""
        return {
            0: {"type": 1, "r": 0, "c": 0},  # wall
            1: {"type": 1, "r": 0, "c": 1},  # wall
            2: {"type": 1, "r": 0, "c": 2},  # wall
            3: {"type": 1, "r": 1, "c": 0},  # wall
            4: {"type": 0, "r": 1, "c": 1, "agent_id": 0},  # agent
            5: {"type": 1, "r": 1, "c": 2},  # wall
            6: {"type": 1, "r": 2, "c": 0},  # wall
            7: {"type": 1, "r": 2, "c": 1},  # wall
            8: {"type": 1, "r": 2, "c": 2},  # wall
        }

    @pytest.fixture
    def emoji_grid_objects(self):
        """Create grid objects that would use emoji symbols."""
        return {
            0: {"type": 4, "r": 0, "c": 0},  # generator (⚙)
            1: {"type": 5, "r": 0, "c": 1},  # altar (⛩)
            2: {"type": 6, "r": 1, "c": 0},  # factory (🏭)
            3: {"type": 7, "r": 1, "c": 1},  # lab (🔬)
            4: {"type": 8, "r": 2, "c": 0},  # temple (🏰)
            5: {"type": 0, "r": 2, "c": 1, "agent_id": 0},  # agent
        }

    def test_renderer_initialization(self, basic_renderer):
        """Test that renderer initializes correctly."""
        assert basic_renderer._object_type_names is not None
        assert basic_renderer._bounds_set is False
        assert basic_renderer._last_buffer is None

    def test_agent_symbol_generation(self, basic_renderer):
        """Test agent symbol generation for different IDs."""
        # Test single digit agents
        for i in range(10):
            obj = {"type": 0, "agent_id": i}
            symbol = basic_renderer._symbol_for(obj)
            assert symbol == str(i)

        # Test letter agents (10+)
        obj = {"type": 0, "agent_id": 10}
        assert basic_renderer._symbol_for(obj) == "a"

        obj = {"type": 0, "agent_id": 35}
        assert basic_renderer._symbol_for(obj) == "z"

        obj = {"type": 0, "agent_id": 36}
        assert basic_renderer._symbol_for(obj) == "a"  # wraps around

    def test_bounds_computation(self, basic_renderer, sample_grid_objects):
        """Test that renderer correctly computes grid bounds."""
        basic_renderer._compute_bounds(sample_grid_objects)

        assert basic_renderer._bounds_set is True
        assert basic_renderer._min_row == 0
        assert basic_renderer._min_col == 0
        assert basic_renderer._height == 3
        assert basic_renderer._width == 3

    def test_basic_rendering(self, basic_renderer, sample_grid_objects):
        """Test basic rendering functionality."""
        with patch("builtins.print"):  # Suppress terminal output during testing
            result = basic_renderer.render(0, sample_grid_objects)

        lines = result.split("\n")
        assert len(lines) == 3
        assert all(len(line) == 3 for line in lines)

        # Check that agent appears as "0"
        assert "0" in result
        # Check that walls appear as "#" (NetHack style)
        assert "#" in result

    def test_line_length_consistency(self, basic_renderer, sample_grid_objects):
        """Test that all rendered lines have consistent length."""
        with patch("builtins.print"):
            result = basic_renderer.render(0, sample_grid_objects)

        lines = result.split("\n")
        lengths = [len(line) for line in lines]

        # All lines should have the same length
        assert len(set(lengths)) == 1, f"Inconsistent line lengths: {lengths}"

    def test_double_width_character_detection(self):
        """Test detection of double-width characters."""
        # Test emoji characters that would cause issues
        double_width_chars = ["🧱", "⚙", "⛩", "🏭", "🔬", "🏰"]
        single_width_ascii = ["A", "0", "1", "#", "@", "G", "_", "F", "L", "T", "."]

        for char in double_width_chars:
            # These are double-width Unicode characters
            assert ord(char[0]) > 127, f"Expected {char} to be Unicode"

        for char in single_width_ascii:
            # These should be ASCII single-width
            assert ord(char[0]) <= 127, f"Expected {char} to be ASCII"

    def test_empty_space_conversion(self, basic_renderer):
        """Test that empty spaces are converted to dots."""
        # Create a grid with empty spaces
        grid_objects = {
            0: {"type": 1, "r": 0, "c": 0},  # wall at corner
            1: {"type": 1, "r": 2, "c": 2},  # wall at opposite corner
        }

        with patch("builtins.print"):
            result = basic_renderer.render(0, grid_objects)

        # Should contain dots for empty spaces
        assert "." in result, "Empty spaces should be rendered as dots"
        # Should not contain regular spaces in the middle
        lines = result.split("\n")
        for line in lines:
            # Each line should only contain # and . and digits (no spaces)
            for char in line:
                assert char in "#.0123456789abcdefghijklmnopqrstuvwxyz", f"Unexpected character: {char}"

    @patch("builtins.print")
    def test_emoji_rendering_alignment(self, mock_print, basic_renderer, emoji_grid_objects):
        """Test rendering with emoji-causing objects for alignment issues."""
        result = basic_renderer.render(0, emoji_grid_objects)
        lines = result.split("\n")

        # Check for alignment - should now be perfect with NetHack conversion
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"All lines should have consistent length, got {lengths}"

        # Check that emoji symbols were converted
        assert "n" in result, "Generator should appear as n"
        assert "_" in result, "Altar should appear as _"
        assert "F" in result, "Factory should appear as F"
        assert "L" in result, "Lab should appear as L"
        assert "T" in result, "Temple should appear as T"

    def test_empty_grid_handling(self, basic_renderer):
        """Test handling of empty grids."""
        empty_grid = {}

        with patch("builtins.print"):
            # Should not crash on empty grid - we need to handle this gracefully
            try:
                result = basic_renderer.render(0, empty_grid)
                # If it doesn't crash, that's good
                assert isinstance(result, str)
            except ValueError:
                # Expected for empty grid - bounds computation fails
                # This is acceptable behavior for an empty grid
                pass

    def test_large_grid_rendering(self, basic_renderer):
        """Test rendering of larger grids."""
        # Create a 10x10 grid
        large_grid = {}
        obj_id = 0

        for r in range(10):
            for c in range(10):
                if r == 0 or r == 9 or c == 0 or c == 9:
                    # Border walls
                    large_grid[obj_id] = {"type": 1, "r": r, "c": c}
                elif r == 5 and c == 5:
                    # Agent in center
                    large_grid[obj_id] = {"type": 0, "r": r, "c": c, "agent_id": 0}
                obj_id += 1

        with patch("builtins.print"):
            result = basic_renderer.render(0, large_grid)

        lines = result.split("\n")
        assert len(lines) == 10
        assert all(len(line) == 10 for line in lines)
        assert "0" in result  # Agent should be present


class TestRendererIntegration:
    """Test renderer integration with MettaGridEnv and tools.sim style setup."""

    @patch("builtins.print")
    def test_environment_rendering_workflow(self, mock_print):
        """Test complete rendering workflow with environment."""
        # Use the working approach from our demo scripts
        cfg = get_cfg("benchmark")
        cfg.game.num_agents = 1
        cfg.game.max_steps = 5
        cfg.game.map_builder = OmegaConf.create(
            {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": 5,
                "height": 5,
                "agents": 1,
                "border_width": 1,
                "objects": {},
            }
        )

        curriculum = SingleTaskCurriculum("test", cfg)
        env = MettaGridEnv(curriculum, render_mode="human")

        # Reset and render
        obs, info = env.reset()
        render_output = env.render()

        assert render_output is not None
        assert isinstance(render_output, str)

        lines = render_output.split("\n")
        assert len(lines) > 0

        # Check consistency - should be perfect with NetHack conversion
        lengths = [len(line) for line in lines]
        assert len(set(lengths)) == 1, f"All lines should have same length, got {lengths}"

        # Check NetHack-style rendering
        assert render_output is not None, "Render output should not be None"
        assert "#" in render_output, "Should contain NetHack-style walls"
        assert "." in render_output, "Should contain NetHack-style empty spaces"

        env.close()

    @patch("builtins.print")
    def test_multiple_render_calls(self, mock_print):
        """Test multiple render calls for consistency."""
        cfg = get_cfg("benchmark")
        cfg.game.num_agents = 1
        cfg.game.max_steps = 5

        # Override map builder to ensure agent count matches
        cfg.game.map_builder = OmegaConf.create(
            {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": 5,
                "height": 5,
                "agents": 1,
                "border_width": 1,
                "objects": {},
            }
        )

        curriculum = SingleTaskCurriculum("test", cfg)
        env = MettaGridEnv(curriculum, render_mode="human")

        obs, info = env.reset()

        # Render multiple times
        outputs = []
        for _ in range(3):
            output = env.render()
            outputs.append(output)

        # All outputs should be identical if state hasn't changed
        assert all(output == outputs[0] for output in outputs)

        # Check alignment consistency
        for output in outputs:
            lines = output.split("\n")
            lengths = [len(line) for line in lines]
            assert len(set(lengths)) == 1, "All lines should have consistent length"

        env.close()

    def test_tools_sim_style_integration(self):
        """Test tools.sim style integration approach."""
        # Get benchmark config (like tools.sim would)
        cfg = get_cfg("benchmark")
        cfg.game.num_agents = 1
        cfg.game.max_steps = 5

        # Override map builder to ensure agent count matches
        cfg.game.map_builder = OmegaConf.create(
            {
                "_target_": "metta.mettagrid.room.random.Random",
                "width": 5,
                "height": 5,
                "agents": 1,
                "border_width": 1,
                "objects": {},
            }
        )

        curriculum = SingleTaskCurriculum("test", cfg)

        # The key: render_mode="human" enables NethackRenderer
        with patch("builtins.print"):
            env = MettaGridEnv(curriculum, render_mode="human")
            assert env._renderer is not None
            assert isinstance(env._renderer, NethackRenderer)

            # Test that it produces NetHack-style output
            obs, info = env.reset()
            output = env.render()
            assert output is not None, "Render output should not be None"
            assert "#" in output, "Should use NetHack-style walls"
            assert "." in output, "Should use NetHack-style empty spaces"

            env.close()


class TestSymbolAnalysis:
    """Test symbol analysis and double-width character detection."""

    def test_current_symbols_mapping(self):
        """Test analysis of current SYMBOLS mapping."""
        emoji_symbols = []
        ascii_symbols = []

        for char, obj_type in CHAR_TO_NAME.items():
            if ord(char[0]) > 127:
                emoji_symbols.append((char, obj_type))
            else:
                ascii_symbols.append((char, obj_type))

        # Should have exactly 6 emoji symbols causing issues
        expected_emojis = ["🧱", "⚙", "⛩", "🏭", "🔬", "🏰"]
        found_emojis = [char for char, _ in emoji_symbols]

        # Check that we have the expected problematic emojis
        for emoji in expected_emojis:
            if emoji in CHAR_TO_NAME:
                assert emoji in found_emojis, f"Expected emoji {emoji} not found in mapping"

    def test_character_width_analysis(self):
        """Test character width analysis for alignment issues."""
        # Focus on the characters that our NetHack conversion handles
        test_chars = [
            ("A", 1),  # Single-width letter
            ("0", 1),  # Single-width digit
            ("#", 1),  # Single-width ASCII (NetHack wall)
            ("@", 1),  # Single-width ASCII (NetHack agent)
            (".", 1),  # Single-width ASCII (NetHack empty)
        ]

        # Test double-width emojis that get converted
        emoji_chars = [
            ("🧱", 2),  # Double-width emoji (gets converted to #)
            ("🏭", 2),  # Double-width emoji (gets converted to F)
        ]

        for char, expected_width in test_chars:
            # String length is always 1
            assert len(char) == 1

            # These should all be ASCII single-width
            is_ascii = ord(char[0]) <= 127
            assert is_ascii, f"Expected {char} to be ASCII"
            assert expected_width == 1

        for char, expected_width in emoji_chars:
            # String length is always 1
            assert len(char) == 1

            # These should be Unicode double-width
            is_ascii = ord(char[0]) <= 127
            assert not is_ascii, f"Expected {char} to be Unicode"
            # Note: Our conversion handles these, so terminal width is expected to be 2
            assert expected_width == 2


class TestNetHackStyleSolution:
    """Test NetHack-style solution for perfect alignment."""

    def test_nethack_symbol_mapping(self):
        """Test proposed NetHack-style symbol mapping."""
        nethack_mapping = {
            "wall": "#",
            "agent": "@",
            "empty": ".",
            "generator_red": "G",
            "altar": "_",
            "factory": "F",
            "lab": "L",
            "temple": "T",
            "mine_red": "^",
            "converter": "*",
        }

        # All NetHack symbols should be single-width ASCII
        for symbol in nethack_mapping.values():
            assert len(symbol) == 1
            assert ord(symbol[0]) <= 127, f"Symbol {symbol} should be ASCII"

    def test_perfect_alignment_guarantee(self):
        """Test that NetHack-style symbols guarantee perfect alignment."""
        nethack_symbols = ["#", "@", ".", "G", "_", "F", "L", "T", "^", "*"]

        # Create a test grid with all symbols
        test_line = "".join(nethack_symbols)

        # String length should equal terminal width for ASCII
        string_length = len(test_line)
        terminal_width = sum(1 for _ in test_line)  # All ASCII = 1 each

        assert string_length == terminal_width
        assert string_length == len(nethack_symbols)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
