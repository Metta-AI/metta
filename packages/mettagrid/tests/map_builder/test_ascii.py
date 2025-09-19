import os
import tempfile

import numpy as np
import pytest

from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.map_builder import GameMap, map_grid_dtype


class TestAsciiMapBuilderConfig:
    def test_create(self):
        # Create a temporary file for this test
        ascii_content = "#@#"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(ascii_content)
            temp_file = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = config.create()
            assert isinstance(builder, AsciiMapBuilder)
        finally:
            os.unlink(temp_file)


class TestAsciiMapBuilder:
    def test_build_simple_map(self):
        # Create a temporary ASCII map file
        ascii_content = """###
#.@
###"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(ascii_content)
            temp_file = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = config.create()
            game_map = builder.build()

            assert isinstance(game_map, GameMap)
            expected = np.array(
                [["wall", "wall", "wall"], ["wall", "empty", "agent.agent"], ["wall", "wall", "wall"]],
                dtype=map_grid_dtype,
            )

            assert np.array_equal(game_map.grid, expected)
        finally:
            os.unlink(temp_file)

    def test_build_complex_map(self):
        # Test with more complex characters
        ascii_content = """#####
#@._#
#p.P#
#####"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(ascii_content)
            temp_file = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = AsciiMapBuilder(config)
            game_map = builder.build()

            expected = np.array(
                [
                    ["wall", "wall", "wall", "wall", "wall"],
                    ["wall", "agent.agent", "empty", "altar", "wall"],
                    ["wall", "agent.prey", "empty", "agent.predator", "wall"],
                    ["wall", "wall", "wall", "wall", "wall"],
                ],
                dtype=map_grid_dtype,
            )

            assert np.array_equal(game_map.grid, expected)
        finally:
            os.unlink(temp_file)

    def test_build_single_line_map(self):
        ascii_content = "#@#"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(ascii_content)
            temp_file = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = AsciiMapBuilder(config)
            game_map = builder.build()

            expected = np.array([["wall", "agent.agent", "wall"]], dtype=map_grid_dtype)
            assert np.array_equal(game_map.grid, expected)
        finally:
            os.unlink(temp_file)

    def test_build_empty_map(self):
        ascii_content = "."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(ascii_content)
            temp_file = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = AsciiMapBuilder(config)
            game_map = builder.build()

            expected = np.array([["empty"]], dtype=map_grid_dtype)
            assert np.array_equal(game_map.grid, expected)
        finally:
            os.unlink(temp_file)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            config = AsciiMapBuilder.Config.from_uri("nonexistent_file.txt")
            AsciiMapBuilder(config)

    def test_with_aliases(self):
        # Test that aliases work correctly (W for wall, A for agent, etc.)
        ascii_content = """WWW
WA.
WWW"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(ascii_content)
            temp_file = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = AsciiMapBuilder(config)
            game_map = builder.build()

            expected = np.array(
                [["wall", "wall", "wall"], ["wall", "agent.agent", "empty"], ["wall", "wall", "wall"]],
                dtype=map_grid_dtype,
            )

            assert np.array_equal(game_map.grid, expected)
        finally:
            os.unlink(temp_file)

    def test_with_spaces_as_empty(self):
        # Test that spaces are treated as empty
        ascii_content = """###
# @
###"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(ascii_content)
            temp_file = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = AsciiMapBuilder(config)
            game_map = builder.build()

            expected = np.array(
                [["wall", "wall", "wall"], ["wall", "empty", "agent.agent"], ["wall", "wall", "wall"]],
                dtype=map_grid_dtype,
            )

            assert np.array_equal(game_map.grid, expected)
        finally:
            os.unlink(temp_file)

    def test_irregular_shaped_map(self):
        # Test map with lines of different lengths should raise AssertionError
        ascii_content = """###
#@
#._#
###"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(ascii_content)
            temp_file = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            with pytest.raises(AssertionError, match="All lines in ASCII map must have the same length"):
                AsciiMapBuilder(config)
        finally:
            os.unlink(temp_file)

    def test_utf8_encoding(self):
        # Test that UTF-8 encoding works correctly
        ascii_content = "###\n#@.\n###"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(ascii_content)
            temp_file = f.name

        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = AsciiMapBuilder(config)
            game_map = builder.build()

            expected = np.array(
                [["wall", "wall", "wall"], ["wall", "agent.agent", "empty"], ["wall", "wall", "wall"]],
                dtype=map_grid_dtype,
            )

            assert np.array_equal(game_map.grid, expected)
        finally:
            os.unlink(temp_file)
