import os
import tempfile

import numpy as np
import pytest
from pydantic import ValidationError

from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.map_builder import GameMap
from mettagrid.mapgen.types import map_grid_dtype


def make_yaml_map(map_lines: list[str], legend: dict[str, str]) -> str:
    legend_block = "\n".join(f'  "{token}": {name}' for token, name in legend.items())
    map_block = "\n".join(f"  {line}" for line in map_lines)
    return (
        "type: mettagrid.map_builder.ascii.AsciiMapBuilder\n"
        f"map_data: |-\n{map_block}\nchar_to_name_map:\n{legend_block}\n"
    )


def write_temp_map(content: str) -> str:
    temp = tempfile.NamedTemporaryFile(mode="w", suffix=".map", delete=False, encoding="utf-8")
    temp.write(content)
    temp.flush()
    temp.close()
    return temp.name


class TestAsciiMapBuilderConfig:
    def test_create(self):
        yaml_content = make_yaml_map(
            [
                "#@#",
            ],
            {"#": "wall", "@": "agent.agent"},
        )

        temp_file = write_temp_map(yaml_content)
        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = config.create()
            assert isinstance(builder, AsciiMapBuilder)
        finally:
            os.unlink(temp_file)


class TestAsciiMapBuilder:
    BASE_LEGEND = {
        "#": "wall",
        "@": "agent.agent",
        ".": "empty",
        "_": "altar",
        "p": "agent.prey",
        "P": "agent.predator",
    }

    def test_build_simple_map(self):
        yaml_content = make_yaml_map(
            [
                "###",
                "#.@",
                "###",
            ],
            self.BASE_LEGEND,
        )

        temp_file = write_temp_map(yaml_content)
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
        yaml_content = make_yaml_map(
            [
                "#####",
                "#@._#",
                "#p.P#",
                "#####",
            ],
            self.BASE_LEGEND,
        )

        temp_file = write_temp_map(yaml_content)
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
        yaml_content = make_yaml_map(
            [
                "#@#",
            ],
            self.BASE_LEGEND,
        )

        temp_file = write_temp_map(yaml_content)
        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = AsciiMapBuilder(config)
            game_map = builder.build()

            expected = np.array([["wall", "agent.agent", "wall"]], dtype=map_grid_dtype)
            assert np.array_equal(game_map.grid, expected)
        finally:
            os.unlink(temp_file)

    def test_config_accepts_string_map_data(self):
        config = AsciiMapBuilder.Config(
            map_data=[list(v) for v in ["#@", "##"]],
            char_to_name_map={"#": "wall", "@": "agent.agent"},
        )
        assert config.map_data == [["#", "@"], ["#", "#"]]

    def test_config_accepts_list_of_strings(self):
        config = AsciiMapBuilder.Config(
            map_data=[list(v) for v in ["##", "@#"]],
            char_to_name_map={"#": "wall", "@": "agent.agent"},
        )
        assert config.map_data == [["#", "#"], ["@", "#"]]

    def test_config_accepts_legend_as_pairs(self):
        config = AsciiMapBuilder.Config(
            map_data=[list(v) for v in ["##"]],
            char_to_name_map={"#": "wall"},
        )
        assert config.char_to_name_map["#"] == "wall"

    def test_build_empty_map(self):
        yaml_content = make_yaml_map(
            [
                ".",
            ],
            {".": "empty"},
        )

        temp_file = write_temp_map(yaml_content)
        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            builder = AsciiMapBuilder(config)
            game_map = builder.build()

            expected = np.array([["empty"]], dtype=map_grid_dtype)
            assert np.array_equal(game_map.grid, expected)
        finally:
            os.unlink(temp_file)

    def test_empty_legend_allowed(self):
        config = AsciiMapBuilder.Config(
            map_data=[list(v) for v in ["#@."]],
            char_to_name_map={},
        )
        assert config.char_to_name_map["#"] == "wall"
        assert config.char_to_name_map["@"] == "agent.agent"
        assert config.char_to_name_map["."] == "empty"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            config = AsciiMapBuilder.Config.from_uri("nonexistent_file.txt")
            AsciiMapBuilder(config)

    def test_from_uri_loads_legend(self):
        yaml_content = make_yaml_map(
            [
                "A",
            ],
            {"A": "legend_name"},
        )

        temp_file = write_temp_map(yaml_content)
        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            assert config.char_to_name_map["A"] == "legend_name"
        finally:
            os.unlink(temp_file)

    def test_global_defaults_merge_into_legend(self):
        yaml_content = make_yaml_map(
            [
                "#@.",
            ],
            {"X": "custom_tile"},
        )

        config = AsciiMapBuilder.Config.from_str(yaml_content)
        assert config.char_to_name_map["#"] == "wall"
        assert config.char_to_name_map["."] == "empty"
        assert config.char_to_name_map["@"] == "agent.agent"
        assert config.char_to_name_map["X"] == "custom_tile"

    def test_global_defaults_cannot_be_overridden(self):
        yaml_content = make_yaml_map(
            [
                "#",
            ],
            {"#": "lava"},
        )

        with pytest.raises(ValidationError, match=r"Cannot override global default mapping for '#'.*"):
            AsciiMapBuilder.Config.from_str(yaml_content)

    def test_from_ascii_map_string(self):
        ascii_yaml = make_yaml_map(
            [
                "#@",
                ".#",
            ],
            self.BASE_LEGEND,
        )
        config = AsciiMapBuilder.Config.from_str(ascii_yaml)
        assert config.map_data == [["#", "@"], [".", "#"]]
        assert config.char_to_name_map["#"] == "wall"

    def test_with_spaces_raises_error(self):
        yaml_content = make_yaml_map(
            [
                "###",
                "# @",
                "###",
            ],
            self.BASE_LEGEND,
        )

        temp_file = write_temp_map(yaml_content)
        try:
            config = AsciiMapBuilder.Config.from_uri(temp_file)
            with pytest.raises(ValueError, match="Unknown character: ' '"):
                AsciiMapBuilder(config)
        finally:
            os.unlink(temp_file)

    def test_irregular_shaped_map(self):
        yaml_content = make_yaml_map(
            [
                "###",
                "#@",
                "#._#",
            ],
            self.BASE_LEGEND,
        )

        temp_file = write_temp_map(yaml_content)
        try:
            with pytest.raises(ValueError, match="All lines in ASCII map must have the same length"):
                AsciiMapBuilder.Config.from_uri(temp_file)
        finally:
            os.unlink(temp_file)

    def test_legend_with_whitespace_value(self):
        yaml_content = make_yaml_map(
            [
                "#@#",
            ],
            {"#": "wall", "@": "invalid value"},
        )

        temp_file = write_temp_map(yaml_content)
        try:
            with pytest.raises(ValueError, match="String should match pattern"):
                AsciiMapBuilder.Config.from_uri(temp_file)
        finally:
            os.unlink(temp_file)

    def test_utf8_encoding(self):
        yaml_content = make_yaml_map(
            [
                "###",
                "#@.",
                "###",
            ],
            self.BASE_LEGEND,
        )

        temp_file = write_temp_map(yaml_content)
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
