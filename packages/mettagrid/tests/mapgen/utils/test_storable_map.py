import textwrap

from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.map_builder import validate_any_map_builder
from mettagrid.mapgen.utils.storable_map import StorableMap

MAP_LINE_STRINGS = [
    "####",
    "#.m#",
    "#_@#",
    "#n.#",
]

MAP_LINES = [list(line) for line in MAP_LINE_STRINGS]

LEGEND = {
    "#": "wall",
    ".": "empty",
    "m": "mine_red",
    "n": "generator_red",
    "_": "altar",
    "@": "agent.agent",
}


def test_serializes_map_from_yaml_string():
    ascii_yaml = textwrap.dedent(
        """
        type: mettagrid.map_builder.ascii.AsciiMapBuilder
        map_data: |-
          ####
          #.m#
          #_@#
          #n.#
        char_to_name_map:
          "#": wall
          ".": empty
          "m": mine_red
          "n": generator_red
          "_": altar
          "@": agent.agent
        """
    )

    config = AsciiMapBuilder.Config.from_str(ascii_yaml)
    assert ["".join(row) for row in config.map_data] == MAP_LINE_STRINGS
    for token, name in LEGEND.items():
        assert config.char_to_name_map[token] == name

    storable_map = StorableMap.from_cfg(
        validate_any_map_builder(
            {
                "type": "mettagrid.map_builder.ascii.AsciiMapBuilder",
                "map_data": config.map_data,
                "char_to_name_map": config.char_to_name_map,
            }
        )
    )

    # Test model_dump representation
    config_dict = storable_map.config.model_dump()
    assert config_dict["map_data"] == MAP_LINES  # model_dump returns nested lists, not strings
    for token, name in LEGEND.items():
        assert config_dict["char_to_name_map"][token] == name

    # Test string representation via model_dump
    config_str = str(storable_map.config.model_dump())
    assert "map_data" in config_str
    assert "char_to_name_map" in config_str
    assert "['#', '#', '#', '#']" in config_str  # nested list format
    assert "['#', '.', 'm', '#']" in config_str
    assert "['#', '_', '@', '#']" in config_str
    assert "['#', 'n', '.', '#']" in config_str

    # Test that the StorableMap object has the expected attributes
    assert hasattr(storable_map, "grid")
    assert hasattr(storable_map, "config")
    assert hasattr(storable_map, "metadata")
    assert storable_map.width() == 4
    assert storable_map.height() == 4
