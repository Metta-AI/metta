import numpy as np
import yaml

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


def simple_map():
    storable_map = StorableMap.from_cfg(
        validate_any_map_builder(
            {
                "type": "mettagrid.map_builder.ascii.AsciiMapBuilder",
                "map_data": MAP_LINES,
                "char_to_name_map": LEGEND,
            }
        )
    )
    return storable_map


def test_save_and_load_local(tmp_path):
    path = tmp_path / "map.yaml"
    m = simple_map()
    m.save(str(path))

    frontmatter, _ = path.read_text(encoding="utf-8").split("---\n", 1)
    parsed = yaml.safe_load(frontmatter)
    assert parsed["config"]["map_data"] == MAP_LINE_STRINGS
    saved_char_map = parsed["config"].get("char_to_name_map", {})
    for char, name in LEGEND.items():
        assert saved_char_map.get(char) == name, (
            f"Character '{char}' should map to '{name}' but got '{saved_char_map.get(char)}'"
        )

    loaded = StorableMap.from_uri(str(path), char_to_name=m.char_to_name)
    assert np.array_equal(loaded.grid, m.grid)
    assert loaded.char_to_name == m.char_to_name
