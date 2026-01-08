import os

from mettagrid.mapgen.scenes.ascii import Ascii
from mettagrid.test_support.mapgen import render_scene

from .test_utils import assert_grid


def make_yaml_map(map_lines: list[str], legend: dict[str, str]) -> str:
    legend_block = "\n".join(f'  "{token}": {name}' for token, name in legend.items())
    map_block = "\n".join(f"  {line}" for line in map_lines)
    return (
        "type: mettagrid.map_builder.ascii.AsciiMapBuilder.Config\n"
        f"map_data: |-\n{map_block}\nchar_to_map_name:\n{legend_block}\n"
    )


def test_basic():
    uri = f"{os.path.dirname(__file__)}/fixtures/test.map"
    scene = render_scene(Ascii.Config(uri=uri), (4, 4))

    assert_grid(
        scene,
        """
####
#_.#
##.#
####
        """,
    )
