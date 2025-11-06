import os

import mettagrid.mapgen.scenes.ascii
import mettagrid.test_support.mapgen
import packages.mettagrid.tests.mapgen.scenes.test_utils


def make_yaml_map(map_lines: list[str], legend: dict[str, str]) -> str:
    legend_block = "\n".join(f'  "{token}": {name}' for token, name in legend.items())
    map_block = "\n".join(f"  {line}" for line in map_lines)
    return (
        "type: mettagrid.map_builder.ascii.AsciiMapBuilder.Config\n"
        f"map_data: |-\n{map_block}\nchar_to_name_map:\n{legend_block}\n"
    )


def test_basic():
    uri = f"{os.path.dirname(__file__)}/fixtures/test.map"
    scene = mettagrid.test_support.mapgen.render_scene(mettagrid.mapgen.scenes.ascii.Ascii.Config(uri=uri), (4, 4))

    packages.mettagrid.tests.mapgen.scenes.test_utils.assert_grid(
        scene,
        """
####
#_.#
##.#
####
        """,
    )
