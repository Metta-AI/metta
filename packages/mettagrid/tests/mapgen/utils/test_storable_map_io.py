from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.storable_map import StorableMap


def test_simple_map():
    storable_map = StorableMap.from_cfg(
        AsciiMapBuilder.Config(
            map_data=[[".", "#"], ["#", "."]],
        ),
    )
    assert storable_map.width() == 2
    assert storable_map.height() == 2
