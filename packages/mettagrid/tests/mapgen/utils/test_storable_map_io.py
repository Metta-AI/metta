import numpy as np

from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.mapgen.utils.storable_map import StorableMap


def simple_map():
    storable_map = StorableMap.from_cfg(
        AsciiMapBuilder.Config(
            map_data=[[".", "#"], ["#", "."]],
        ),
    )
    return storable_map


def test_save_and_load_local(tmp_path):
    path = tmp_path / "map.yaml"
    m = simple_map()
    m.save(str(path))

    loaded = StorableMap.from_uri(str(path))
    assert np.array_equal(loaded.grid, m.grid)
