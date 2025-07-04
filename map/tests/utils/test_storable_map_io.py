import numpy as np
from omegaconf import DictConfig

from metta.map.utils.storable_map import StorableMap
from metta.mettagrid.util import file as file_utils


def simple_map():
    grid = np.array([["empty", "wall"], ["wall", "empty"]], dtype="<U50")
    return StorableMap(grid, metadata={}, config=DictConfig({}))


def test_save_and_load_local(tmp_path):
    path = tmp_path / "map.yaml"
    m = simple_map()
    m.save(str(path))

    loaded = StorableMap.from_uri(str(path))
    assert np.array_equal(loaded.grid, m.grid)


def test_save_s3_uses_file_utils(monkeypatch):
    calls = []

    def fake_write_data(uri, data, content_type="application/octet-stream"):
        calls.append((uri, data, content_type))

    monkeypatch.setattr(file_utils, "write_data", fake_write_data)

    m = simple_map()
    m.save("s3://bucket/key")

    assert calls == [("s3://bucket/key", str(m), "text/plain")]
