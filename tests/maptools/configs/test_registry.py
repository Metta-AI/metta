from pathlib import Path

from softmax.maptools.gridworks.configs.registry import ConfigMakerRegistry


def test_registry():
    registry = ConfigMakerRegistry(root_dir=Path("tests/maptools/gridworks/configs/fixtures"))

    assert registry.size() > 0
