from pathlib import Path

from metta.gridworks.configs.registry import ConfigMakerRegistry


def test_registry():
    registry = ConfigMakerRegistry(root_dir=Path("tests/gridworks/configs/fixtures"))

    assert registry.size() > 0
