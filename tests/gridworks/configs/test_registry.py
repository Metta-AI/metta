from pathlib import Path

from metta.gridworks.configs.registry import ConfigMakerRegistry


@pytest.mark.skip(reason="Flaky test")
def test_registry():
    registry = ConfigMakerRegistry(root_dirs=[Path("tests/gridworks/configs/fixtures")])

    assert registry.size() > 0
