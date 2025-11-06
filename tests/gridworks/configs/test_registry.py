import pathlib

import pytest

import metta.gridworks.configs.registry


@pytest.mark.skip(reason="Flaky test")
def test_registry():
    registry = metta.gridworks.configs.registry.ConfigMakerRegistry(
        root_dirs=[pathlib.Path("tests/gridworks/configs/fixtures")]
    )

    assert registry.size() > 0
