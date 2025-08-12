import pytest


@pytest.mark.skip(reason="Strict determinism mode (disabling episode desync) not implemented yet")
def test_strict_determinism_mode_placeholder() -> None:
    assert True
