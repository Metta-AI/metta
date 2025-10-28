"""Intentional failing test for CI output verification."""

import pytest


@pytest.mark.ci  # mark to make locating easier
def test_ci_failure_marker() -> None:
    raise AssertionError("intentional python failure for CI visibility")
