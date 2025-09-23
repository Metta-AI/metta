"""Reusable pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from _pytest.fixtures import FixtureRequest


def docker_client_fixture() -> Callable[[FixtureRequest], object]:
    """Return a ``docker_client`` fixture factory.

    The produced fixture attempts to connect to the local Docker daemon. Tests
    are skipped gracefully when Docker is unavailable so suites can run on
    builders without container support.
    """

    @pytest.fixture(scope="class")
    def docker_client(_request: FixtureRequest) -> object:
        try:
            import docker
            from docker.errors import DockerException
        except ImportError as exc:  # pragma: no cover - optional dependency
            pytest.skip(f"Docker is not installed: {exc}")

        try:
            client = docker.from_env(timeout=5)
            client.ping()
            return client
        except (DockerException, ConnectionError, TimeoutError) as exc:  # pragma: no cover - env dependent
            pytest.skip(f"Docker daemon not available: {exc}")

    return docker_client
