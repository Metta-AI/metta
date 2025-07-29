"""Shared test fixtures for metta tests."""

import pytest


def docker_client_fixture():
    """Factory function that creates the docker_client fixture."""

    @pytest.fixture(scope="class")
    def docker_client():
        try:
            import docker
            from docker.errors import DockerException
        except ImportError:
            pytest.skip("Docker is not installed")

        try:
            client = docker.from_env(timeout=5)
            client.ping()
            return client
        except (DockerException, ConnectionError, TimeoutError) as e:
            pytest.skip(f"Docker daemon not available: {e}")

    return docker_client
