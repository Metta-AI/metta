"""Compatibility wrapper around ``softmax.lib.test_support``."""

from softmax.lib.test_support import (
    docker_client_fixture,
    isolated_test_schema_uri,
    pytest_collection_modifyitems,
)

__all__ = [
    "docker_client_fixture",
    "isolated_test_schema_uri",
    "pytest_collection_modifyitems",
]
