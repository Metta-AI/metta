"""Pytest configuration for curriculum tests."""

import logging
import socket
import time
from contextlib import closing

import pytest

from metta.rl.curriculum.curriculum_server import CurriculumServer

# Suppress expected connection failure warnings and errors during tests
logging.getLogger("metta.rl.curriculum.curriculum_client").setLevel(logging.CRITICAL)
logging.getLogger("metta.rl.curriculum.curriculum_server").setLevel(logging.CRITICAL)


def find_free_port():
    """Find a free port to use for testing."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


@pytest.fixture
def free_port():
    """Fixture that provides a free port for each test."""
    return find_free_port()


@pytest.fixture
def curriculum_server(request):
    """Fixture that manages curriculum server lifecycle with a unique port."""
    server = None

    def _create_server(curriculum, host="127.0.0.1", port=None):
        nonlocal server
        if port is None:
            port = find_free_port()
        server = CurriculumServer(curriculum, host=host, port=port)
        server.start(background=True)
        # Give server time to start
        time.sleep(0.5)
        return server, port

    yield _create_server

    # Cleanup
    if server is not None:
        server.stop()
        # Give server time to fully shut down
        time.sleep(0.2)
