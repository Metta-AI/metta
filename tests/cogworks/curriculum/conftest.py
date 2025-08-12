"""Test fixtures and utilities for curriculum tests."""

import pytest

from metta.rl.env_config import EnvConfig


def create_test_env_config(seed: int = 42, device: str = "cpu", vectorization: str = "serial") -> EnvConfig:
    """Create a test EnvConfig with specified parameters."""
    return EnvConfig(seed=seed, device=device, vectorization=vectorization)


@pytest.fixture
def env_cfg1():
    """Fixture for first test env config."""
    return create_test_env_config(seed=1, device="cpu")


@pytest.fixture
def env_cfg2():
    """Fixture for second test env config."""
    return create_test_env_config(seed=2, device="cuda")


@pytest.fixture
def env_cfg3():
    """Fixture for third test env config."""
    return create_test_env_config(seed=3, device="cpu", vectorization="multiprocessing")
