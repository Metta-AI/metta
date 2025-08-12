"""Test fixtures and utilities for curriculum tests."""

import pytest

from metta.mettagrid.mettagrid_config import ActionConfig, ActionsConfig, AgentConfig, EnvConfig, GameConfig


def create_test_env_config(seed: int = 42, device: str = "cpu", vectorization: str = "serial") -> EnvConfig:
    """Create a test EnvConfig with specified parameters."""
    # Create a minimal GameConfig for testing
    game_config = GameConfig(
        num_agents=2,
        agent=AgentConfig(),
        groups={"default": {"id": 0, "props": AgentConfig()}},
        actions=ActionsConfig(noop=ActionConfig()),
        objects={},
    )
    return EnvConfig(game=game_config)


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
