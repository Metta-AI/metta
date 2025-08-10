import pytest

from .test_mettagrid import TestEnvironmentBuilder


@pytest.fixture
def basic_env():
    """Create a basic test environment."""
    builder = TestEnvironmentBuilder()
    game_map = builder.create_basic_grid()
    game_map = builder.place_agents(game_map, [(1, 1), (2, 4)])
    return builder.create_environment(game_map)


@pytest.fixture
def adjacent_agents_env():
    """Create an environment with adjacent agents."""
    builder = TestEnvironmentBuilder()
    game_map = builder.create_basic_grid(5, 5)
    game_map = builder.place_agents(game_map, [(2, 1), (2, 2)])
    return builder.create_environment(game_map)
