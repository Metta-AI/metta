import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.test_support import TestEnvironmentBuilder


@pytest.fixture
def basic_env() -> MettaGrid:
    """Create a basic test environment."""
    builder = TestEnvironmentBuilder()
    game_map = builder.create_basic_grid()
    game_map = builder.place_agents(game_map, [(1, 1), (2, 4)])
    return builder.create_environment(game_map, obs_width=3, obs_height=3, num_observation_tokens=100)


@pytest.fixture
def adjacent_agents_env() -> MettaGrid:
    """Create an environment with adjacent agents."""
    builder = TestEnvironmentBuilder()
    game_map = builder.create_basic_grid(5, 5)
    game_map = builder.place_agents(game_map, [(2, 1), (2, 2)])
    return builder.create_environment(game_map, obs_width=3, obs_height=3, num_observation_tokens=50)
