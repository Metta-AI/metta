import pytest
from copy import deepcopy


def base_game_config():
    """Return a base game configuration that tests can override as needed."""
    return {
        "max_steps": 10,
        "num_agents": 1,
        "obs_width": 3,  # Most common default from tests
        "obs_height": 3,  # Most common default from tests
        "num_observation_tokens": 100,  # Common default from tests
        "inventory_item_names": ["laser", "armor"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": False},
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False, "number_of_glyphs": 4},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {"wall": {"type_id": 1}},
        "agent": {},
    }


def merge_configs(base_config, override_config):
    """Deep merge override_config into base_config."""
    result = deepcopy(base_config)
    
    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
    
    deep_merge(result, override_config)
    return result


@pytest.fixture
def basic_env():
    """Create a basic test environment."""
    from .test_mettagrid import TestEnvironmentBuilder
    builder = TestEnvironmentBuilder()
    game_map = builder.create_basic_grid()
    game_map = builder.place_agents(game_map, [(1, 1), (2, 4)])
    return builder.create_environment(game_map)


@pytest.fixture
def adjacent_agents_env():
    """Create an environment with adjacent agents."""
    from .test_mettagrid import TestEnvironmentBuilder
    builder = TestEnvironmentBuilder()
    game_map = builder.create_basic_grid(5, 5)
    game_map = builder.place_agents(game_map, [(2, 1), (2, 2)])
    return builder.create_environment(game_map)
