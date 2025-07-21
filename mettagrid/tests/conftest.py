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


# Preset configurations for common test scenarios
class TestConfigPresets:
    """Common preset configurations for different test scenarios."""
    
    @staticmethod
    def combat_config():
        """Config for tests involving combat mechanics."""
        return {
            "inventory_item_names": ["laser", "armor", "heart"],
            "actions": {
                "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
                "get_items": {"enabled": True},
                "put_items": {"enabled": True},
            },
            "agent": {
                "default_resource_limit": 50,
                "freeze_duration": 10,
                "rewards": {
                    "inventory": {
                        "heart": 1.0,
                        "armor": 0.1,
                        "laser": 0.1,
                    }
                }
            }
        }
    
    @staticmethod
    def resource_config():
        """Config for tests involving resource collection and conversion."""
        return {
            "inventory_item_names": ["ore_red", "ore_blue", "battery_red", "battery_blue", "heart"],
            "actions": {
                "get_items": {"enabled": True},
                "put_items": {"enabled": True},
            },
            "agent": {
                "default_resource_limit": 50,
                "rewards": {
                    "inventory": {
                        "ore_red": 0.005,
                        "ore_blue": 0.005,
                        "battery_red": 0.01,
                        "battery_blue": 0.01,
                        "heart": 1.0,
                    }
                }
            },
            "objects": {
                "altar": {
                    "type_id": 8,
                    "output_resources": {"heart": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_resource_count": 1,
                },
                "mine_red": {
                    "type_id": 9,
                    "output_resources": {"ore_red": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 50,
                    "initial_resource_count": 1,
                },
                "generator_red": {
                    "type_id": 10,
                    "input_resources": {"ore_red": 1},
                    "output_resources": {"battery_red": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 25,
                    "initial_resource_count": 1,
                }
            }
        }
    
    @staticmethod
    def movement_config():
        """Config for tests focusing on movement and positioning."""
        return {
            "inventory_item_names": [],
            "actions": {
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "swap": {"enabled": True},
            },
            "objects": {
                "wall": {"type_id": 1, "swappable": False},
                "block": {"type_id": 14, "swappable": True},
            }
        }
    
    @staticmethod
    def full_actions_config():
        """Config with all actions enabled for comprehensive testing."""
        return {
            "inventory_item_names": ["laser", "armor", "ore", "battery", "heart"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "swap": {"enabled": True},
                "change_color": {"enabled": True},
                "change_glyph": {"enabled": True, "number_of_glyphs": 4},
            },
            "agent": {
                "default_resource_limit": 50,
                "rewards": {
                    "inventory": {
                        "heart": 1.0,
                        "battery": 0.1,
                        "ore": 0.05,
                    }
                }
            }
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


def create_test_config(overrides=None, preset=None):
    """
    Create a comprehensive test configuration with sensible defaults.
    
    This factory provides a complete MettaGrid configuration that can be used
    across all tests, reducing boilerplate. Tests can override specific fields
    as needed.
    
    Args:
        overrides: Dictionary of configuration overrides to apply
        preset: Optional preset name ('combat', 'resource', 'movement', 'full_actions') 
                or preset config dict to use as base
        
    Returns:
        Complete test configuration dictionary
    """
    base_config = {
        "name": "TestEnv",
        "report_stats_interval": 100,
        "sampling": 0,
        "game": {
            # Core game settings
            "max_steps": 100,
            "num_agents": 2,
            "obs_width": 7,
            "obs_height": 7,
            "num_observation_tokens": 50,
            
            # Only use inventory_item_names (not inventory_items)
            "inventory_item_names": [
                "ore_red", "ore_blue", "battery_red", "battery_blue", "heart", "armor", "laser"
            ],
            
            # Agent configuration
            "agent": {
                "default_resource_limit": 50,
                "freeze_duration": 10,
                "rewards": {
                    "inventory": {
                        "ore_red": 0.005,
                        "ore_blue": 0.005,
                        "ore_green": 0.005,
                        "battery_red": 0.01,
                        "battery_blue": 0.01,
                        "battery_green": 0.01,
                        "heart": 1.0,
                        "heart_max": None,
                        "battery_red_max": 5,
                        "battery_blue_max": 5,
                        "battery_green_max": 5
                    }
                }
            },
            
            # Groups configuration
            "groups": {
                "agent": {"id": 0, "sprite": 0, "props": {}},
                "red": {"id": 0, "props": {}},  # Alias for compatibility
                "team_1": {"id": 1, "sprite": 1, "group_reward_pct": 0.5, "props": {}},
                "team_2": {"id": 2, "sprite": 4, "group_reward_pct": 0.5, "props": {}},
                "team_3": {"id": 3, "sprite": 8, "group_reward_pct": 0.5, "props": {}},
                "team_4": {"id": 4, "sprite": 1, "group_reward_pct": 0.5, "props": {}},
                "prey": {"id": 5, "sprite": 12, "props": {}},
                "predator": {"id": 6, "sprite": 6, "props": {}}
            },
            
            # Objects configuration
            "objects": {
                "wall": {"type_id": 1, "swappable": False},
                "block": {"type_id": 2, "swappable": True},  # blocks need type_id
                "altar": {
                    "type_id": 3,
                    "input_resources": {"battery_red": 3},
                    "output_resources": {"heart": 1},
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 10,
                    "initial_resource_count": 1
                },
                "mine_red": {
                    "type_id": 4,
                    "output_resources": {"ore_red": 1},
                    "color": 0,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 50,
                    "initial_resource_count": 1
                },
                "mine_blue": {
                    "type_id": 5,
                    "output_resources": {"ore_blue": 1},
                    "color": 1,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 50,
                    "initial_resource_count": 1
                },
                "generator_red": {
                    "type_id": 6,
                    "input_resources": {"ore_red": 1},
                    "output_resources": {"battery_red": 1},
                    "color": 0,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 25,
                    "initial_resource_count": 1
                },
                "generator_blue": {
                    "type_id": 7,
                    "input_resources": {"ore_blue": 1},
                    "output_resources": {"battery_blue": 1},
                    "color": 1,
                    "max_output": 5,
                    "conversion_ticks": 1,
                    "cooldown": 25,
                    "initial_resource_count": 1
                }
            },
            
            # Actions configuration
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "put_items": {"enabled": True},
                "get_items": {"enabled": True},
                "attack": {
                    "enabled": True,
                    "consumed_resources": {"laser": 1},
                    "defense_resources": {"armor": 1}
                },
                "swap": {"enabled": True},
                "change_color": {"enabled": True},
                "change_glyph": {"enabled": True, "number_of_glyphs": 4}
            },
            
            # Map builder configuration
            # Note: most mettagrid tests create their own maps, so we don't include
            # map_builder by default. Tests that need it can add it via overrides.
        }
    }
    
    # Apply preset if provided
    if preset:
        if isinstance(preset, str):
            # Get preset by name
            preset_method = getattr(TestConfigPresets, f"{preset}_config", None)
            if preset_method:
                preset_config = preset_method()
                base_config = merge_configs(base_config, {"game": preset_config})
        elif isinstance(preset, dict):
            # Direct preset config
            base_config = merge_configs(base_config, {"game": preset})
    
    # Apply overrides last
    if overrides:
        return merge_configs(base_config, overrides)
    return base_config


def create_minimal_test_config(overrides=None, preset=None):
    """
    Create a minimal test configuration for simple tests.
    Uses smaller observation space and fewer features.
    
    Args:
        overrides: Dict of values to override in the config
        preset: Optional preset name or preset config dict to use as base
    
    Returns:
        A minimal configuration dict suitable for simple tests
    """
    minimal_config = {
        "max_steps": 10,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 20,
        "inventory_item_names": [],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": False},
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False, "number_of_glyphs": 0},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {"wall": {"type_id": 1}},
        "agent": {
            "default_resource_limit": 10,
            "rewards": {},
        },
    }
    
    # Apply preset if provided
    if preset:
        if isinstance(preset, str):
            # Get preset by name
            preset_method = getattr(TestConfigPresets, f"{preset}_config", None)
            if preset_method:
                preset_config = preset_method()
                minimal_config = merge_configs(minimal_config, preset_config)
        elif isinstance(preset, dict):
            # Direct preset config
            minimal_config = merge_configs(minimal_config, preset)
    
    # Apply overrides last
    if overrides:
        minimal_config = merge_configs(minimal_config, overrides)
    
    return {"game": minimal_config}


def create_test_config_from_preset(preset_name, overrides=None):
    """
    Create a test configuration from a named preset.
    
    Args:
        preset_name: Name of the preset ('combat', 'resource', 'movement', 'full_actions')
        overrides: Additional overrides to apply on top of the preset
        
    Returns:
        Complete test configuration with preset applied
    """
    return create_test_config(overrides=overrides, preset=preset_name)


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
