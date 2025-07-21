# MettagridTest Config Preset System

## Overview

To reduce boilerplate in MettagridEnvironment test configurations, we've introduced a preset system that provides common configurations for different test scenarios. This allows tests to focus on their specific requirements rather than repeating common setup.

## Available Presets

### 1. `combat` - Combat Mechanics Testing
```python
# Includes: laser, armor, heart items
# Actions: attack, get_items, put_items
# Rewards: heart=1.0, armor=0.1, laser=0.1
config = create_minimal_test_config(preset="combat")
```

### 2. `resource` - Resource Collection & Conversion
```python
# Includes: ore_red/blue, battery_red/blue, heart
# Actions: get_items, put_items
# Objects: altar, mine_red, generator_red
# Rewards: ore=0.005, battery=0.01, heart=1.0
config = create_minimal_test_config(preset="resource")
```

### 3. `movement` - Movement & Positioning
```python
# Includes: No inventory items
# Actions: move, rotate, swap
# Objects: wall (non-swappable), block (swappable)
config = create_minimal_test_config(preset="movement")
```

### 4. `full_actions` - All Actions Enabled
```python
# Includes: laser, armor, ore, battery, heart
# Actions: ALL actions enabled
# Rewards: heart=1.0, battery=0.1, ore=0.05
config = create_test_config(preset="full_actions")
```

## Usage Examples

### Basic Usage with Preset
```python
from mettagrid.tests.conftest import create_minimal_test_config

def test_combat_scenario():
    # Use combat preset for a test that needs attack mechanics
    config = create_minimal_test_config(preset="combat")
    game_config = config["game"]
    
    env = MettaGrid(from_mettagrid_config(game_config), game_map, seed)
```

### Preset with Overrides
```python
def test_extended_combat():
    # Start with combat preset and add/modify specific fields
    config = create_minimal_test_config(
        preset="combat",
        overrides={
            "max_steps": 500,  # Longer test
            "num_agents": 4,   # More agents
            "actions": {
                "swap": {"enabled": True},  # Add swap to combat
            },
            "agent": {
                "rewards": {
                    "inventory": {
                        "heart": 2.0,  # Double heart reward
                    }
                }
            }
        }
    )
```

### Using with create_test_config
```python
# For comprehensive configs with all MettaGrid fields
config = create_test_config(preset="resource", overrides={...})

# Or use the helper function
config = create_test_config_from_preset("resource", overrides={...})
```

## Before vs After

### Before (Lots of boilerplate):
```python
game_config = create_minimal_test_config({
    "max_steps": 10,
    "num_agents": 1,
    "obs_width": 3,
    "obs_height": 3,
    "num_observation_tokens": 100,
    "inventory_item_names": ["laser", "armor", "heart"],
    "actions": {
        "noop": {"enabled": True},
        "move": {"enabled": True},
        "rotate": {"enabled": True},
        "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
        "put_items": {"enabled": True},
        "get_items": {"enabled": True},
        "swap": {"enabled": False},
        "change_color": {"enabled": False},
        "change_glyph": {"enabled": False},
    },
    "agent": {
        "default_resource_limit": 50,
        "rewards": {
            "inventory": {
                "heart": 1.0,
                "armor": 0.1,
                "laser": 0.1,
            }
        }
    }
})["game"]
```

### After (Clean and focused):
```python
game_config = create_minimal_test_config(
    preset="combat",
    overrides={
        "max_steps": 10,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 100,
    }
)["game"]
```

## Creating Custom Presets

If you need a new preset for a common test pattern, add it to `TestConfigPresets` in `conftest.py`:

```python
class TestConfigPresets:
    @staticmethod
    def your_custom_config():
        """Config for your specific test scenario."""
        return {
            "inventory_item_names": [...],
            "actions": {...},
            "objects": {...},
            "agent": {...}
        }
```

Then use it like any other preset:
```python
config = create_minimal_test_config(preset="your_custom")
```

## Best Practices

1. **Choose the closest preset** - Start with a preset that matches your test's main focus
2. **Override only what's different** - Let the preset handle common settings
3. **Document preset usage** - Comment why you chose a specific preset if it's not obvious
4. **Create new presets for patterns** - If you find yourself repeating the same overrides, consider making a new preset