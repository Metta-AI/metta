"""Test utilities for mettagrid tests."""

from typing import Any, Optional


def make_test_config(
    num_agents: int = 1,
    map: Optional[list[list[str]]] = None,
    actions: Optional[dict[str, dict[str, Any]]] = None,
    **kwargs
) -> dict[str, Any]:
    """Create a test configuration for MettaGrid.
    
    Args:
        num_agents: Number of agents in the environment
        map: 2D list of strings representing the map
        actions: Action configuration overrides
        **kwargs: Additional configuration parameters
    
    Returns:
        Complete configuration dictionary for MettaGrid
    """
    if map is None:
        map = [
            [".", ".", "."],
            [".", "agent.player", "."],
            [".", ".", "."],
        ]
    
    # Default actions configuration
    default_actions = {
        "noop": {"enabled": True},
        "move": {"enabled": True},
        "rotate": {"enabled": True},
        "put_items": {"enabled": True},
        "get_items": {"enabled": True},
        "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
        "swap": {"enabled": True},
        "change_color": {"enabled": False},
        "change_glyph": {"enabled": False, "number_of_glyphs": 4},
    }
    
    # Override with provided actions
    if actions:
        for action_name, action_config in actions.items():
            if action_name in default_actions:
                default_actions[action_name].update(action_config)
            else:
                default_actions[action_name] = action_config
    
    config = {
        "num_agents": num_agents,
        "max_steps": 1000,
        "episode_truncates": False,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        "inventory_item_names": [
            "ore_red",
            "ore_blue",
            "ore_green",
            "battery_red",
            "battery_blue",
            "battery_green",
            "heart",
            "armor",
            "laser",
            "blueprint",
        ],
        "global_obs": {
            "episode_completion_pct": True,
            "last_action": True,
            "last_reward": True,
            "resource_rewards": False,
        },
        "track_movement_metrics": False,
        "recipe_details_obs": False,
        "actions": default_actions,
        "agent": {
            "default_resource_limit": 0,
            "resource_limits": {},
            "freeze_duration": 0,
            "rewards": {
                "inventory": {},
                "stats": {},
            },
            "action_failure_penalty": 0,
        },
        "groups": {
            "player": {
                "id": 0,
                "sprite": None,
                "group_reward_pct": 0,
                "props": {},
            },
            "enemy": {
                "id": 1,
                "sprite": None,
                "group_reward_pct": 0,
                "props": {},
            },
        },
        "objects": {
            "wall": {"type_id": 1, "swappable": False},
            "altar": {
                "type_id": 2,
                "input_resources": {},
                "output_resources": {},
                "max_output": -1,
                "max_conversions": -1,
                "conversion_ticks": 0,
                "cooldown": 0,
                "initial_resource_count": 0,
                "color": 0,
            },
        },
    }
    
    # Apply any additional kwargs (except map which is handled separately)
    for k, v in kwargs.items():
        if k != "map":
            config[k] = v
    
    # Store map separately for MettaGrid constructor
    config["map"] = map
    
    return config