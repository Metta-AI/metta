"""Shared utilities for formatting grid object data in replays and live streams."""

from typing import Union

import numpy as np


def inventory_format(inventory: dict) -> list:
    """Convert inventory dict to list format expected by frontend.

    Args:
        inventory: Dict mapping item_id -> amount

    Returns:
        List of [item_id, amount] pairs
    """
    result = []
    for item_id, amount in inventory.items():
        result.append([item_id, amount])
    return result


def format_grid_object_base(grid_object: dict) -> dict:
    """Format the base properties common to all grid objects.

    Args:
        grid_object: Raw grid object from environment

    Returns:
        Dict with formatted base properties
    """
    update_object = {}
    update_object["id"] = grid_object["id"]
    update_object["type_id"] = grid_object["type_id"]
    update_object["location"] = grid_object["location"]
    update_object["orientation"] = grid_object.get("orientation", 0)
    update_object["inventory"] = inventory_format(grid_object.get("inventory", {}))
    update_object["inventory_max"] = grid_object.get("inventory_max", 0)
    update_object["color"] = grid_object.get("color", 0)
    update_object["is_swappable"] = grid_object.get("is_swappable", False)
    return update_object


def format_agent_properties(
    grid_object: dict,
    update_object: dict,
    actions: np.ndarray,
    env_action_success: Union[np.ndarray, list],
    rewards: np.ndarray,
    total_rewards: np.ndarray,
) -> None:
    """Add agent-specific properties to the update object.

    Args:
        grid_object: Raw grid object from environment
        update_object: Dict to update with agent properties
        actions: Array of actions for all agents
        env_action_success: Array or list of action success flags
        rewards: Array of current step rewards
        total_rewards: Array of cumulative rewards
    """
    agent_id = grid_object["agent_id"]
    update_object["agent_id"] = agent_id
    update_object["vision_size"] = 11  # TODO: Waiting for env to support this
    update_object["action_id"] = int(actions[agent_id][0])
    update_object["action_param"] = int(actions[agent_id][1])
    update_object["action_success"] = bool(env_action_success[agent_id])
    update_object["current_reward"] = rewards[agent_id].item()
    update_object["total_reward"] = total_rewards[agent_id].item()
    update_object["freeze_remaining"] = grid_object.get("freeze_remaining", 0)
    update_object["is_frozen"] = grid_object.get("is_frozen", False)
    update_object["freeze_duration"] = grid_object.get("freeze_duration", 0)
    update_object["group_id"] = grid_object["group_id"]


def format_building_properties(grid_object: dict, update_object: dict) -> None:
    """Add building/converter-specific properties to the update object.

    Args:
        grid_object: Raw grid object from environment
        update_object: Dict to update with building properties
    """
    update_object["input_resources"] = inventory_format(grid_object.get("input_resources", {}))
    update_object["output_resources"] = inventory_format(grid_object.get("output_resources", {}))
    update_object["output_limit"] = grid_object.get("output_limit", 0)
    update_object["conversion_remaining"] = 0  # TODO: Waiting for env to support this
    update_object["is_converting"] = grid_object.get("is_converting", False)
    update_object["conversion_duration"] = grid_object.get("conversion_duration", 0)
    update_object["cooldown_remaining"] = 0  # TODO: Waiting for env to support this
    update_object["is_cooling_down"] = grid_object.get("is_cooling_down", False)
    update_object["cooldown_duration"] = grid_object.get("cooldown_duration", 0)


def format_grid_object(
    grid_object: dict,
    actions: np.ndarray,
    env_action_success: Union[np.ndarray, list],
    rewards: np.ndarray,
    total_rewards: np.ndarray,
) -> dict:
    """Format a grid object with validation for both replay recording and live streaming.

    Args:
        grid_object: Raw grid object from environment
        actions: Array of actions for all agents
        env_action_success: Array or list of action success flags
        rewards: Array of current step rewards
        total_rewards: Array of cumulative rewards

    Returns:
        Formatted grid object dict with all necessary properties
    """
    # Validate basic object properties
    assert isinstance(grid_object["id"], int), (
        f"Expected grid_object['id'] to be an integer, got {type(grid_object['id'])}"
    )
    assert isinstance(grid_object["type_id"], int), (
        f"Expected grid_object['type_id'] to be an integer, got {type(grid_object['type_id'])}"
    )
    assert isinstance(grid_object["location"], (tuple, list)) and len(grid_object["location"]) == 3, (
        f"Expected location to be tuple/list of 3 elements, got {type(grid_object['location'])}"
    )
    assert all(isinstance(coord, (int, float)) for coord in grid_object["location"]), (
        "Expected all location coordinates to be numbers"
    )

    update_object = format_grid_object_base(grid_object)

    if "agent_id" in grid_object:
        # Add agent-specific validation
        agent_id = grid_object["agent_id"]
        assert isinstance(agent_id, int), f"Expected agent_id to be an integer, got {type(agent_id)}"
        assert isinstance(grid_object["group_id"], int), (
            f"Expected group_id to be an integer, got {type(grid_object['group_id'])}"
        )

        update_object["is_agent"] = True
        format_agent_properties(grid_object, update_object, actions, env_action_success, rewards, total_rewards)

    elif "input_resources" in grid_object:
        format_building_properties(grid_object, update_object)

    return update_object
