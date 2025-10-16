"""Shared utilities for formatting grid object data in replays and play streams."""

from typing import Callable, Optional, Tuple, Union

import numpy as np


def format_grid_object_base(grid_object: dict) -> dict:
    """Format the base properties common to all grid objects."""
    update_object = {}
    update_object["id"] = grid_object["id"]
    update_object["type_id"] = grid_object["type_id"]
    update_object["location"] = grid_object["location"]
    update_object["orientation"] = grid_object.get("orientation", 0)
    update_object["inventory"] = list(grid_object.get("inventory", {}).items())
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
    decode_flat_action: Optional[Callable[[int], Tuple[int, int]]] = None,
) -> None:
    """Add agent-specific properties to the update object."""
    agent_id = grid_object["agent_id"]
    update_object["agent_id"] = agent_id
    update_object["vision_size"] = 11  # TODO: Waiting for env to support this
    agent_action = np.asarray(actions[agent_id]).reshape(-1)
    action_id = 0
    action_param = 0
    if agent_action.size >= 2:
        action_id = int(agent_action[0])
        action_param = int(agent_action[1])
    elif agent_action.size == 1:
        flat_index = int(agent_action[0])
        if decode_flat_action is not None and flat_index >= 0:
            decoded_action_id, decoded_param = decode_flat_action(flat_index)
            action_id = decoded_action_id
            action_param = decoded_param
        else:
            action_id = flat_index
            action_param = 0
    update_object["action_id"] = action_id
    update_object["action_param"] = action_param
    update_object["action_success"] = bool(env_action_success[agent_id])
    update_object["current_reward"] = rewards[agent_id].item()
    update_object["total_reward"] = total_rewards[agent_id].item()
    update_object["freeze_remaining"] = grid_object.get("freeze_remaining", 0)
    update_object["is_frozen"] = grid_object.get("is_frozen", False)
    update_object["freeze_duration"] = grid_object.get("freeze_duration", 0)
    update_object["group_id"] = grid_object["group_id"]


def format_converter_properties(grid_object: dict, update_object: dict) -> None:
    """Add building/converter-specific properties to the update object."""
    update_object["input_resources"] = list(grid_object.get("input_resources", {}).items())
    update_object["output_resources"] = list(grid_object.get("output_resources", {}).items())
    update_object["output_limit"] = grid_object.get("output_limit", 0)
    update_object["conversion_remaining"] = 0  # TODO: Waiting for env to support this
    update_object["is_converting"] = grid_object.get("is_converting", False)
    update_object["conversion_duration"] = grid_object.get("conversion_duration", 0)
    update_object["cooldown_remaining"] = 0  # TODO: Waiting for env to support this
    update_object["is_cooling_down"] = grid_object.get("is_cooling_down", False)
    update_object["cooldown_duration"] = grid_object.get("cooldown_duration", 0)


def format_assembler_properties(grid_object: dict, update_object: dict) -> None:
    # Assembler properties
    update_object["cooldown_remaining"] = grid_object.get("cooldown_remaining", 0)
    update_object["cooldown_duration"] = grid_object.get("cooldown_duration", 0)
    update_object["is_clipped"] = grid_object.get("is_clipped", False)
    update_object["is_clip_immune"] = grid_object.get("is_clip_immune", False)
    update_object["uses_count"] = grid_object.get("uses_count", 0)
    update_object["max_uses"] = grid_object.get("max_uses", 0)
    update_object["allow_partial_usage"] = grid_object.get("allow_partial_usage", False)

    update_object["recipes"] = []
    for recipe in grid_object.get("recipes", []):
        update_recipe = {}
        update_recipe["inputs"] = list(recipe.get("inputs", {}).items())
        update_recipe["outputs"] = list(recipe.get("outputs", {}).items())
        update_recipe["cooldown"] = recipe["cooldown"]
        update_object["recipes"].append(update_recipe)


def format_grid_object(
    grid_object: dict,
    actions: np.ndarray,
    env_action_success: Union[np.ndarray, list],
    rewards: np.ndarray,
    total_rewards: np.ndarray,
    decode_flat_action: Optional[Callable[[int], Tuple[int, int]]] = None,
) -> dict:
    """Format a grid object with validation for both replay recording and play streaming."""
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
        agent_id = grid_object["agent_id"]
        assert isinstance(agent_id, int), f"Expected agent_id to be an integer, got {type(agent_id)}"
        assert isinstance(grid_object["group_id"], int), (
            f"Expected group_id to be an integer, got {type(grid_object['group_id'])}"
        )

        update_object["is_agent"] = True
        format_agent_properties(
            grid_object,
            update_object,
            actions,
            env_action_success,
            rewards,
            total_rewards,
            decode_flat_action=decode_flat_action,
        )

    elif "input_resources" in grid_object:
        format_converter_properties(grid_object, update_object)

    elif "recipes" in grid_object:
        format_assembler_properties(grid_object, update_object)

    return update_object
