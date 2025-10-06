from typing import Any, Optional

import numpy as np

from mettagrid import (
    MettaGridEnv,
    dtype_actions,
)
from mettagrid.mettagrid_c import MettaGrid
from mettagrid.test_support.orientation import Orientation


def generate_valid_random_actions(
    env: MettaGridEnv,
    num_agents: int,
    force_action_type: Optional[int] = None,
    force_action_arg: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate valid actions for all agents, respecting maximum argument values.

    Args:
        env: MettaGridEnv instance
        num_agents: Number of agents to generate actions for
        force_action_type: If provided, use this action type for all agents
        force_action_arg: If provided, use this action arg (clamped to valid range) for all agents
        seed: Optional random seed for deterministic action generation

    Returns:
        NumPy array of valid actions with shape (num_agents,)
    """
    # Set the random seed if provided (for deterministic behavior)
    if seed is not None:
        np.random.seed(seed)

    flattened_map = np.asarray(env.flattened_action_map, dtype=np.int32)
    flattened_by_type: dict[int, dict[int, int]] = {}
    for idx, (act_type, act_arg) in enumerate(flattened_map):
        flattened_by_type.setdefault(int(act_type), {})[int(act_arg)] = int(idx)

    num_action_types = len(flattened_by_type)
    max_args = env.max_action_args

    actions = np.zeros((num_agents,), dtype=dtype_actions)

    for i in range(num_agents):
        # Determine action type
        if force_action_type is None:
            act_type = np.random.randint(0, num_action_types) if num_action_types > 0 else 0
        else:
            act_type = min(force_action_type, num_action_types - 1) if num_action_types > 0 else 0

        # Get maximum allowed argument for this action type
        max_allowed = max_args[act_type] if act_type < len(max_args) else 0

        # Determine action argument
        if force_action_arg is None:
            act_arg = np.random.randint(0, max_allowed + 1) if max_allowed >= 0 else 0
        else:
            act_arg = min(force_action_arg, max_allowed)

        action_idx = flattened_by_type.get(act_type, {}).get(act_arg)
        if action_idx is None:
            # Fallback to first available arg for this type
            type_entries = flattened_by_type.get(act_type, {})
            if not type_entries:
                action_idx = 0
            else:
                action_idx = next(iter(type_entries.values()))

        actions[i] = action_idx

    return actions


def move(env: MettaGrid, direction: Orientation, agent_idx: int = 0) -> dict[str, Any]:
    """
    Movement helper supporting all 8 orientation directions.

    Args:
        env: MettaGrid environment
        direction: Orientation direction
        agent_idx: Agent index (default 0)

    Returns:
        Dict with success status and error if any
    """
    result = {"success": False, "error": None}
    action_names = env.action_names()

    if "move" not in action_names:
        result["error"] = "move not available"
        return result

    move_idx = action_names.index("move")

    # Get initial position for verification
    position_before = get_agent_position(env, agent_idx)

    # Orientation values map directly to movement indices
    movement_idx = direction.value

    mapping = np.asarray(env.flattened_action_map, dtype=np.int32)
    matches = np.where((mapping[:, 0] == move_idx) & (mapping[:, 1] == movement_idx))[0]
    if matches.size == 0:
        result["error"] = "move action mapping not found"
        return result

    move_action_index = int(matches[0])

    move_action = np.zeros((env.num_agents,), dtype=dtype_actions)
    move_action[agent_idx] = move_action_index

    env.step(move_action)

    if not env.action_success()[agent_idx]:
        result["error"] = f"Failed to move {str(direction)}"
        return result

    # Check if position changed
    position_after = get_agent_position(env, agent_idx)
    if position_after != position_before:
        result["success"] = True
    else:
        result["error"] = "Position unchanged (likely blocked)"

    return result


def rotate(env: MettaGrid, orientation: Orientation, agent_idx: int = 0) -> dict[str, Any]:
    """
    Rotate agent to face specified direction.

    Args:
        env: MettaGrid environment
        orientation: Orientation enum (NORTH, SOUTH, WEST, EAST for cardinal directions)
        agent_idx: Agent index (default 0)

    Returns:
        Dict with rotation results and validation
    """

    direction_name = str(orientation)

    result = {
        "success": False,
        "action_success": False,
        "orientation_before": None,
        "orientation_after": None,
        "rotated_correctly": False,
        "error": None,
        "direction": direction_name,
        "target_orientation": orientation.value,
    }

    try:
        action_names = env.action_names()

        if "rotate" not in action_names:
            result["error"] = "Rotate action not available"
            return result

        rotate_action_idx = action_names.index("rotate")

        # Get initial orientation
        result["orientation_before"] = get_agent_orientation(env, agent_idx)

        print(f"Rotating agent {agent_idx} to face {direction_name} (orientation {orientation.value})")
        print(f"  Before: {result['orientation_before']}")

        # Perform rotation
        rotate_action = np.zeros((env.num_agents, 2), dtype=dtype_actions)
        rotate_action[agent_idx] = [rotate_action_idx, orientation.value]

        env.step(rotate_action)
        action_success = env.action_success()
        result["action_success"] = bool(action_success[agent_idx])

        # Get final orientation
        result["orientation_after"] = get_agent_orientation(env, agent_idx)

        print(f"  After: {result['orientation_after']}")
        print(f"  Action success: {result['action_success']}")

        # Validate rotation
        result["rotated_correctly"] = result["orientation_after"] == orientation.value

        if result["rotated_correctly"]:
            print(f"  ✅ Rotated correctly to face {direction_name}")
        else:
            print(f"  ❌ Rotation failed. Expected {orientation.value}, got {result['orientation_after']}")

        # Overall success
        result["success"] = result["action_success"] and result["rotated_correctly"]

        if not result["success"] and not result["error"]:
            if not result["action_success"]:
                result["error"] = "Rotate action failed"
            elif not result["rotated_correctly"]:
                result["error"] = f"Failed to rotate to orientation {orientation.value}"

    except Exception as e:
        result["error"] = f"Exception during rotation: {str(e)}"

    return result


def noop(env: MettaGrid, agent_idx: int = 0) -> dict[str, Any]:
    """
    Perform a no-operation action.

    Args:
        env: MettaGrid environment
        agent_idx: Agent index (default 0)

    Returns:
        Dict with success status
    """
    result = {"success": False, "error": None}
    action_names = env.action_names()

    if "noop" not in action_names:
        result["error"] = "Noop action not available"
        return result

    noop_idx = action_names.index("noop")

    # Perform noop
    noop_action = np.zeros((env.num_agents, 2), dtype=dtype_actions)
    noop_action[agent_idx] = [noop_idx, 0]
    env.step(noop_action)

    result["success"] = bool(env.action_success()[agent_idx])
    if not result["success"]:
        result["error"] = "Noop action failed"

    return result


def attack(env: MettaGrid, target_arg: int = 0, agent_idx: int = 0) -> dict[str, Any]:
    """
    Perform an attack action.

    The attack searches for agents in a 3x3 grid in front of the attacker:
    7 6 8  (3 cells forward)
    4 3 5  (2 cells forward)
    1 0 2  (1 cell forward)
      A    (Attacker position)

    The target_arg (0-8) selects which agent to attack based on scan order.
    If target_arg > number of agents found, attacks the last agent found.

    Args:
        env: MettaGrid environment
        target_arg: Which agent to target in the 3x3 grid (0-8, default 0)
        agent_idx: Attacking agent index (default 0)

    Returns:
        Dict with success status, attack details, and any frozen/stolen info
    """
    result = {
        "success": False,
        "error": None,
        "target_arg": target_arg,
        "agent_idx": agent_idx,
        "attack_position": None,
        "target_frozen": False,
        "resources_stolen": {},
        "defense_used": False,
    }

    action_names = env.action_names()

    if "attack" not in action_names:
        result["error"] = "Attack action not available"
        return result

    attack_idx = action_names.index("attack")

    # Get initial state for comparison
    objects_before = env.grid_objects()

    # Get attacker's resources before attack
    attacker_resources_before = {}
    for _obj_id, obj_data in objects_before.items():
        if obj_data.get("agent_id") == agent_idx:
            attacker_resources_before = obj_data.get("resources", {}).copy()
            break

    # Perform attack
    attack_action = np.zeros((env.num_agents, 2), dtype=dtype_actions)
    attack_action[agent_idx] = [attack_idx, target_arg]
    env.step(attack_action)

    result["success"] = bool(env.action_success()[agent_idx])

    if result["success"]:
        # Analyze the results
        objects_after = env.grid_objects()

        # Find which agent was affected
        for obj_id, obj_data in objects_after.items():
            if obj_data.get("type") == 0:  # Agent type
                obj_before = objects_before.get(obj_id, {})

                # Check if this agent was frozen
                freeze_after = obj_data.get("freeze_remaining", 0)
                freeze_before = obj_before.get("freeze_remaining", 0)

                if freeze_after > 0 and freeze_before == 0:
                    # This agent was just frozen
                    result["target_frozen"] = True
                    result["frozen_agent_id"] = obj_id
                    result["freeze_duration"] = freeze_after
                    result["target_position"] = (obj_data["r"], obj_data["c"])

                    # Check for stolen resources
                    target_resources_before = obj_before.get("resources", {})
                    target_resources_after = obj_data.get("resources", {})

                    for item, amount_before in target_resources_before.items():
                        amount_after = target_resources_after.get(item, 0)
                        if amount_after < amount_before:
                            result["resources_stolen"][item] = amount_before - amount_after

                elif freeze_after > 0 and freeze_before > 0:
                    # Attack hit an already frozen target (wasted)
                    result["wasted_on_frozen"] = True
                    result["frozen_agent_id"] = obj_id

        # Check if attacker gained resources
        for _obj_id, obj_data in objects_after.items():
            if obj_data.get("agent_id") == agent_idx:
                attacker_resources_after = obj_data.get("resources", {})
                for item, amount_after in attacker_resources_after.items():
                    amount_before = attacker_resources_before.get(item, 0)
                    if amount_after > amount_before:
                        gain = amount_after - amount_before
                        if item not in result["resources_stolen"]:
                            result["resources_stolen"][item] = 0
                        # Verify this matches what was stolen
                        if result["resources_stolen"][item] == gain:
                            result["resources_gained"] = result.get("resources_gained", {})
                            result["resources_gained"][item] = gain
                break

    else:
        result["error"] = "Attack action failed (no valid target found or blocked)"

    return result


def swap(env: MettaGrid, agent_idx: int = 0) -> dict[str, Any]:
    """
    Perform a swap action with whatever is in front of the agent.

    Args:
        env: MettaGrid environment
        agent_idx: Agent index (default 0)

    Returns:
        Dict with success status and swap details
    """
    result = {
        "success": False,
        "error": None,
        "agent_idx": agent_idx,
        "position_before": None,
        "position_after": None,
    }

    action_names = env.action_names()

    if "swap" not in action_names:
        result["error"] = "Swap action not available"
        return result

    swap_idx = action_names.index("swap")

    # Get initial position
    result["position_before"] = get_agent_position(env, agent_idx)

    # Perform swap
    swap_action = np.zeros((env.num_agents, 2), dtype=dtype_actions)
    swap_action[agent_idx] = [swap_idx, 0]  # Swap argument is typically ignored
    env.step(swap_action)

    result["success"] = bool(env.action_success()[agent_idx])

    if result["success"]:
        result["position_after"] = get_agent_position(env, agent_idx)
        if result["position_after"] == result["position_before"]:
            result["error"] = "Position unchanged after swap"
            result["success"] = False
    else:
        result["error"] = "Swap action failed (target may not be swappable)"

    return result


def get_current_observation(env: MettaGrid, agent_idx: int):
    """Get current observation using noop action."""
    try:
        action_names = env.action_names()
        if "noop" in action_names:
            noop_idx = action_names.index("noop")
            noop_action = np.zeros((env.num_agents, 2), dtype=dtype_actions)
            noop_action[agent_idx] = [noop_idx, 0]
            obs, _, _, _, _ = env.step(noop_action)
            return obs.copy()
        else:
            # If no noop, just reset and return observation
            obs, _ = env.reset()
            return obs.copy()
    except Exception:
        obs, _ = env.reset()
        return obs.copy()


def get_agent_position(env: MettaGrid, agent_idx: int = 0) -> tuple[int, int]:
    grid_objects = env.grid_objects()
    for _obj_id, obj_data in grid_objects.items():
        if "agent_id" in obj_data and obj_data.get("agent_id") == agent_idx:
            return (obj_data["r"], obj_data["c"])
    raise ValueError(f"Agent {agent_idx} not found in grid objects")


def get_agent_orientation(env: MettaGrid, agent_idx: int = 0) -> int:
    grid_objects = env.grid_objects()
    for _obj_id, obj_data in grid_objects.items():
        if "agent_id" in obj_data and obj_data.get("agent_id") == agent_idx:
            return obj_data["agent:orientation"]
    raise ValueError(f"Agent {agent_idx} not found in grid objects")
