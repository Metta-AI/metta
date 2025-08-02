from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from metta.mettagrid import (
    MettaGridEnv,
    dtype_actions,
)
from metta.mettagrid.mettagrid_c import MettaGrid


class Orientation(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __new__(cls, value):
        """Create new Orientation instance."""
        if isinstance(value, str):
            # Handle string initialization like Orientation("up")
            value = value.upper()
            for member in cls:
                if member.name == value:
                    return member
            raise ValueError(f"Invalid orientation string: '{value}'. Valid options: {[m.name.lower() for m in cls]}")

        # Handle integer initialization (internal use)
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __str__(self) -> str:
        """String representation for printing."""
        return self.name.lower()

    @property
    def movement_delta(self) -> tuple[int, int]:
        """Get the (row_delta, col_delta) for this orientation."""
        deltas = {
            Orientation.UP: (-1, 0),
            Orientation.DOWN: (1, 0),
            Orientation.LEFT: (0, -1),
            Orientation.RIGHT: (0, 1),
        }
        return deltas[self]


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
        NumPy array of valid actions with shape (num_agents, 2)
    """
    # Set the random seed if provided (for deterministic behavior)
    if seed is not None:
        np.random.seed(seed)

    # Get the action space parameters
    # For MultiDiscrete, shape[0] gives the number of dimensions (should be 2: action_type and action_arg)
    # and nvec[0] gives the number of possible values for the first dimension (action types)
    action_space = env.single_action_space
    num_actions = action_space.nvec[0]  # Number of action types

    # Get the maximum argument values for each action type
    max_args = env.max_action_args

    # Initialize actions array with correct dtype
    actions = np.zeros((num_agents, 2), dtype=dtype_actions)

    for i in range(num_agents):
        # Determine action type
        if force_action_type is None:
            # Random action type if not forced
            act_type = np.random.randint(0, num_actions)
        else:
            # Use forced action type (ensure it's valid)
            act_type = min(force_action_type, num_actions - 1) if num_actions > 0 else 0

        # Get maximum allowed argument for this action type
        max_allowed = max_args[act_type] if act_type < len(max_args) else 0

        # Determine action argument
        if force_action_arg is None:
            # Random valid argument if not forced
            act_arg = np.random.randint(0, max_allowed + 1) if max_allowed >= 0 else 0
        else:
            # Use forced argument (clamped to valid range)
            act_arg = min(force_action_arg, max_allowed)

        # Set the action values
        actions[i, 0] = act_type
        actions[i, 1] = act_arg

    return actions


def move(env: MettaGrid, orientation: Orientation, agent_idx: int = 0) -> Dict[str, Any]:
    """
    Move agent in specified direction with full validation.

    Args:
        env: MettaGrid environment
        orientation: Orientation enum, string ("up", "down", "left", "right"), or int (0=Up, 1=Down, 2=Left, 3=Right)
        agent_idx: Agent index (default 0)

    Returns:
        Dict with movement results and validation
    """
    direction_name = str(orientation)

    result = {
        "success": False,
        "rotate_success": False,
        "move_success": False,
        "obs_before": None,
        "obs_after": None,
        "position_before": None,
        "position_after": None,
        "orientation_before": None,
        "orientation_after": None,
        "moved": False,
        "moved_correctly": False,
        "obs_changed": False,
        "error": None,
        "direction": direction_name,
        "target_orientation": orientation.value,
    }

    try:
        action_names = env.action_names()

        # Check required actions exist
        if "move" not in action_names:
            result["error"] = "Move action not available"
            return result
        if "rotate" not in action_names:
            result["error"] = "Rotate action not available"
            return result

        move_action_idx = action_names.index("move")
        rotate_action_idx = action_names.index("rotate")

        print(f"Moving agent {agent_idx} {direction_name} (orientation {orientation.value})")

        # Get initial state
        result["obs_before"] = get_current_observation(env, agent_idx)
        result["position_before"] = get_agent_position(env, agent_idx)
        result["orientation_before"] = get_agent_orientation(env, agent_idx)

        print(f"  Before: pos={result['position_before']}, orient={result['orientation_before']}")

        # Step 1: Rotate to face target direction
        rotate_action = np.zeros((env.num_agents, 2), dtype=dtype_actions)
        rotate_action[agent_idx] = [rotate_action_idx, orientation.value]

        env.step(rotate_action)
        rotate_success = env.action_success()
        result["rotate_success"] = bool(rotate_success[agent_idx])

        if not result["rotate_success"]:
            result["error"] = f"Failed to rotate to {direction_name}"
            return result

        # Verify rotation worked
        current_orientation = get_agent_orientation(env, agent_idx)
        if current_orientation != orientation.value:
            result["error"] = f"Rotation failed: expected {orientation.value}, got {current_orientation}"
            return result

        print(f"  Rotated to face {direction_name}")

        # Step 2: Move forward
        move_action = np.zeros((env.num_agents, 2), dtype=dtype_actions)
        move_action[agent_idx] = [move_action_idx, 0]  # Move forward

        obs_after, rewards, terminals, truncations, info = env.step(move_action)
        move_success = env.action_success()
        result["move_success"] = bool(move_success[agent_idx])

        # Get final state
        result["obs_after"] = obs_after.copy()
        result["position_after"] = get_agent_position(env, agent_idx)
        result["orientation_after"] = get_agent_orientation(env, agent_idx)

        print(f"  After: pos={result['position_after']}, orient={result['orientation_after']}")
        print(f"  Move action success: {result['move_success']}")

        # Validate movement
        if result["position_before"] and result["position_after"]:
            result["moved"] = result["position_before"] != result["position_after"]

            if result["moved"]:
                # Check if movement was in correct direction using enum
                dr = result["position_after"][0] - result["position_before"][0]
                dc = result["position_after"][1] - result["position_before"][1]

                expected_dr, expected_dc = orientation.movement_delta
                result["moved_correctly"] = dr == expected_dr and dc == expected_dc

                if result["moved_correctly"]:
                    print(f"  ✅ Moved correctly {direction_name}")
                else:
                    print(f"  ❌ Wrong direction. Expected {orientation.movement_delta}, got ({dr}, {dc})")
            else:
                if result["move_success"]:
                    print("  ⚠️ Move action succeeded but position unchanged (blocked?)")
                else:
                    print("  ❌ No movement - action failed")

        # Validate observation changes
        if result["obs_before"] is not None and result["obs_after"] is not None:
            result["obs_changed"] = not np.array_equal(result["obs_before"][agent_idx], result["obs_after"][agent_idx])

            if result["moved"] and result["obs_changed"]:
                print("  ✅ Observations changed correctly with movement")
            elif result["moved"] and not result["obs_changed"]:
                print("  ⚠️ Agent moved but observations didn't change")
            elif not result["moved"] and result["obs_changed"]:
                print("  ⚠️ Observations changed but agent didn't move")
            else:
                print("  ✅ No movement, no observation change (consistent)")

        # Overall success
        result["success"] = (
            result["rotate_success"] and result["move_success"] and result["moved"] and result["moved_correctly"]
        )

        if not result["success"] and not result["error"]:
            if not result["move_success"]:
                result["error"] = "Move action failed"
            elif not result["moved"]:
                result["error"] = "No movement detected"
            elif not result["moved_correctly"]:
                result["error"] = "Moved in wrong direction"

    except Exception as e:
        result["error"] = f"Exception during move: {str(e)}"

    return result


def rotate(env: MettaGrid, orientation: Orientation, agent_idx: int = 0) -> Dict[str, Any]:
    """
    Rotate agent to face specified direction.

    Args:
        env: MettaGrid environment
        orientation: Orientation enum, string ("up", "down", "left", "right"), or int (0=Up, 1=Down, 2=Left, 3=Right)
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


def get_agent_position(env: MettaGrid, agent_idx: int = 0) -> Optional[tuple]:
    """Get agent's current position (r, c)."""
    try:
        grid_objects = env.grid_objects()
        for _obj_id, obj_data in grid_objects.items():
            if "agent_id" in obj_data and obj_data.get("agent_id") == agent_idx:
                return (obj_data["r"], obj_data["c"])
        return None
    except Exception:
        return None


def get_agent_orientation(env: MettaGrid, agent_idx: int = 0) -> Optional[int]:
    """Get agent's current orientation."""
    try:
        grid_objects = env.grid_objects()
        for _obj_id, obj_data in grid_objects.items():
            if "agent_id" in obj_data and obj_data.get("agent_id") == agent_idx:
                return obj_data.get("agent:orientation", None)
        return None
    except Exception:
        return None
