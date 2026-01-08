import random
from typing import Any, Optional

import numpy as np

from mettagrid.config.mettagrid_config import Direction
from mettagrid.mettagrid_c import MettaGrid, dtype_actions
from mettagrid.simulator import Action, Simulation
from mettagrid.test_support.orientation import Orientation


def generate_valid_random_actions(
    sim: Simulation,
    num_agents: int,
    action_prefix: Optional[str] = None,
    seed: Optional[int] = None,
) -> list[Action]:
    # Set the random seed if provided (for deterministic behavior)
    random.seed(seed)
    action_prefix = action_prefix or ""
    action_names = [name for name in sim.action_names if name.startswith(action_prefix)]
    return [Action(name=random.choice(action_names)) for _ in range(num_agents)]


def move(sim: Simulation, direction: Orientation | Direction, agent_idx: int = 0) -> dict[str, Any]:
    """
    Movement helper supporting all 8 cardinal directions.

    Args:
        sim: Simulation
        direction: Orientation enum or direction string
        agent_idx: Agent index (default 0)

    Returns:
        Dict with success status and error if any
    """
    result = {"success": False, "error": None}

    # Convert Orientation to direction string if needed
    if isinstance(direction, Orientation):
        direction_str = direction.name.lower()
    else:
        direction_str = direction

    # Get position before move
    position_before = get_agent_position(sim, agent_idx)

    # Create and execute move action
    action_name = f"move_{direction_str}"
    if action_name not in sim.action_names:
        result["error"] = f"Action {action_name} not available"
        return result

    action = Action(name=action_name)

    # Set action for all agents (only the specified agent will move, others get noop)
    for i in range(sim.num_agents):
        if i == agent_idx:
            sim.agent(i).set_action(action)
        else:
            # Set noop for other agents
            sim.agent(i).set_action(Action(name="noop"))

    sim.step()

    # Check if position changed
    position_after = get_agent_position(sim, agent_idx)
    if position_after != position_before:
        result["success"] = True
    else:
        result["error"] = "Position unchanged (likely blocked)"

    return result


def noop(env: Simulation | MettaGrid, agent_idx: int = 0) -> dict[str, Any]:
    """
    Perform a no-operation action.

    Args:
        env: Simulation or MettaGrid environment
        agent_idx: Agent index (default 0)

    Returns:
        Dict with success status
    """
    result = {"success": False, "error": None}

    # Handle both Simulation (property) and MettaGrid (method)
    if isinstance(env, Simulation):
        action_names = env.action_names
    else:
        action_names = env.action_names() if callable(env.action_names) else env.action_names

    if "noop" not in action_names:
        result["error"] = "Noop action not available"
        return result

    # Handle Simulation vs MettaGrid
    if isinstance(env, Simulation):
        action = Action(name="noop")
        for i in range(env.num_agents):
            env.agent(i).set_action(action)
        env.step()
        action_success = env.action_success
    else:
        noop_idx = action_names.index("noop")
        c_env = env._c_sim if hasattr(env, "_c_sim") else env
        c_env.actions()[:, 0] = 0
        c_env.actions()[agent_idx, 0] = noop_idx
        if hasattr(env, "step") and hasattr(env, "_c_sim"):
            env.step()
        else:
            c_env.step()
        action_success = c_env.action_success if not callable(c_env.action_success) else c_env.action_success()

    result["success"] = bool(action_success[agent_idx])
    if not result["success"]:
        result["error"] = "Noop action failed"

    return result


def attack(env: Simulation | MettaGrid, target_arg: int = 0, agent_idx: int = 0) -> dict[str, Any]:
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
        env: Simulation or MettaGrid environment
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

    # Handle both Simulation (property) and MettaGrid (method)
    if isinstance(env, Simulation):
        action_names = env.action_names
    else:
        action_names = env.action_names() if callable(env.action_names) else env.action_names

    attack_variants = sorted(
        (name for name in action_names if name.startswith("attack_") and name.removeprefix("attack_").isdigit()),
        key=lambda n: int(n.split("_", maxsplit=1)[1]),
    )

    attack_name: Optional[str] = None
    selected_arg: Optional[int] = None

    if attack_variants:
        candidate_arg = max(0, min(target_arg, len(attack_variants) - 1))
        attack_name = attack_variants[candidate_arg]
        selected_arg = candidate_arg
    elif "attack" in action_names:
        attack_name = "attack"
    elif "attack_nearest" in action_names:
        attack_name = "attack_nearest"

    if attack_name is None:
        result["error"] = "Attack action not available"
        return result

    # Get initial state for comparison
    objects_before = env.grid_objects()

    # Get attacker's resources before attack
    attacker_resources_before = {}
    for _obj_id, obj_data in objects_before.items():
        if obj_data.get("agent_id") == agent_idx:
            attacker_resources_before = obj_data.get("resources", {}).copy()
            break

    # Perform attack - handle Simulation vs MettaGrid
    if isinstance(env, Simulation):
        action = Action(name=attack_name)
        for i in range(env.num_agents):
            if i == agent_idx:
                env.agent(i).set_action(action)
            else:
                env.agent(i).set_action(Action(name="noop"))
        env.step()
        action_success = env.action_success
    else:
        attack_idx = action_names.index(attack_name)
        attack_action = np.zeros((env.num_agents,), dtype=dtype_actions)
        attack_action[agent_idx] = attack_idx
        env.step(attack_action)
        action_success = env.action_success if not callable(env.action_success) else env.action_success()

    result["success"] = bool(action_success[agent_idx])

    if selected_arg is not None:
        result["target_arg"] = selected_arg

    if result["success"]:
        # Analyze the results
        objects_after = env.grid_objects()

        # Find which agent was affected
        for obj_id, obj_data in objects_after.items():
            if obj_data.get("type_name") == "agent":
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


def get_current_observation(env: MettaGrid, agent_idx: int):
    """Get current observation using noop action."""
    try:
        # Try to get action_names from Simulation if available
        if hasattr(env, "_sim"):
            action_names = env._sim.action_names
        elif hasattr(env, "action_names"):
            action_names = env.action_names
            if callable(action_names):
                action_names = action_names()
        else:
            raise AttributeError("Cannot determine action_names")

        if "noop" in action_names:
            noop_idx = action_names.index("noop")
            noop_action = np.zeros((env.num_agents,), dtype=dtype_actions)
            noop_action[agent_idx] = noop_idx
            obs, _, _, _, _ = env.step(noop_action)
            return obs.copy()
        else:
            # If no noop, just reset and return observation
            obs, _ = env.reset()
            return obs.copy()
    except Exception:
        obs, _ = env.reset()
        return obs.copy()


def get_agent_position(env: Simulation | MettaGrid, agent_idx: int = 0) -> tuple[int, int]:
    grid_objects = env.grid_objects() if isinstance(env, Simulation) else env.grid_objects()
    for _obj_id, obj_data in grid_objects.items():
        if "agent_id" in obj_data and obj_data.get("agent_id") == agent_idx:
            return (obj_data["r"], obj_data["c"])
    raise ValueError(f"Agent {agent_idx} not found in grid objects")


def get_agent_orientation(env: Simulation | MettaGrid, agent_idx: int = 0) -> int:
    grid_objects = env.grid_objects()
    for _obj_id, obj_data in grid_objects.items():
        if "agent_id" in obj_data and obj_data.get("agent_id") == agent_idx:
            return obj_data["agent:orientation"]
    raise ValueError(f"Agent {agent_idx} not found in grid objects")


def action_index(env, base: str, orientation: Orientation | None = None) -> int:
    """Return the flattened action index for a given action name."""
    target = base if orientation is None else f"{base}_{orientation.name.lower()}"

    # Try to get action_names from Simulation if available
    if isinstance(env, Simulation):
        names = env.action_names
    elif hasattr(env, "_sim"):
        names = env._sim.action_names
    else:
        names_getter = getattr(env, "action_names", None)
        if callable(names_getter):
            names = names_getter()
        else:
            names = names_getter

    if names is None:
        raise AttributeError("Cannot determine action_names for this environment")
    if target not in names:
        raise AssertionError(f"Action {target} not available; available actions: {names}")
    return names.index(target)
