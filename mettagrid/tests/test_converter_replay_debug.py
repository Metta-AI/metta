"""Debug test for converter behavior during replays.

This test helps diagnose why get_items might fail on converters during actual gameplay.
"""

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.mettagrid_env import (
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from metta.mettagrid.util.actions import (
    Orientation,
    rotate,
)


def create_realistic_converter_env():
    """Create an environment that matches actual game configs."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "mine_red", ".", "wall"],
        ["wall", ".", "generator_red", ".", "wall"],
        ["wall", ".", "altar", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = {
        "max_steps": 100,
        "num_agents": 1,
        "obs_width": 5,
        "obs_height": 5,
        "num_observation_tokens": 100,
        "inventory_item_names": ["ore_red", "battery_red", "heart"],
        "actions": {
            "noop": {"enabled": True},
            "get_items": {"enabled": True},
            "put_items": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1},
            "mine_red": {
                "type_id": 2,
                "output_resources": {"ore_red": 1},
                "max_output": 5,
                "conversion_ticks": 1,
                "cooldown": 50,
                "initial_resource_count": 0,  # Realistic: starts empty
            },
            "generator_red": {
                "type_id": 3,
                "input_resources": {"ore_red": 1},
                "output_resources": {"battery_red": 1},
                "max_output": 5,
                "conversion_ticks": 1,
                "cooldown": 25,
                "initial_resource_count": 0,  # Realistic: starts empty
            },
            "altar": {
                "type_id": 4,
                "input_resources": {"battery_red": 3},
                "output_resources": {"heart": 1},
                "max_output": 5,
                "conversion_ticks": 1,
                "cooldown": 10,
                "initial_resource_count": 0,  # Realistic: starts empty
            },
        },
        "agent": {
            "default_resource_limit": 50,
            "freeze_duration": 0,
            "rewards": {
                "inventory": {
                    "ore_red": 0.005,
                    "battery_red": 0.01,
                    "heart": 1.0,
                }
            },
        },
    }

    return MettaGrid(from_mettagrid_config(game_config), game_map, 42)


def check_converter_state(env, converter_name):
    """Check the state of a converter in the environment."""
    grid_objects = env.grid_objects()

    for _obj_id, obj in grid_objects.items():
        if obj.get("type_name") == converter_name:
            print(f"\n{converter_name} state:")
            print(f"  Position: ({obj['r']}, {obj['c']})")
            print(f"  Inventory: {obj.get('inventory', {})}")
            # Check for converter-specific features
            if "converter:converting_or_cooling_down" in obj:
                print(f"  Converting/Cooling: {obj['converter:converting_or_cooling_down']}")
            return obj

    return None


def perform_action(env, action_name, action_arg=0):
    """Perform an action and return observation, reward, and success status."""
    action_idx = env.action_names().index(action_name)
    actions = np.array([[action_idx, action_arg]], dtype=dtype_actions)

    obs, rewards, terminals, truncations, info = env.step(actions)
    success = env.action_success()[0]

    return obs, rewards[0], success


def test_converter_production_cycle():
    """Test the full converter production cycle."""
    env = create_realistic_converter_env()

    # Create buffers
    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)

    env.set_buffers(observations, terminals, truncations, rewards)
    env.reset()

    print("=== Initial State ===")
    check_converter_state(env, "mine_red")

    # Try to get from empty mine
    rotate(env, Orientation.RIGHT)  # Face the mine
    obs, reward, success = perform_action(env, "get_items", 0)
    print(f"\nTrying to get from empty mine: success={success}")

    if not success:
        print("✅ Correctly failed - mine is empty initially")

        # Wait for mine to produce
        print("\n=== Waiting for mine to produce ===")
        for step in range(60):  # Mine has 50 tick cooldown
            obs, reward, success = perform_action(env, "noop", 0)

            if step % 10 == 0:
                check_converter_state(env, "mine_red")

        # Try again after waiting
        obs, reward, success = perform_action(env, "get_items", 0)
        print(f"\nTrying to get from mine after waiting: success={success}")

        if success:
            print("✅ Successfully collected ore after mine produced!")
            # Check agent inventory
            grid_objects = env.grid_objects()
            for _obj_id, obj in grid_objects.items():
                if obj.get("type_name") == "agent":
                    print(f"Agent inventory: {obj.get('inventory', {})}")
        else:
            print("❌ Still couldn't collect - mine might need more time or is at max output")
    else:
        print("🤔 Unexpectedly succeeded - check initial_resource_count")


def test_converter_with_initial_resources():
    """Test converter behavior with initial resources (like in some configs)."""
    # Create environment with initial resources
    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "altar", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    game_config = {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 100,
        "inventory_item_names": ["battery_red", "heart"],
        "actions": {
            "noop": {"enabled": True},
            "get_items": {"enabled": True},
            "rotate": {"enabled": True},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1},
            "altar": {
                "type_id": 4,
                "input_resources": {"battery_red": 3},
                "output_resources": {"heart": 1},
                "max_output": 5,
                "conversion_ticks": 1,
                "cooldown": 10,
                "initial_resource_count": 1,  # Starts with 1 heart
            },
        },
        "agent": {
            "default_resource_limit": 50,
            "freeze_duration": 0,
            "rewards": {"inventory": {"heart": 1.0}},
        },
    }

    env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

    # Create buffers
    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)

    env.set_buffers(observations, terminals, truncations, rewards)
    env.reset()

    print("\n=== Testing Altar with initial_resource_count=1 ===")
    check_converter_state(env, "altar")

    # Try to get the initial heart
    rotate(env, Orientation.RIGHT)  # Face the altar
    obs, reward, success = perform_action(env, "get_items", 0)
    print(f"\nTrying to get initial heart from altar: success={success}, reward={reward}")

    if success:
        print("✅ Successfully collected initial heart!")
    else:
        print("❌ Failed to collect initial heart - check converter state")


if __name__ == "__main__":
    print("=== Converter Debug Tests ===\n")

    print("Test 1: Realistic converter production cycle")
    test_converter_production_cycle()

    print("\n" + "=" * 50 + "\n")

    print("Test 2: Converter with initial resources")
    test_converter_with_initial_resources()

    print("\n✅ Debug tests complete!")
