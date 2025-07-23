"""Test getting items from converters functionality.

This test file verifies that agents can successfully retrieve items from converters
using the get_items action.
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


def create_converter_test_env(initial_output_count=5, max_steps=100, converter_type="mine_red"):
    """Create a test environment with converters for item collection testing."""

    # Simple map with agent next to specific converter for easy testing
    if converter_type == "mine_red":
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "mine_red", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]
    elif converter_type == "generator_red":
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "generator_red", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]
    elif converter_type == "altar":
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "altar", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]
    elif converter_type == "lasery":
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "lasery", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]
    elif converter_type == "armory":
        game_map = [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "armory", "wall"],
            ["wall", "wall", "wall", "wall"],
        ]
    else:
        # Default map with all converters
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "mine_red", ".", "wall"],
            ["wall", ".", "generator_red", ".", "wall"],
            ["wall", ".", "altar", ".", "wall"],
            ["wall", ".", "lasery", ".", "wall"],
            ["wall", ".", "armory", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

    game_config = {
        "max_steps": max_steps,
        "num_agents": 1,
        "obs_width": 5,
        "obs_height": 5,
        "num_observation_tokens": 100,
        "inventory_item_names": ["ore_red", "battery_red", "heart", "laser", "armor"],
        "actions": {
            "noop": {"enabled": True},
            "get_items": {"enabled": True},
            "put_items": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": False},
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1},
            "mine_red": {
                "type_id": 2,
                # Mine produces ore without inputs
                "output_resources": {"ore_red": 1},
                "max_output": 10,
                "conversion_ticks": 1,
                "cooldown": 5,
                "initial_resource_count": initial_output_count,
            },
            "generator_red": {
                "type_id": 3,
                # Generator converts ore to batteries
                "input_resources": {"ore_red": 1},
                "output_resources": {"battery_red": 1},
                "max_output": 10,
                "conversion_ticks": 2,
                "cooldown": 3,
                "initial_resource_count": initial_output_count,
            },
            "altar": {
                "type_id": 4,
                # Altar converts batteries to hearts
                "input_resources": {"battery_red": 3},
                "output_resources": {"heart": 1},
                "max_output": 5,
                "conversion_ticks": 5,
                "cooldown": 10,
                "initial_resource_count": initial_output_count,
            },
            "lasery": {
                "type_id": 5,
                # Lasery creates lasers from ore and batteries
                "input_resources": {"ore_red": 1, "battery_red": 2},
                "output_resources": {"laser": 1},
                "max_output": 5,
                "conversion_ticks": 3,
                "cooldown": 5,
                "initial_resource_count": initial_output_count,
            },
            "armory": {
                "type_id": 6,
                # Armory creates armor from ore
                "input_resources": {"ore_red": 3},
                "output_resources": {"armor": 1},
                "max_output": 5,
                "conversion_ticks": 4,
                "cooldown": 8,
                "initial_resource_count": initial_output_count,
            },
        },
        "agent": {
            "default_resource_limit": 50,
            "freeze_duration": 0,
            "rewards": {
                "inventory": {
                    "ore_red": 0.01,
                    "battery_red": 0.05,
                    "heart": 1.0,
                    "laser": 0.1,
                    "armor": 0.1,
                }
            },
        },
    }

    return MettaGrid(from_mettagrid_config(game_config), game_map, 42)


def perform_action(env, action_name, action_arg=0):
    """Perform an action and return observation, reward, and success status."""
    action_idx = env.action_names().index(action_name)
    actions = np.array([[action_idx, action_arg]], dtype=dtype_actions)

    obs, rewards, terminals, truncations, info = env.step(actions)
    success = env.action_success()[0]

    return obs, rewards[0], success


def get_items_from_converter(env, orientation_to_converter, agent_idx=0):
    """Rotate to face converter and collect items. Returns (success, items_collected)."""
    # Rotate to face the converter
    rotate_result = rotate(env, orientation_to_converter, agent_idx=agent_idx)
    if not rotate_result["success"]:
        return False, {}

    # Get items from converter
    obs, reward, success = perform_action(env, "get_output", 0)

    if success:
        # Check inventory after action
        grid_objects = env.grid_objects()
        agent_id = None
        for obj_id, obj in grid_objects.items():
            if obj.get("type_name") == "agent":
                agent_id = obj_id
                break

        if agent_id:
            return True, grid_objects[agent_id].get("inventory", {})

    return False, {}


class TestConverterItems:
    """Test suite for converter item retrieval functionality."""

    def test_get_items_from_mine(self):
        """Test getting ore from a mine (simple converter with no inputs)."""
        env = create_converter_test_env(initial_output_count=3)

        # Create buffers
        observations = np.zeros((1, 100, 3), dtype=dtype_observations)
        terminals = np.zeros(1, dtype=dtype_terminals)
        truncations = np.zeros(1, dtype=dtype_truncations)
        rewards = np.zeros(1, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Agent starts at (1,1), mine is at (1,2) - to the right
        success, inventory = get_items_from_converter(env, Orientation.RIGHT)

        assert success, "Should successfully get items from mine"
        assert 0 in inventory, "Should have ore_red (item 0) in inventory"
        assert inventory[0] >= 1, f"Should have at least 1 ore_red, got {inventory.get(0, 0)}"

        print(f"✅ Successfully collected {inventory[0]} ore_red from mine")

    def test_get_items_from_generator(self):
        """Test getting batteries from a generator."""
        env = create_converter_test_env(initial_output_count=2, converter_type="generator_red")

        # Create buffers
        observations = np.zeros((1, 100, 3), dtype=dtype_observations)
        terminals = np.zeros(1, dtype=dtype_terminals)
        truncations = np.zeros(1, dtype=dtype_truncations)
        rewards = np.zeros(1, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Agent starts at (1,1), generator is at (1,2) - to the right
        success, inventory = get_items_from_converter(env, Orientation.RIGHT)

        assert success, "Should successfully get items from generator"
        assert 1 in inventory, "Should have battery_red (item 1) in inventory"
        assert inventory[1] >= 1, f"Should have at least 1 battery_red, got {inventory.get(1, 0)}"

        print(f"✅ Successfully collected {inventory[1]} battery_red from generator")

    def test_get_items_multiple_converters(self):
        """Test getting items from multiple different converters."""
        converters = [
            ("generator_red", 1),  # Generator produces battery_red (item 1)
            ("lasery", 3),  # Lasery produces laser (item 3)
            ("armory", 4),  # Armory produces armor (item 4)
        ]

        total_items = {}

        for converter_type, expected_item in converters:
            env = create_converter_test_env(initial_output_count=1, converter_type=converter_type)

            # Create buffers
            observations = np.zeros((1, 100, 3), dtype=dtype_observations)
            terminals = np.zeros(1, dtype=dtype_terminals)
            truncations = np.zeros(1, dtype=dtype_truncations)
            rewards = np.zeros(1, dtype=dtype_rewards)

            env.set_buffers(observations, terminals, truncations, rewards)
            env.reset()

            # Agent is always next to converter to the right
            success, inventory = get_items_from_converter(env, Orientation.RIGHT)

            assert success, f"Should successfully get items from {converter_type}"
            assert expected_item in inventory, f"Should have item {expected_item} from {converter_type}"

            # Update total items
            for item_id, count in inventory.items():
                total_items[item_id] = count

            print(f"✅ Collected from {converter_type}: {inventory}")

        # Verify we collected from all converters
        assert len(total_items) >= 3, "Should have collected at least 3 different item types"
        print(f"✅ Total inventory after visiting all converters: {total_items}")

    def test_empty_converter_no_items(self):
        """Test that get_items fails when converter has no output items."""
        env = create_converter_test_env(initial_output_count=0, converter_type="generator_red")

        # Create buffers
        observations = np.zeros((1, 100, 3), dtype=dtype_observations)
        terminals = np.zeros(1, dtype=dtype_terminals)
        truncations = np.zeros(1, dtype=dtype_truncations)
        rewards = np.zeros(1, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Try to get items from generator with no initial resources
        success, inventory = get_items_from_converter(env, Orientation.RIGHT)

        # The action should fail because there are no items to collect
        assert not success, "Should fail to get items from empty converter"
        assert 1 not in inventory or inventory.get(1, 0) == 0, "Should have no batteries"

        print("✅ Correctly failed to get items from empty converter")

    def test_converter_with_production(self):
        """Test waiting for converter to produce items then collecting them."""
        # First test - mine with initial resources
        env = create_converter_test_env(initial_output_count=5, converter_type="mine_red")

        # Create buffers
        observations = np.zeros((1, 100, 3), dtype=dtype_observations)
        terminals = np.zeros(1, dtype=dtype_terminals)
        truncations = np.zeros(1, dtype=dtype_truncations)
        rewards = np.zeros(1, dtype=dtype_rewards)

        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        # Get ore from mine
        success, inventory = get_items_from_converter(env, Orientation.RIGHT)
        assert success, "Should get ore from mine"
        ore_count = inventory.get(0, 0)
        assert ore_count > 0, "Should have collected ore"

        print(f"✅ Collected {ore_count} ore from mine")
        print("✅ Full test of converter production cycle would require put_items implementation")


if __name__ == "__main__":
    test = TestConverterItems()
    test.test_get_items_from_mine()
    test.test_get_items_from_generator()
    test.test_get_items_multiple_converters()
    test.test_empty_converter_no_items()
    test.test_converter_with_production()
    print("\n✅ All converter item tests passed!")
