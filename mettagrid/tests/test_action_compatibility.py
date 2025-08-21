"""Test cases for action system compatibility and behavior."""

import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid, dtype_actions
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.test_support.actions import get_agent_position, get_current_observation


def create_basic_config():
    """Create a minimal valid game configuration."""
    return {
        "inventory_item_names": ["ore", "wood"],
        "num_agents": 1,
        "max_steps": 100,
        "obs_width": 7,
        "obs_height": 7,
        "num_observation_tokens": 50,
        "agent": {
            "freeze_duration": 0,
            "resource_limits": {"ore": 10, "wood": 10},
            "rewards": {
                "inventory": {},
                "stats": {},
            },
        },
        "groups": {
            "default": {
                "id": 0,
                "group_reward_pct": 1.0,
            }
        },
        "actions": {
            "move": {"enabled": True},
            "noop": {"enabled": True},
            "rotate": {"enabled": True},
        },
        "objects": {
            "wall": {
                "type_id": 1,
                "swappable": False,
            }
        },
    }


def create_simple_map():
    """Create a simple 5x5 map with walls around edges."""
    return [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.default", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]


def create_multi_agent_map():
    """Create a simple 7x7 map with multiple agents."""
    return [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.default", "empty", "agent.default", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "agent.default", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]


@pytest.fixture
def basic_config():
    """Fixture for basic configuration."""
    return create_basic_config()


@pytest.fixture
def simple_map():
    """Fixture for simple map."""
    return create_simple_map()


@pytest.fixture
def multi_agent_map():
    """Fixture for multi-agent map."""
    return create_multi_agent_map()


class TestActionOrdering:
    """Tests related to action ordering and indexing."""

    def test_action_order_is_fixed(self, basic_config, simple_map):
        """Test that action order is deterministic regardless of config order."""
        # Create environment with original config
        env1 = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        action_names1 = env1.action_names()

        # Create config with different action order in the dictionary
        reordered_config = basic_config.copy()
        reordered_config["actions"] = {
            "rotate": basic_config["actions"]["rotate"],
            "noop": basic_config["actions"]["noop"],
            "move": basic_config["actions"]["move"],
        }

        env2 = MettaGrid(from_mettagrid_config(reordered_config), simple_map, 42)
        action_names2 = env2.action_names()

        # Action order should remain the same despite different config order
        assert action_names1 == action_names2, "Action order should be deterministic"

        # Verify the expected order
        assert action_names1 == ["get_items", "move", "noop", "rotate"]

    def test_action_indices_consistency(self, basic_config, simple_map):
        """Test that action indices remain consistent."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        action_names = env.action_names()

        # Verify indices
        assert action_names.index("move") == 1
        assert action_names.index("noop") == 2
        assert action_names.index("rotate") == 3

        # get_items is always present even if not explicitly configured
        assert action_names.index("get_items") == 0


class TestActionValidation:
    """Tests for action validation and error handling."""

    def test_invalid_action_type(self, basic_config, simple_map):
        """Test that invalid action types are handled properly."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        env.reset()

        # Try invalid action type
        invalid_action = np.array([[99, 0]], dtype=dtype_actions)
        env.step(invalid_action)

        # Action should fail
        assert not env.action_success()[0], "Invalid action type should fail"

    def test_invalid_action_argument(self, basic_config, simple_map):
        """Test that invalid action arguments are handled properly."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)
        env.reset()

        move_idx = env.action_names().index("move")
        max_args = env.max_action_args()

        # Try argument exceeding max_arg
        invalid_arg = max_args[move_idx] + 10
        invalid_action = np.array([[move_idx, invalid_arg]], dtype=dtype_actions)
        env.step(invalid_action)

        # Action should fail
        assert not env.action_success()[0], "Invalid action argument should fail"

    def test_max_action_args(self, basic_config, simple_map):
        """Test max_action_args for different actions."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)

        action_names = env.action_names()
        max_args = env.max_action_args()

        # Check expected max args
        move_idx = action_names.index("move")
        noop_idx = action_names.index("noop")
        rotate_idx = action_names.index("rotate")

        assert max_args[move_idx] == 7, "Move should have max_arg=7 (8 directions)"
        assert max_args[noop_idx] == 0, "Noop should have max_arg=0"
        assert max_args[rotate_idx] == 3, "Rotate should have max_arg=3 (4 orientations)"


class TestResourceRequirements:
    """Tests for action resource requirements."""

    def test_action_with_resource_requirement(self, basic_config, simple_map):
        """Test that actions fail when resource requirements aren't met."""
        # Add resource requirement to move
        config = basic_config.copy()
        config["actions"]["move"]["required_resources"] = {"ore": 1}

        env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
        env.reset()

        move_idx = env.action_names().index("move")
        move_action = np.array([[move_idx, 0]], dtype=dtype_actions)

        # Agent starts with no resources, so move should fail
        env.step(move_action)
        assert not env.action_success()[0], "Move should fail without required resources"

        # Give agent resources and try again
        # Note: This would require a way to add resources to agent
        # which depends on the environment setup

    def test_action_consumes_resources(self, basic_config, simple_map):
        """Test that actions consume resources when configured."""
        config = basic_config.copy()
        config["actions"]["move"]["consumed_resources"] = {"ore": 1}

        # Give agent initial ore to test consumption
        config["agent"]["initial_inventory"] = {"ore": 5, "wood": 3}

        # Create simple map
        env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
        env.reset()

        # Get initial observation
        initial_obs = get_current_observation(env, agent_idx=0)
        print(f"Observation shape: {initial_obs.shape}")

        # Extract inventory tokens using ObservationHelper
        # First, let's see all tokens to understand the structure
        print("\nAll tokens in observation:")
        for i in range(min(20, initial_obs.shape[1])):
            token = initial_obs[0, i, :]
            print(f"  Token {i}: {token}")

        # Get inventory item names to know the expected feature IDs
        inventory_names = env.inventory_item_names()
        print(f"\nInventory item names: {inventory_names}")

        # Feature IDs for inventory items start at 15 (ObservationFeatureCount)
        # ore should be feature ID 15, wood should be 16
        ore_feature_id = 15
        wood_feature_id = 16

        # Find inventory tokens by feature type
        print("\nLooking for inventory tokens...")
        initial_ore_count = None
        initial_wood_count = None

        for i in range(initial_obs.shape[1]):
            token = initial_obs[0, i, :]
            if token[1] == ore_feature_id:
                initial_ore_count = token[2]
                print(f"  Found ore token: {token} - count: {initial_ore_count}")
            elif token[1] == wood_feature_id:
                initial_wood_count = token[2]
                print(f"  Found wood token: {token} - count: {initial_wood_count}")

        # Verify initial inventory
        assert initial_ore_count == 5, f"Expected initial ore to be 5, got {initial_ore_count}"
        assert initial_wood_count == 3, f"Expected initial wood to be 3, got {initial_wood_count}"
        print(f"\nInitial inventory verified: ore={initial_ore_count}, wood={initial_wood_count}")

        # Get agent position
        agent_pos = get_agent_position(env, 0)
        print(f"\nAgent position: {agent_pos}")

        # Move east (direction 2)
        move_idx = env.action_names().index("move")
        move_action = np.array([[move_idx, 2]], dtype=dtype_actions)

        print("\nPerforming move action...")
        obs_after, _rewards, _dones, _truncs, _infos = env.step(move_action)
        action_success = env.action_success()[0]
        print(f"Move action success: {action_success}")

        # Check new position
        new_pos = get_agent_position(env, 0)
        print(f"New agent position: {new_pos}")
        position_changed = new_pos != agent_pos
        print(f"Position changed: {position_changed}")

        # Get final inventory counts
        final_ore_count = None
        final_wood_count = None

        for i in range(obs_after.shape[1]):
            token = obs_after[0, i, :]
            if token[1] == ore_feature_id:
                final_ore_count = token[2]
            elif token[1] == wood_feature_id:
                final_wood_count = token[2]

        print(f"\nFinal inventory: ore={final_ore_count}, wood={final_wood_count}")

        # Verify resource consumption
        if action_success and position_changed:
            assert final_ore_count == initial_ore_count - 1, (
                f"Ore should be consumed on successful move: expected {initial_ore_count - 1}, got {final_ore_count}"
            )
            assert final_wood_count == initial_wood_count, (
                f"Wood should not be consumed: expected {initial_wood_count}, got {final_wood_count}"
            )
            print("✓ Resources consumed correctly (ore decreased by 1, wood unchanged)")
        else:
            assert final_ore_count == initial_ore_count, "Ore shouldn't be consumed on failed/blocked move"
            assert final_wood_count == initial_wood_count, "Wood shouldn't be consumed on failed/blocked move"
            print("✓ Resources not consumed (move failed/blocked)")


class TestActionSpace:
    """Tests for action space properties."""

    def test_action_space_shape(self, basic_config, simple_map):
        """Test action space dimensions."""
        env = MettaGrid(from_mettagrid_config(basic_config), simple_map, 42)

        action_space = env.action_space

        # Should be MultiDiscrete with 2 dimensions
        assert hasattr(action_space, "nvec"), "Action space should be MultiDiscrete"
        assert len(action_space.nvec) == 2, "Action space should have 2 dimensions"

        num_actions = action_space.nvec[0]
        max_arg_plus_one = action_space.nvec[1]

        # Should match our configuration
        assert num_actions == len(env.action_names())

        # Max arg is the maximum across all actions
        max_args = env.max_action_args()
        assert max_arg_plus_one == max(max_args) + 1

    def test_single_action_space(self, basic_config, multi_agent_map):
        """Test action space for multi-agent environment."""
        config = basic_config.copy()
        config["num_agents"] = 3

        env = MettaGrid(from_mettagrid_config(config), multi_agent_map, 42)

        # Check that action_space exists
        assert hasattr(env, "action_space")

        action_space = env.action_space
        assert action_space is not None

        # The action space should be MultiDiscrete for each agent's action
        assert hasattr(action_space, "nvec"), "Action space should be MultiDiscrete"
        assert len(action_space.nvec) == 2, "Action space should have 2 dimensions (action_type, action_arg)"

        # First dimension is number of action types
        num_actions = action_space.nvec[0]
        assert num_actions == len(env.action_names())

        # Second dimension is max action argument + 1
        max_arg_plus_one = action_space.nvec[1]
        max_args = env.max_action_args()
        assert max_arg_plus_one == max(max_args) + 1

        # When stepping, we need to provide actions for all agents
        env.reset()

        # Create actions for all 3 agents (noop for each)
        noop_idx = env.action_names().index("noop")
        actions = np.array([[noop_idx, 0]] * 3, dtype=dtype_actions)

        # This should work without error
        env.step(actions)

        # Verify we get results for all agents
        assert len(env.action_success()) == 3, "Should get action success for all agents"


class TestSpecialActions:
    """Tests for special action types."""

    def test_attack_action_registration(self, basic_config, simple_map):
        """Test that attack action is properly registered when enabled."""
        config = basic_config.copy()
        config["actions"]["attack"] = {
            "enabled": True,
            "required_resources": {},
            "consumed_resources": {},
            "defense_resources": {},
        }

        env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
        action_names = env.action_names()

        # Attack should be present
        assert "attack" in action_names

        # Check the expected order (attack comes before other actions)
        expected_actions = ["attack", "get_items", "move", "noop", "rotate"]
        assert action_names == expected_actions

    def test_swap_action_registration(self, basic_config, simple_map):
        """Test that swap action is properly registered when enabled."""
        config = basic_config.copy()
        config["actions"]["swap"] = {"enabled": True}

        env = MettaGrid(from_mettagrid_config(config), simple_map, 42)
        action_names = env.action_names()

        assert "swap" in action_names


class TestInventoryItemOrdering:
    """Tests for inventory item ordering effects."""

    def test_inventory_item_order(self, basic_config, simple_map):
        """Test that inventory items maintain their order."""
        # Config with ore first
        config1 = basic_config.copy()
        config1["inventory_item_names"] = ["ore", "wood"]

        # Config with wood first
        config2 = basic_config.copy()
        config2["inventory_item_names"] = ["wood", "ore"]

        env1 = MettaGrid(from_mettagrid_config(config1), simple_map, 42)
        env2 = MettaGrid(from_mettagrid_config(config2), simple_map, 42)

        assert env1.inventory_item_names() == ["ore", "wood"]
        assert env2.inventory_item_names() == ["wood", "ore"]

        # This affects resource indices in the implementation
        # ore is index 0 in env1, but index 1 in env2
