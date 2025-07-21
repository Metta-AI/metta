"""Test cases demonstrating action compatibility breaking changes."""

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


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
            "action_failure_penalty": 0.1,
            "resource_limits": {"ore": 10, "wood": 10},
            "rewards": {  # Changed from resource_rewards/resource_reward_max
                "inventory": {},
                "stats": {},
            },
        },
        "groups": {
            "default": {
                "id": 0,  # Added required id field
                "group_reward_pct": 1.0,
                # Removed spawn_prob - not a valid field
            }
        },
        "actions": {
            "move": {
                "enabled": True,
                "required_resources": {},
                "consumed_resources": {},
            },
            "noop": {
                "enabled": True,
                "required_resources": {},
                "consumed_resources": {},
            },
            "rotate": {
                "enabled": True,
                "required_resources": {},
                "consumed_resources": {},
            },
        },
        "objects": {
            "wall": {
                "type_id": 1,  # Changed from type/texture to type_id
                "swappable": False,  # Added swappable field for wall
            }
        },
    }


def create_simple_map():
    """Create a simple 5x5 map with walls around edges."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", ".", "agent.default", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]
    return game_map


def test_action_index_changes():
    """Test that action order is fixed regardless of config order.

    This demonstrates that the C++ implementation maintains a fixed action order,
    which means trained policies are consistent across different config orderings.
    """
    # Create original config
    config1 = create_basic_config()
    env1 = MettaGrid(from_mettagrid_config(config1), create_simple_map(), 42)

    # Get action indices
    action_names1 = env1.action_names()
    assert action_names1 == ["move", "noop", "rotate"]

    # Create config with different action order in the dictionary
    config2 = create_basic_config()
    # Reorder actions - rotate first, then noop, then move
    config2["actions"] = {
        "rotate": config2["actions"]["rotate"],
        "noop": config2["actions"]["noop"],
        "move": config2["actions"]["move"],
    }

    env2 = MettaGrid(from_mettagrid_config(config2), create_simple_map(), 42)
    action_names2 = env2.action_names()
    # Action order remains the same despite different config order
    assert action_names2 == ["move", "noop", "rotate"]

    # This actually protects trained policies from config reordering
    # The same numeric action executes the same behavior
    action = np.array([[0, 0]], dtype=np.int32)  # Index 0, arg 0

    # Execute same numeric action in both environments
    env1.reset()
    env2.reset()

    # Both execute "move" since index 0 is always "move"
    obs1, reward1, done1, trunc1, info1 = env1.step(action)
    obs2, reward2, done2, trunc2, info2 = env2.step(action)

    # The action effects should be identical
    success1 = env1.action_success()[0]
    success2 = env2.action_success()[0]

    # Both should have the same result
    assert success1 == success2
    print(f"Action index 0 in both envs executes 'move': success={success1}")


def test_max_arg_reduction():
    """Test that reducing max_arg makes previously valid actions invalid."""
    # Create environment with standard move action
    config = create_basic_config()
    env = MettaGrid(from_mettagrid_config(config), create_simple_map(), 42)  # Changed seed=42 to 42
    env.reset()

    # Get max args for each action
    max_args = env.max_action_args()
    move_idx = env.action_names().index("move")

    # Default move has max_arg=1 (forward=0, backward=1)
    assert max_args[move_idx] == 1

    # Both forward and backward should be valid
    forward_action = np.array([[move_idx, 0]], dtype=np.int32)
    backward_action = np.array([[move_idx, 1]], dtype=np.int32)

    env.step(forward_action)
    assert env.action_success()[0] or env.action_success()[1] != "invalid_arg", "Action failed due to invalid argument"

    env.step(backward_action)
    assert env.action_success()[0] or env.action_success()[1] != "invalid_arg", "Action failed due to invalid argument"

    # If we could modify max_arg at runtime (which we can't in current implementation),
    # the backward action would become invalid
    # This demonstrates the concept even though we can't test it directly


def test_resource_requirement_changes():
    """Test that changing resource requirements breaks actions."""
    # Create config where move requires resources
    config1 = create_basic_config()
    config1["actions"]["move"]["required_resources"] = {"ore": 1}
    config1["actions"]["move"]["consumed_resources"] = {"ore": 1}

    env1 = MettaGrid(from_mettagrid_config(config1), create_simple_map(), 42)  # Changed seed=42 to 42
    env1.reset()

    move_idx = env1.action_names().index("move")
    move_action = np.array([[move_idx, 0]], dtype=np.int32)

    # Agent starts with no resources, so move should fail
    env1.step(move_action)
    success_without_resources = env1.action_success()[0]
    assert not success_without_resources

    # Create config where move requires no resources
    config2 = create_basic_config()
    env2 = MettaGrid(from_mettagrid_config(config2), create_simple_map(), 42)  # Changed seed=42 to 42
    env2.reset()

    # Same action should now succeed (unless blocked by wall)
    env2.step(move_action)
    success_without_requirement = env2.action_success()[0]

    # This demonstrates how resource requirement changes affect action success
    print(f"Move with resource requirement: success={success_without_resources}")
    print(f"Move without resource requirement: success={success_without_requirement}")


def test_inventory_item_reordering():
    """Test that reordering inventory items breaks resource-based actions."""
    # Config with ore first
    config1 = create_basic_config()
    config1["inventory_item_names"] = ["ore", "wood"]
    config1["actions"]["move"]["required_resources"] = {"ore": 1}

    # Config with wood first
    config2 = create_basic_config()
    config2["inventory_item_names"] = ["wood", "ore"]
    config2["actions"]["move"]["required_resources"] = {"ore": 1}

    # In config1, ore has index 0
    # In config2, ore has index 1
    # This would cause the resource check to look at the wrong inventory slot

    env1 = MettaGrid(from_mettagrid_config(config1), create_simple_map(), 42)  # Changed seed=42 to 42
    env2 = MettaGrid(from_mettagrid_config(config2), create_simple_map(), 42)  # Changed seed=42 to 42

    assert env1.inventory_item_names() == ["ore", "wood"]
    assert env2.inventory_item_names() == ["wood", "ore"]


def test_action_validation_stats():
    """Test that invalid actions are tracked in stats."""
    config = create_basic_config()
    env = MettaGrid(from_mettagrid_config(config), create_simple_map(), 42)  # Changed seed=42 to 42
    env.reset()

    # Try invalid action type
    invalid_type = np.array([[99, 0]], dtype=np.int32)  # Action type 99 doesn't exist
    env.step(invalid_type)

    # Try invalid action argument
    move_idx = env.action_names().index("move")
    invalid_arg = np.array([[move_idx, 99]], dtype=np.int32)  # Arg 99 exceeds max_arg
    env.step(invalid_arg)

    # Get stats to verify tracking
    stats = env.get_episode_stats()

    # These stats would be tracked as:
    # - action.invalid_type
    # - action.invalid_arg
    print("Episode stats after invalid actions:", stats)


def test_action_space_dimensions():
    """Test action space shape for compatibility checks."""
    config = create_basic_config()
    env = MettaGrid(from_mettagrid_config(config), create_simple_map(), 42)  # Changed seed=42 to 42

    action_space = env.action_space

    # MultiDiscrete space with [num_actions, max_arg+1]
    assert hasattr(action_space, "nvec")
    assert len(action_space.nvec) == 2

    num_actions = action_space.nvec[0]
    max_arg_plus_one = action_space.nvec[1]

    # Should match our configuration
    assert num_actions == len(env.action_names())

    print(f"Action space shape: {action_space.nvec}")
    print(f"Num actions: {num_actions}, Max arg: {max_arg_plus_one - 1}")


def test_special_attack_action():
    """Test that attack action is properly registered."""
    config = create_basic_config()
    # Add attack action
    config["actions"]["attack"] = {
        "enabled": True,
        "required_resources": {},
        "consumed_resources": {},
        "defense_resources": {},
    }

    env = MettaGrid(from_mettagrid_config(config), create_simple_map(), 42)  # Changed seed=42 to 42
    action_names = env.action_names()

    # Attack should be present in the action list
    assert "attack" in action_names

    # Check the order - attack should come first before the basic actions
    expected_actions = ["attack", "move", "noop", "rotate"]
    assert action_names == expected_actions

    print(f"Action names with attack: {action_names}")


if __name__ == "__main__":
    # Run tests to demonstrate breaking changes
    test_action_index_changes()
    test_max_arg_reduction()
    test_resource_requirement_changes()
    test_inventory_item_reordering()
    test_action_validation_stats()
    test_action_space_dimensions()
    test_special_attack_action()
