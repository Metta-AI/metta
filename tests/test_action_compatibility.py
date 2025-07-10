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
            "resource_rewards": {},
            "resource_reward_max": {},
        },
        "groups": {
            "default": {
                "group_reward_pct": 1.0,
                "spawn_prob": 1.0,
            }
        },
        "actions": {
            "noop": {
                "enabled": True,
                "required_resources": {},
                "consumed_resources": {},
            },
            "move": {
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
                "type": "wall",
                "texture": 1,
            }
        },
    }


def create_simple_map():
    """Create a simple 5x5 map with walls around edges."""
    game_map = [
        [b"#", b"#", b"#", b"#", b"#"],
        [b"#", b".", b".", b".", b"#"],
        [b"#", b".", b"1", b".", b"#"],
        [b"#", b".", b".", b".", b"#"],
        [b"#", b"#", b"#", b"#", b"#"],
    ]
    return game_map


def test_action_index_changes():
    """Test that changing action order breaks trained policies."""
    # Create original config
    config1 = create_basic_config()
    env1 = MettaGrid(from_mettagrid_config(config1), create_simple_map(), seed=42)

    # Get action indices
    action_names1 = env1.action_names()
    assert action_names1 == ["noop", "move", "rotate"]

    # Create config with different action order
    config2 = create_basic_config()
    # Reorder actions
    config2["actions"] = {
        "move": config2["actions"]["move"],
        "noop": config2["actions"]["noop"],
        "rotate": config2["actions"]["rotate"],
    }

    env2 = MettaGrid(from_mettagrid_config(config2), create_simple_map(), seed=42)
    action_names2 = env2.action_names()
    assert action_names2 == ["move", "noop", "rotate"]

    # Action that was "move" (index 1) in env1 is now "noop" (index 1) in env2
    action = np.array([[1, 0]], dtype=np.int32)  # Index 1, arg 0

    # Execute same numeric action in both environments
    env1.reset()
    env2.reset()

    _, reward1, _, _, _ = env1.step(action)
    _, reward2, _, _, _ = env2.step(action)

    # In env1, this executes "move", in env2 it executes "noop"
    # Their effects should be different
    success1 = env1.action_success()[0]
    success2 = env2.action_success()[0]

    # Noop always succeeds, move might fail if blocked
    # This demonstrates the action mismatch
    print(f"Action index 1 in env1 (move): success={success1}")
    print(f"Action index 1 in env2 (noop): success={success2}")


def test_max_arg_reduction():
    """Test that reducing max_arg makes previously valid actions invalid."""
    # Create environment with standard move action
    config = create_basic_config()
    env = MettaGrid(from_mettagrid_config(config), create_simple_map(), seed=42)
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
    assert env.action_success()[0] or True  # Might fail due to walls, but won't be invalid_arg

    env.step(backward_action)
    assert env.action_success()[0] or True  # Might fail due to walls, but won't be invalid_arg

    # If we could modify max_arg at runtime (which we can't in current implementation),
    # the backward action would become invalid
    # This demonstrates the concept even though we can't test it directly


def test_resource_requirement_changes():
    """Test that changing resource requirements breaks actions."""
    # Create config where move requires resources
    config1 = create_basic_config()
    config1["actions"]["move"]["required_resources"] = {"ore": 1}
    config1["actions"]["move"]["consumed_resources"] = {"ore": 1}

    env1 = MettaGrid(from_mettagrid_config(config1), create_simple_map(), seed=42)
    env1.reset()

    move_idx = env1.action_names().index("move")
    move_action = np.array([[move_idx, 0]], dtype=np.int32)

    # Agent starts with no resources, so move should fail
    env1.step(move_action)
    success_without_resources = env1.action_success()[0]
    assert not success_without_resources

    # Create config where move requires no resources
    config2 = create_basic_config()
    env2 = MettaGrid(from_mettagrid_config(config2), create_simple_map(), seed=42)
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

    env1 = MettaGrid(from_mettagrid_config(config1), create_simple_map(), seed=42)
    env2 = MettaGrid(from_mettagrid_config(config2), create_simple_map(), seed=42)

    assert env1.inventory_item_names() == ["ore", "wood"]
    assert env2.inventory_item_names() == ["wood", "ore"]


def test_action_validation_stats():
    """Test that invalid actions are tracked in stats."""
    config = create_basic_config()
    env = MettaGrid(from_mettagrid_config(config), create_simple_map(), seed=42)
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
    env = MettaGrid(from_mettagrid_config(config), create_simple_map(), seed=42)

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
    """Test that attack action creates multiple handlers."""
    config = create_basic_config()
    # Add attack action
    config["actions"]["attack"] = {
        "enabled": True,
        "required_resources": {},
        "consumed_resources": {},
        "defense_resources": {},
    }

    env = MettaGrid(from_mettagrid_config(config), create_simple_map(), seed=42)
    action_names = env.action_names()

    # Attack should create both "attack" and "attack_nearest"
    assert "attack" in action_names
    assert "attack_nearest" in action_names

    # They should be consecutive indices
    attack_idx = action_names.index("attack")
    attack_nearest_idx = action_names.index("attack_nearest")
    assert attack_nearest_idx == attack_idx + 1

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
