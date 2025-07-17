"""Test that attack actions properly consume configured resources."""

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


def test_attack_consumes_laser_resource():
    """Test that attacks consume laser resources as configured in mettagrid.yaml."""
    # Create a simple map with two agents facing each other
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", "agent.blue", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = {
        "max_steps": 50,
        "num_agents": 2,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        "inventory_item_names": ["laser", "armor", "heart"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {
                "enabled": True,
                "consumed_resources": {"laser": 1},
                "defense_resources": {"armor": 1}
            },
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "swap": {"enabled": True},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False, "number_of_glyphs": 4},
        },
        "groups": {
            "red": {"id": 0, "props": {}},
            "blue": {"id": 1, "props": {}}
        },
        "objects": {
            "wall": {"type_id": 1}
        },
        "agent": {
            "default_resource_limit": 10,
            "freeze_duration": 5,
            "rewards": {}
        }
    }

    # Create the environment
    env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

    # Set up observation and reward buffers
    num_agents = 2
    num_obs_tokens = 200
    obs_token_size = 3
    observations = np.zeros((num_agents, num_obs_tokens, obs_token_size), dtype=np.uint8)
    terminals = np.zeros(num_agents, dtype=np.bool_)
    truncations = np.zeros(num_agents, dtype=np.bool_)
    rewards = np.zeros(num_agents, dtype=np.float32)

    env.set_buffers(observations, terminals, truncations, rewards)
    env.reset()

    # Give agent 0 some lasers
    initial_laser_count = 5
    laser_id = env._inventory_item_names.index("laser")
    agent0 = env._agents[0]
    agent0.update_inventory(laser_id, initial_laser_count)

    # Verify agent has lasers
    assert agent0.inventory[laser_id] == initial_laser_count

    # Position agents to face each other (agent 0 faces right, agent 1 is to the right)
    agent1 = env._agents[1]
    agent0.orientation = 3  # Face right

    # Get attack action id
    attack_action_id = None
    for i, handler in enumerate(env._action_handlers):
        if handler.action_name() == "attack":
            attack_action_id = i
            break

    assert attack_action_id is not None, "Attack action not found"

    # Execute attack with arg 0 (target directly in front)
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0, 0] = attack_action_id  # Attack action
    actions[0, 1] = 0  # Target directly in front

    env.step(actions)

    # Check that laser was consumed
    final_laser_count = agent0.inventory[laser_id]
    assert final_laser_count == initial_laser_count - 1, \
        f"Attack should consume 1 laser, but laser count went from {initial_laser_count} to {final_laser_count}"

    # Verify the attack was successful (agent 1 should be frozen)
    assert agent1.frozen > 0, "Target agent should be frozen after successful attack"


def test_attack_fails_without_laser():
    """Test that attacks fail when agent doesn't have required laser resource."""
    # Create a simple map with two agents
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", "agent.blue", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = {
        "max_steps": 50,
        "num_agents": 2,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        "inventory_item_names": ["laser", "armor", "heart"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {
                "enabled": True,
                "consumed_resources": {"laser": 1},
                "defense_resources": {"armor": 1}
            },
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "swap": {"enabled": True},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False, "number_of_glyphs": 4},
        },
        "groups": {
            "red": {"id": 0, "props": {}},
            "blue": {"id": 1, "props": {}}
        },
        "objects": {
            "wall": {"type_id": 1}
        },
        "agent": {
            "default_resource_limit": 10,
            "freeze_duration": 5,
            "rewards": {},
            "action_failure_penalty": 0.1
        }
    }

    # Create the environment
    env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

    # Set up buffers
    num_agents = 2
    observations = np.zeros((num_agents, 200, 3), dtype=np.uint8)
    terminals = np.zeros(num_agents, dtype=np.bool_)
    truncations = np.zeros(num_agents, dtype=np.bool_)
    rewards = np.zeros(num_agents, dtype=np.float32)

    env.set_buffers(observations, terminals, truncations, rewards)
    env.reset()

    # Don't give agent any lasers
    agent0 = env._agents[0]
    agent1 = env._agents[1]

    # Position agents to face each other
    agent0.orientation = 3  # Face right

    # Get attack action id
    attack_action_id = None
    for i, handler in enumerate(env._action_handlers):
        if handler.action_name() == "attack":
            attack_action_id = i
            break

    # Try to execute attack without laser
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0, 0] = attack_action_id
    actions[0, 1] = 0

    env.step(actions)

    # Check that attack failed (agent 1 should not be frozen)
    assert agent1.frozen == 0, "Attack should fail without required laser resource"

    # Check that agent received failure penalty
    assert rewards[0] < 0, f"Agent should receive penalty for failed action, but got reward {rewards[0]}"


def test_attack_without_laser_in_inventory_is_free():
    """Test the bug: when laser is not in inventory_item_names, attacks are free."""
    # Create config without laser in inventory_item_names (mimicking the bug)
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", "agent.blue", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    game_config = {
        "max_steps": 50,
        "num_agents": 2,
        "obs_width": 11,
        "obs_height": 11,
        "num_observation_tokens": 200,
        # Note: laser is NOT in inventory_item_names, which causes the bug
        "inventory_item_names": ["armor", "heart"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {
                "enabled": True,
                "consumed_resources": {"laser": 1},  # This gets ignored!
                "defense_resources": {"armor": 1}
            },
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "swap": {"enabled": True},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False, "number_of_glyphs": 4},
        },
        "groups": {
            "red": {"id": 0, "props": {}},
            "blue": {"id": 1, "props": {}}
        },
        "objects": {
            "wall": {"type_id": 1}
        },
        "agent": {
            "default_resource_limit": 10,
            "freeze_duration": 5,
            "rewards": {}
        }
    }

    # Create the environment
    env = MettaGrid(from_mettagrid_config(game_config), game_map, 42)

    # Set up buffers
    num_agents = 2
    observations = np.zeros((num_agents, 200, 3), dtype=np.uint8)
    terminals = np.zeros(num_agents, dtype=np.bool_)
    truncations = np.zeros(num_agents, dtype=np.bool_)
    rewards = np.zeros(num_agents, dtype=np.float32)

    env.set_buffers(observations, terminals, truncations, rewards)
    env.reset()

    agent0 = env._agents[0]
    agent1 = env._agents[1]

    # Position agents to face each other
    agent0.orientation = 3  # Face right

    # Get attack action id
    attack_action_id = None
    for i, handler in enumerate(env._action_handlers):
        if handler.action_name() == "attack":
            attack_action_id = i
            break

    # Execute attack - this should succeed even without laser!
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0, 0] = attack_action_id
    actions[0, 1] = 0

    env.step(actions)

    # The bug: attack succeeds even without laser because laser isn't in inventory
    # This test demonstrates the bug - attacks are free when laser is not in inventory_item_names
    assert agent1.frozen > 0, "Bug confirmed: Attack succeeds without laser when laser is not in inventory_item_names"


if __name__ == "__main__":
    print("Running test_attack_consumes_laser_resource...")
    test_attack_consumes_laser_resource()
    print("✓ Passed")

    print("\nRunning test_attack_fails_without_laser...")
    test_attack_fails_without_laser()
    print("✓ Passed")

    print("\nRunning test_attack_without_laser_in_inventory_is_free...")
    test_attack_without_laser_in_inventory_is_free()
    print("✓ Passed (demonstrates the bug)")

    print("\nAll tests completed!")
