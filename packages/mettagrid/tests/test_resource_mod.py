# Test file for resource modification action with our simplified implementation
"""Test resource modification with simple ModifyTarget C++ action."""

import numpy as np
import pytest

from mettagrid.mettagrid_c import (
    ActionConfig as CppActionConfig,
    AgentConfig as CppAgentConfig,
    ConverterConfig as CppConverterConfig,
    GameConfig as CppGameConfig,
    GlobalObsConfig as CppGlobalObsConfig,
    MettaGrid,
    ModifyTargetConfig,
    WallConfig as CppWallConfig,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import GameConfig, ActionConfig


def test_modify_target_config():
    """Test that ModifyTargetConfig can be created directly."""
    config = ModifyTargetConfig(
        required_resources={0: 5},
        consumed_resources={0: 1.5},
        modifies={1: 2.0, 2: -0.5},
    )

    assert config.required_resources == {0: 5}
    assert config.consumed_resources == {0: 1.5}
    assert config.modifies == {1: 2.0, 2: -0.5}


def test_modify_target_consumption():
    """Test that ModifyTarget correctly consumes resources from the actor."""
    # Create a simple map with one agent
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Create ModifyTargetConfig that consumes resources
    modify_target_config = ModifyTargetConfig(
        required_resources={0: 1},  # Need at least 1 energy to use
        consumed_resources={0: 1.0},  # 100% chance to consume 1 unit of resource 0 (energy)
        modifies={},  # No modifications to others
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=1,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "modify_target": modify_target_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 10, 1: 100},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    # Create environment
    env = MettaGrid(cpp_config, game_map, 42)

    # Set up buffers
    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    # Get initial agent state
    objects = env.grid_objects()
    agent_id = None
    for oid, obj in objects.items():
        if obj.get("type") == 0:  # Agent type
            agent_id = oid
            break

    assert agent_id is not None
    initial_energy = objects[agent_id].get("inventory", {}).get(0, 0)

    # Execute modify_target action
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0x77]  # Target self at offset (0, 0)
    env.step(actions)

    # Check that energy was consumed
    objects = env.grid_objects()
    final_energy = objects[agent_id].get("inventory", {}).get(0, 0)

    # With 100% probability, should consume exactly 1 unit
    assert final_energy == initial_energy - 1


def test_simple_modify_target():
    """Test using ModifyTarget action directly to modify a specific target."""
    # Create a map with two agents
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Create a simple ModifyTarget action
    modify_config = ModifyTargetConfig(
        required_resources={0: 1},
        consumed_resources={0: 1.0},
        modifies={1: 5.0},  # Add 5 health to target
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=2,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "modify_target": modify_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 10},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=1,
                group_name="blue",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 10},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env = MettaGrid(cpp_config, game_map, 42)

    # Set up buffers
    observations = np.zeros((2, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(2, dtype=dtype_terminals)
    truncations = np.zeros(2, dtype=dtype_truncations)
    rewards = np.zeros(2, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    # Get initial states
    objects = env.grid_objects()
    red_agent_id = None
    blue_agent_id = None
    for oid, obj in objects.items():
        if obj.get("type") == 0:
            if obj["r"] == 1 and obj["c"] == 1:
                red_agent_id = oid
            elif obj["r"] == 1 and obj["c"] == 2:
                blue_agent_id = oid

    red_health_before = objects[red_agent_id].get("inventory", {}).get(1, 0)
    blue_health_before = objects[blue_agent_id].get("inventory", {}).get(1, 0)
    red_energy_before = objects[red_agent_id].get("inventory", {}).get(0, 0)

    # Red agent targets blue agent (offset: row=0, col=1)
    # Encoding: ((0+7) << 4) | (1+7) = 0x78
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0] = [action_idx, 0x78]  # Target at offset (0, 1)
    actions[1] = [0, 0]  # Blue does noop

    env.step(actions)

    # Check results
    objects = env.grid_objects()
    red_health_after = objects[red_agent_id].get("inventory", {}).get(1, 0)
    blue_health_after = objects[blue_agent_id].get("inventory", {}).get(1, 0)
    red_energy_after = objects[red_agent_id].get("inventory", {}).get(0, 0)

    # Red should have consumed energy
    assert red_energy_after == red_energy_before - 1

    # Blue should have gained health
    assert blue_health_after == blue_health_before + 5

    # Red's health should be unchanged
    assert red_health_after == red_health_before


def test_modify_target_converters():
    """Test that ModifyTarget can affect converters."""
    # Create a map with agent and converter
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "converter.blue", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    modify_config = ModifyTargetConfig(
        required_resources={},
        consumed_resources={},
        modifies={0: 10.0},  # Add 10 energy to target
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=1,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "modify_target": modify_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 50},
            ),
            "converter.blue": CppConverterConfig(
                type_id=100,
                type_name="converter",
                input_resources={0: 1},  # Input resource requirements
                output_resources={1: 1},  # Output resources produced
                max_output=100,
                max_conversions=1000,
                conversion_ticks=1,
                cooldown=5,
                initial_resource_count=5,
                color=1,
                recipe_details_obs=False,
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env = MettaGrid(cpp_config, game_map, 42)

    # Set up buffers
    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    # Get initial states
    objects = env.grid_objects()
    converter_id = None
    for oid, obj in objects.items():
        if obj.get("type") == 100:  # Converter type
            converter_id = oid
            break

    converter_energy_before = objects[converter_id].get("inventory", {}).get(0, 0)

    # Agent targets converter (offset: row=0, col=1)
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0x78]  # Target at offset (0, 1)

    env.step(actions)

    # Check that converter gained energy
    objects = env.grid_objects()
    converter_energy_after = objects[converter_id].get("inventory", {}).get(0, 0)

    # Converter auto-converts when it has resources, consuming 1 energy, so it has 9 left
    assert converter_energy_after == converter_energy_before + 9


def test_modify_target_self():
    """Test that ModifyTarget can affect the actor itself."""
    game_map = [
        ["wall", "wall", "wall"],
        ["wall", "agent.red", "wall"],
        ["wall", "wall", "wall"],
    ]

    modify_config = ModifyTargetConfig(
        required_resources={0: 1},  # Must have required_resources >= ceil(consumed_resources)
        consumed_resources={0: 1.0},
        modifies={1: 3.0},  # Heal self for 3
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=1,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "modify_target": modify_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 20},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env = MettaGrid(cpp_config, game_map, 42)

    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    objects = env.grid_objects()
    agent_id = None
    for oid, obj in objects.items():
        if obj.get("type") == 0:
            agent_id = oid
            break

    health_before = objects[agent_id].get("inventory", {}).get(1, 0)
    energy_before = objects[agent_id].get("inventory", {}).get(0, 0)

    # Target self (offset: row=0, col=0)
    # Encoding: ((0+7) << 4) | (0+7) = 0x77
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0x77]  # Target self

    env.step(actions)

    objects = env.grid_objects()
    health_after = objects[agent_id].get("inventory", {}).get(1, 0)
    energy_after = objects[agent_id].get("inventory", {}).get(0, 0)

    # Should have consumed energy and gained health
    assert energy_after == energy_before - 1
    assert health_after == health_before + 3


def test_modify_target_negative():
    """Test that ModifyTarget can apply negative modifications (damage)."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    modify_config = ModifyTargetConfig(
        required_resources={},
        consumed_resources={},
        modifies={1: -3.0},  # Deal 3 damage
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=2,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "modify_target": modify_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 50},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=1,
                group_name="blue",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 20},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env = MettaGrid(cpp_config, game_map, 42)

    observations = np.zeros((2, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(2, dtype=dtype_terminals)
    truncations = np.zeros(2, dtype=dtype_truncations)
    rewards = np.zeros(2, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    objects = env.grid_objects()
    blue_agent_id = None
    for oid, obj in objects.items():
        if obj.get("type") == 0 and obj["c"] == 2:
            blue_agent_id = oid
            break

    blue_health_before = objects[blue_agent_id].get("inventory", {}).get(1, 0)

    # Red targets blue (offset: row=0, col=1)
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0] = [action_idx, 0x78]
    actions[1] = [0, 0]

    env.step(actions)

    objects = env.grid_objects()
    blue_health_after = objects[blue_agent_id].get("inventory", {}).get(1, 0)

    # Blue should have lost health
    assert blue_health_after == blue_health_before - 3


def test_modify_target_out_of_bounds():
    """Test that ModifyTarget fails gracefully when target is out of bounds."""
    game_map = [
        ["wall", "wall", "wall"],
        ["wall", "agent.red", "wall"],
        ["wall", "wall", "wall"],
    ]

    modify_config = ModifyTargetConfig(
        required_resources={0: 1},  # Must have required_resources >= ceil(consumed_resources)
        consumed_resources={0: 1.0},
        modifies={1: 5.0},
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=1,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "modify_target": modify_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 10, 1: 50},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env = MettaGrid(cpp_config, game_map, 42)

    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    objects = env.grid_objects()
    agent_id = None
    for oid, obj in objects.items():
        if obj.get("type") == 0:
            agent_id = oid
            break

    energy_before = objects[agent_id].get("inventory", {}).get(0, 0)

    # Try to target far out of bounds (offset: row=8, col=8) - max offset
    # Encoding: ((8+7) << 4) | (8+7) = 0xFF (15, 15)
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0xFF]

    env.step(actions)

    # Check that action failed (no energy consumed)
    objects = env.grid_objects()
    energy_after = objects[agent_id].get("inventory", {}).get(0, 0)

    # Energy should be unchanged since action failed
    assert energy_after == energy_before


def test_modify_target_negative_offsets():
    """Test that ModifyTarget properly handles negative offsets at grid edges."""
    # Create a small map with agent at edge
    game_map = [
        ["wall", "wall", "wall"],
        ["agent.red", "empty", "wall"],  # Agent at left edge
        ["wall", "wall", "wall"],
    ]

    modify_config = ModifyTargetConfig(
        required_resources={0: 1},
        consumed_resources={0: 1.0},
        modifies={1: 5.0},
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=1,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "modify_target": modify_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 10, 1: 50},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env = MettaGrid(cpp_config, game_map, 42)

    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    objects = env.grid_objects()
    agent_id = None
    for oid, obj in objects.items():
        if obj.get("type") == 0:
            agent_id = oid
            break

    energy_before = objects[agent_id].get("inventory", {}).get(0, 0)

    # Try to target position with negative offset (row=0, col=-1)
    # This would be out of bounds to the left
    # Encoding: ((0+7) << 4) | (-1+7) = 0x76
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0x76]

    env.step(actions)

    # Check that action failed (no energy consumed)
    objects = env.grid_objects()
    energy_after = objects[agent_id].get("inventory", {}).get(0, 0)

    # Energy should be unchanged since action failed due to out of bounds
    assert energy_after == energy_before


def test_modify_target_wall_failure():
    """Test that ModifyTarget fails when targeting a wall and doesn't consume resources."""
    # Create a map with agent next to wall
    game_map = [
        ["wall", "wall", "wall"],
        ["wall", "agent.red", "wall"],
        ["wall", "wall", "wall"],
    ]

    modify_config = ModifyTargetConfig(
        required_resources={0: 1},
        consumed_resources={0: 1.0},
        modifies={1: 5.0},
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=1,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "modify_target": modify_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 10, 1: 50},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env = MettaGrid(cpp_config, game_map, 42)

    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    objects = env.grid_objects()
    agent_id = None
    for oid, obj in objects.items():
        if obj.get("type") == 0:
            agent_id = oid
            break

    energy_before = objects[agent_id].get("inventory", {}).get(0, 0)

    # Try to target the wall to the right (row=0, col=1)
    # Encoding: ((0+7) << 4) | (1+7) = 0x78
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0x78]

    env.step(actions)

    # Check that action failed (no energy consumed)
    objects = env.grid_objects()
    energy_after = objects[agent_id].get("inventory", {}).get(0, 0)

    # Energy should be unchanged since action failed due to targeting a wall
    assert energy_after == energy_before


def test_modify_target_invalid_magnitude():
    """Test that ModifyTargetConfig throws on invalid magnitudes."""
    # Test NaN value
    with pytest.raises(Exception):
        ModifyTargetConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: float('nan')}
        )

    # Test value too large (>255)
    with pytest.raises(Exception):
        ModifyTargetConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: 1000.0}
        )

    # Test negative value too large (abs > 255)
    with pytest.raises(Exception):
        ModifyTargetConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: -500.0}
        )

    # Test infinity
    with pytest.raises(Exception):
        ModifyTargetConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: float('inf')}
        )

    # Test that valid values work
    config = ModifyTargetConfig(
        required_resources={},
        consumed_resources={},
        modifies={0: 255.0, 1: -255.0, 2: 0.5}
    )
    assert config.modifies[0] == 255.0
    assert config.modifies[1] == -255.0
    assert config.modifies[2] == 0.5
