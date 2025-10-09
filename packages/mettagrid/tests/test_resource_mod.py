# Test file for resource modification action with AoE implementation
"""Test resource modification with ResourceMod C++ action."""

import numpy as np
import pytest

from mettagrid.mettagrid_c import ActionConfig as CppActionConfig
from mettagrid.mettagrid_c import AgentConfig as CppAgentConfig
from mettagrid.mettagrid_c import ConverterConfig as CppConverterConfig
from mettagrid.mettagrid_c import GameConfig as CppGameConfig
from mettagrid.mettagrid_c import GlobalObsConfig as CppGlobalObsConfig
from mettagrid.mettagrid_c import InventoryConfig as CppInventoryConfig
from mettagrid.mettagrid_c import (
    MettaGrid,
    ResourceModConfig,
    dtype_actions,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.mettagrid_c import WallConfig as CppWallConfig


def test_resource_mod_config():
    """Test that ResourceModConfig can be created directly."""
    config = ResourceModConfig(
        required_resources={0: 5},
        consumed_resources={0: 1.5},
        modifies={1: 2.0, 2: -0.5},
        agent_radius=2,
        converter_radius=1,
        scales=True,
    )

    assert config.required_resources == {0: 5}
    assert config.consumed_resources == {0: 1.5}
    assert config.modifies == {1: 2.0, 2: -0.5}
    assert config.agent_radius == 2
    assert config.converter_radius == 1
    assert config.scales


def test_resource_mod_consumption():
    """Test that ResourceMod correctly consumes resources from the actor."""
    # Create a simple map with one agent
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Create ResourceModConfig that consumes resources
    resource_mod_config = ResourceModConfig(
        required_resources={0: 1},  # Need at least 1 energy to use
        consumed_resources={0: 1.0},  # 100% chance to consume 1 unit of resource 0 (energy)
        modifies={1: 5.0},  # Add 5 health
        agent_radius=1,  # Affects self (distance 0)
        converter_radius=0,
        scales=False,
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
        actions=[
            ("noop", CppActionConfig(required_resources={}, consumed_resources={})),
            ("resource_mod", resource_mod_config),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 10, 1: 50},
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
    initial_health = objects[agent_id].get("inventory", {}).get(1, 0)

    # Execute resource_mod action
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros(1, dtype=dtype_actions)
    actions[0] = action_idx
    env.step(actions)

    # Check that energy was consumed and health was added
    objects = env.grid_objects()
    final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    final_health = objects[agent_id].get("inventory", {}).get(1, 0)

    # With 100% probability, should consume exactly 1 unit and add 5 health
    assert final_energy == initial_energy - 1
    assert final_health == initial_health + 5


def test_resource_mod_aoe_agents():
    """Test using ResourceMod action with AoE affecting multiple agents."""
    # Create a map with multiple agents close together
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "agent.green", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Create a ResourceMod action that affects all agents within radius 2
    modify_config = ResourceModConfig(
        required_resources={0: 3},
        consumed_resources={0: 3.0},
        modifies={1: 6.0},  # Add 6 health total
        agent_radius=2,  # Affect agents within manhattan distance 2
        converter_radius=0,
        scales=True,  # Divide by number of affected agents
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=3,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions=[
            ("noop", CppActionConfig(required_resources={}, consumed_resources={})),
            ("resource_mod", modify_config),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 10},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=1,
                group_name="blue",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 10},
            ),
            "agent.green": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=2,
                group_name="green",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 10},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env = MettaGrid(cpp_config, game_map, 42)

    # Set up buffers
    observations = np.zeros((3, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(3, dtype=dtype_terminals)
    truncations = np.zeros(3, dtype=dtype_truncations)
    rewards = np.zeros(3, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    # Get initial states
    objects = env.grid_objects()
    agent_healths_before = {}
    red_energy_before = None

    for oid, obj in objects.items():
        if obj.get("type") == 0:
            agent_healths_before[oid] = obj.get("inventory", {}).get(1, 0)
            if obj["c"] == 1:  # Red agent
                red_energy_before = obj.get("inventory", {}).get(0, 0)

    # Red agent uses resource_mod (affects all 3 agents within radius 2)
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros(3, dtype=dtype_actions)
    actions[0] = action_idx  # Red uses resource_mod centered on self (arg ignored)

    env.step(actions)

    # Check results
    objects = env.grid_objects()
    agent_healths_after = {}
    red_energy_after = None

    for oid, obj in objects.items():
        if obj.get("type") == 0:
            agent_healths_after[oid] = obj.get("inventory", {}).get(1, 0)
            if obj["c"] == 1:  # Red agent
                red_energy_after = obj.get("inventory", {}).get(0, 0)

    # Red should have consumed energy
    assert red_energy_after == red_energy_before - 3

    # All agents should have gained health (6 total divided by 3 agents = 2 each)
    for oid in agent_healths_before:
        assert agent_healths_after[oid] == agent_healths_before[oid] + 2


def test_resource_mod_converters():
    """Test that ResourceMod can affect converters."""
    # Create a map with agent and converters
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "converter.blue", "agent.red", "converter.green", "wall"],
        ["wall", "empty", "converter.yellow", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    modify_config = ResourceModConfig(
        required_resources={},
        consumed_resources={},
        modifies={0: 12.0},  # Add 12 energy total
        agent_radius=0,  # Don't affect agents
        converter_radius=2,  # Affect converters within radius 2
        scales=True,  # Divide by number of affected converters (3)
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
        actions=[
            ("noop", CppActionConfig(required_resources={}, consumed_resources={})),
            ("resource_mod", modify_config),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
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
                cooldown_time=[5],
                initial_resource_count=0,
                recipe_details_obs=False,
            ),
            "converter.green": CppConverterConfig(
                type_id=101,
                type_name="converter",
                input_resources={0: 1},  # Input resource requirements
                output_resources={1: 1},  # Output resources produced
                max_output=100,
                max_conversions=1000,
                conversion_ticks=1,
                cooldown_time=[5],
                initial_resource_count=0,
                recipe_details_obs=False,
            ),
            "converter.yellow": CppConverterConfig(
                type_id=102,
                type_name="converter",
                input_resources={0: 1},  # Input resource requirements
                output_resources={1: 1},  # Output resources produced
                max_output=100,
                max_conversions=1000,
                conversion_ticks=1,
                cooldown_time=[5],
                initial_resource_count=0,
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
    converter_energies_before = {}
    for oid, obj in objects.items():
        if obj.get("type") in [100, 101, 102]:  # Converter types
            converter_energies_before[oid] = obj.get("inventory", {}).get(0, 0)

    # Agent uses resource_mod
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros(1, dtype=dtype_actions)
    actions[0] = action_idx

    env.step(actions)

    # Check that converters gained energy
    objects = env.grid_objects()
    for oid in converter_energies_before:
        obj = objects[oid]
        energy_after = obj.get("inventory", {}).get(0, 0)
        # Each converter should get 12/3 = 4 energy
        # But converters auto-convert when they have resources, consuming 1 energy
        # So they should have 3 left
        assert energy_after == converter_energies_before[oid] + 3


def test_resource_mod_negative():
    """Test that ResourceMod can apply negative modifications (damage)."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "agent.green", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    modify_config = ResourceModConfig(
        required_resources={},
        consumed_resources={},
        modifies={1: -6.0},  # Deal 6 damage total
        agent_radius=2,  # Affect all agents within radius 2
        converter_radius=0,
        scales=True,  # Divide by number of affected (3 agents = -2 each)
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=3,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions=[
            ("noop", CppActionConfig(required_resources={}, consumed_resources={})),
            ("resource_mod", modify_config),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 50},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=1,
                group_name="blue",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 20},
            ),
            "agent.green": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=2,
                group_name="green",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 20},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env = MettaGrid(cpp_config, game_map, 42)

    observations = np.zeros((3, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(3, dtype=dtype_terminals)
    truncations = np.zeros(3, dtype=dtype_truncations)
    rewards = np.zeros(3, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    objects = env.grid_objects()
    agent_healths_before = {}
    for oid, obj in objects.items():
        if obj.get("type") == 0:
            agent_healths_before[oid] = obj.get("inventory", {}).get(1, 0)

    # Red uses AoE damage
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros(3, dtype=dtype_actions)
    actions[0] = action_idx

    env.step(actions)

    objects = env.grid_objects()
    # All agents should have lost health (-6 divided by 3 = -2 each)
    for oid in agent_healths_before:
        obj = objects[oid]
        health_after = obj.get("inventory", {}).get(1, 0)
        assert health_after == agent_healths_before[oid] - 2


def test_resource_mod_scaling_vs_no_scaling():
    """Test the difference between scales=True and scales=False."""
    # Create two identical configs except for scaling
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Config with scaling
    config_with_scaling = ResourceModConfig(
        required_resources={},
        consumed_resources={},
        modifies={1: 10.0},
        agent_radius=1,
        converter_radius=0,
        scales=True,  # Divide by number of affected
    )

    # Config without scaling
    config_no_scaling = ResourceModConfig(
        required_resources={},
        consumed_resources={},
        modifies={1: 10.0},
        agent_radius=1,
        converter_radius=0,
        scales=False,  # Each gets full amount
    )

    # Test with scaling
    cpp_config_scale = CppGameConfig(
        max_steps=10,
        num_agents=2,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions=[
            ("noop", CppActionConfig(required_resources={}, consumed_resources={})),
            ("resource_mod", config_with_scaling),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 10},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=1,
                group_name="blue",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 10},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    # Test without scaling
    cpp_config_no_scale = CppGameConfig(
        max_steps=10,
        num_agents=2,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions=[
            ("noop", CppActionConfig(required_resources={}, consumed_resources={})),
            ("resource_mod", config_no_scaling),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 10},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=1,
                group_name="blue",
                freeze_duration=10,
                action_failure_penalty=0,
                inventory_config=CppInventoryConfig(limits=[[[0], 100], [[1], 100]]),
                initial_inventory={0: 50, 1: 10},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    # Test with scaling
    env_scale = MettaGrid(cpp_config_scale, game_map, 42)
    observations = np.zeros((2, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(2, dtype=dtype_terminals)
    truncations = np.zeros(2, dtype=dtype_truncations)
    rewards = np.zeros(2, dtype=dtype_rewards)
    env_scale.set_buffers(observations, terminals, truncations, rewards)
    env_scale.reset()

    action_idx = env_scale.action_names().index("resource_mod")
    actions = np.zeros(2, dtype=dtype_actions)
    actions[0] = action_idx
    env_scale.step(actions)

    # Check scaled results (10 divided by 2 agents = 5 each)
    objects_scale = env_scale.grid_objects()
    for _oid, obj in objects_scale.items():
        if obj.get("type") == 0:
            health = obj.get("inventory", {}).get(1, 0)
            assert health == 15  # Initial 10 + scaled 5

    # Test without scaling
    env_no_scale = MettaGrid(cpp_config_no_scale, game_map, 42)
    env_no_scale.set_buffers(observations, terminals, truncations, rewards)
    env_no_scale.reset()

    actions = np.zeros(2, dtype=dtype_actions)
    actions[0] = action_idx
    env_no_scale.step(actions)

    # Check non-scaled results (each gets full 10)
    objects_no_scale = env_no_scale.grid_objects()
    for _oid, obj in objects_no_scale.items():
        if obj.get("type") == 0:
            health = obj.get("inventory", {}).get(1, 0)
            assert health == 20  # Initial 10 + full 10


def test_resource_mod_invalid_magnitude():
    """Test that ResourceModConfig throws on invalid magnitudes."""
    # Test NaN value
    with pytest.raises(ValueError):
        ResourceModConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: float("nan")},
            agent_radius=1,
            converter_radius=0,
            scales=False,
        )

    # Test value too large (>255)
    with pytest.raises(ValueError):
        ResourceModConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: 1000.0},
            agent_radius=1,
            converter_radius=0,
            scales=False,
        )

    # Test negative value too large (abs > 255)
    with pytest.raises(ValueError):
        ResourceModConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: -500.0},
            agent_radius=1,
            converter_radius=0,
            scales=False,
        )

    # Test infinity
    with pytest.raises(ValueError):
        ResourceModConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: float("inf")},
            agent_radius=1,
            converter_radius=0,
            scales=False,
        )

    # Test that valid values work
    config = ResourceModConfig(
        required_resources={},
        consumed_resources={},
        modifies={0: 255.0, 1: -255.0, 2: 0.5},
        agent_radius=1,
        converter_radius=0,
        scales=False,
    )
    assert config.modifies[0] == 255.0
    assert config.modifies[1] == -255.0
    assert config.modifies[2] == 0.5
