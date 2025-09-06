"""Test ResourceModAction with floating-point resource values."""

import numpy as np

from metta.mettagrid.mettagrid_c import (
    ActionConfig as CppActionConfig,
)
from metta.mettagrid.mettagrid_c import (
    AgentConfig as CppAgentConfig,
)
from metta.mettagrid.mettagrid_c import (
    GameConfig as CppGameConfig,
)
from metta.mettagrid.mettagrid_c import (
    GlobalObsConfig as CppGlobalObsConfig,
)
from metta.mettagrid.mettagrid_c import (
    MettaGrid,
    ResourceModActionConfig,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from metta.mettagrid.mettagrid_c import (
    ConverterConfig as CppConverterConfig,
)
from metta.mettagrid.mettagrid_c import (
    WallConfig as CppWallConfig,
)


def test_resource_mod_config_creation():
    """Test that ResourceModActionConfig can be created and configured properly."""
    # Test basic config creation
    config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 0.2},
        modifies={1: 0.1, 2: -1.0},
        scales=False,
        agent_radius=3,
        converter_radius=2,
    )

    # Check that properties are accessible
    assert not config.scales
    assert config.agent_radius == 3
    assert config.converter_radius == 2
    assert 0 in config.consumes
    assert abs(config.consumes[0] - 0.2) < 0.001
    assert 1 in config.modifies
    assert abs(config.modifies[1] - 0.1) < 0.001
    assert 2 in config.modifies
    assert abs(config.modifies[2] - (-1.0)) < 0.001

    # Test with scaling enabled
    config_scaled = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 1.0},
        modifies={1: 1.0},
        scales=True,
        agent_radius=2,
        converter_radius=0,
    )

    assert config_scaled.scales
    assert config_scaled.agent_radius == 2
    assert config_scaled.converter_radius == 0


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

    # Create ResourceModActionConfig that consumes resources
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 5.0},  # Consume 5 units of resource 0 (energy)
        modifies={},  # No modifications to others
        scales=False,
        agent_radius=0,  # Don't affect other agents
        converter_radius=0,  # Don't affect converters
    )

    # Create C++ GameConfig directly
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
            "resource_mod": resource_mod_config,
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
    print(f"Initial energy: {initial_energy}")

    # Execute resource_mod action
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]
    env.step(actions)

    # Check that energy was consumed
    objects = env.grid_objects()
    final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    print(f"Final energy: {final_energy}")

    assert final_energy == initial_energy - 5, (
        f"Energy should be reduced by 5, but went from {initial_energy} to {final_energy}"
    )


def test_resource_mod_nearby_agents():
    """Test that ResourceMod affects nearby agents within radius."""
    # Create a map with multiple agents
    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "agent.blue", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "agent.green", "empty", "empty", "empty", "agent.yellow", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    # Create ResourceModActionConfig that heals nearby agents
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 10.0},  # Consume 10 energy from actor
        modifies={1: 5.0},  # Add 5 health to nearby agents
        scales=False,  # Don't scale by number of targets
        agent_radius=2,  # Affect agents within radius 2
        converter_radius=0,  # Don't affect converters
    )

    # Create C++ GameConfig directly
    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=4,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "resource_mod": resource_mod_config,
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
            "agent.green": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=2,
                group_name="green",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 10},
            ),
            "agent.yellow": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=3,
                group_name="yellow",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 10},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    # Create environment
    env = MettaGrid(cpp_config, game_map, 42)

    # Set up buffers
    observations = np.zeros((4, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(4, dtype=dtype_terminals)
    truncations = np.zeros(4, dtype=dtype_truncations)
    rewards = np.zeros(4, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    # Get initial agent states
    objects = env.grid_objects()
    agents_info = []
    for oid, obj in objects.items():
        if obj.get("type") == 0:  # Agent type
            agents_info.append(
                {
                    "id": oid,
                    "pos": (obj["r"], obj["c"]),
                    "health_before": obj.get("inventory", {}).get(1, 0),
                    "energy_before": obj.get("inventory", {}).get(0, 0),
                }
            )

    # Sort by position for consistent ordering
    agents_info.sort(key=lambda a: a["pos"])

    print("Initial agent states:")
    for agent in agents_info:
        print(f"  Agent at {agent['pos']}: energy={agent['energy_before']}, health={agent['health_before']}")

    # Red agent (at position 1,1) uses resource_mod
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((4, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]  # Red agent uses resource_mod
    actions[1:] = [0, 0]  # Others do noop

    env.step(actions)

    # Check final states
    objects = env.grid_objects()
    for agent in agents_info:
        obj = objects[agent["id"]]
        agent["health_after"] = obj.get("inventory", {}).get(1, 0)
        agent["energy_after"] = obj.get("inventory", {}).get(0, 0)

        # Calculate Manhattan distance from red agent (1,1)
        distance = abs(agent["pos"][0] - 1) + abs(agent["pos"][1] - 1)
        agent["distance"] = distance

    print("\nFinal agent states:")
    for agent in agents_info:
        health_change = agent["health_after"] - agent["health_before"]
        energy_change = agent["energy_after"] - agent["energy_before"]
        print(
            f"  Agent at {agent['pos']} (distance={agent['distance']}): "
            f"energy {agent['energy_before']}→{agent['energy_after']} ({energy_change:+d}), "
            f"health {agent['health_before']}→{agent['health_after']} "
            f"({health_change:+d})"
        )

    # Verify results
    # Red agent at (1,1) should have consumed energy and gained health
    red_agent = agents_info[0]
    assert red_agent["pos"] == (1, 1)
    assert red_agent["energy_after"] == red_agent["energy_before"] - 10, "Red agent should consume 10 energy"
    assert red_agent["health_after"] == red_agent["health_before"] + 5, "Red agent should gain 5 health (affects self)"

    # Blue agent at (1,3) is within radius 2 (distance = 2)
    blue_agent = agents_info[1]
    assert blue_agent["pos"] == (1, 3)
    assert blue_agent["distance"] == 2
    assert blue_agent["health_after"] == blue_agent["health_before"] + 5, "Blue agent should gain 5 health"

    # Green agent at (3,1) is within radius 2 (distance = 2)
    green_agent = agents_info[2]
    assert green_agent["pos"] == (3, 1)
    assert green_agent["distance"] == 2
    assert green_agent["health_after"] == green_agent["health_before"] + 5, "Green agent should gain 5 health"

    # Yellow agent at (3,5) is outside radius 2 (distance = 6)
    yellow_agent = agents_info[3]
    assert yellow_agent["pos"] == (3, 5)
    assert yellow_agent["distance"] == 6
    assert yellow_agent["health_after"] == yellow_agent["health_before"], "Yellow agent should not be affected"


def test_resource_mod_scaling():
    """Test that ResourceMod scales modifications when scaling is enabled."""
    # Create a map with multiple agents close together
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "agent.green", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Create ResourceModActionConfig with scaling
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 6.0},  # Consume 6 energy from actor
        modifies={1: 12.0},  # Total 12 health to distribute
        scales=True,  # Scale by number of targets
        agent_radius=1,  # Affect agents within radius 1
        converter_radius=0,  # Don't affect converters
    )

    # Create C++ GameConfig directly
    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=3,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "resource_mod": resource_mod_config,
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
            "agent.green": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=2,
                group_name="green",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 20},
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    # Create environment
    env = MettaGrid(cpp_config, game_map, 42)

    # Set up buffers
    observations = np.zeros((3, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(3, dtype=dtype_terminals)
    truncations = np.zeros(3, dtype=dtype_truncations)
    rewards = np.zeros(3, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    env.reset()

    # Get initial agent states
    objects = env.grid_objects()
    agents_info = []
    for oid, obj in objects.items():
        if obj.get("type") == 0:  # Agent type
            agents_info.append(
                {
                    "id": oid,
                    "pos": (obj["r"], obj["c"]),
                    "health_before": obj.get("inventory", {}).get(1, 0),
                }
            )

    agents_info.sort(key=lambda a: a["pos"])

    print("Initial positions:")
    for agent in agents_info:
        print(f"  Agent at {agent['pos']}: health={agent['health_before']}")

    # Red agent uses resource_mod
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((3, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]  # Red agent uses resource_mod
    actions[1:] = [0, 0]  # Others do noop

    env.step(actions)

    # Check final states
    objects = env.grid_objects()
    total_health_added = 0
    affected_count = 0

    for agent in agents_info:
        obj = objects[agent["id"]]
        agent["health_after"] = obj.get("inventory", {}).get(1, 0)

        # Calculate Manhattan distance from red agent (1,1)
        distance = abs(agent["pos"][0] - 1) + abs(agent["pos"][1] - 1)
        agent["distance"] = distance

        health_change = agent["health_after"] - agent["health_before"]

        if distance <= 1:  # Within radius
            total_health_added += health_change
            affected_count += 1
            print(
                f"  Agent at {agent['pos']} (distance={distance}): health {agent['health_before']}→{agent['health_after']} ({health_change:+d})"
            )

    print(f"\nTotal health distributed: {total_health_added}")
    print(f"Affected agents: {affected_count}")

    # With scaling, 12 health should be distributed among affected agents
    # Red at (1,1) and Blue at (1,2) should each get 12/2 = 6 health
    # Green at (1,3) is distance 2, outside radius 1

    red_agent = agents_info[0]
    assert red_agent["pos"] == (1, 1)
    assert red_agent["health_after"] == red_agent["health_before"] + 6, "Red agent should gain 6 health (12/2 agents)"

    blue_agent = agents_info[1]
    assert blue_agent["pos"] == (1, 2)
    assert blue_agent["distance"] == 1
    assert blue_agent["health_after"] == blue_agent["health_before"] + 6, (
        "Blue agent should gain 6 health (12/2 agents)"
    )

    green_agent = agents_info[2]
    assert green_agent["pos"] == (1, 3)
    assert green_agent["distance"] == 2
    assert green_agent["health_after"] == green_agent["health_before"], (
        "Green agent should not be affected (outside radius)"
    )


def test_resource_mod_fractional_amounts():
    """Test that fractional resource amounts are handled with probabilistic rounding."""
    # Create a simple map with one agent
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Test fractional consumption and modification
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 2.3},  # Consume 2.3 energy (should round to 2 or 3)
        modifies={1: 1.7},  # Add 1.7 health (should round to 1 or 2)
        scales=False,
        agent_radius=0,  # Only affect self
        converter_radius=0,
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
            "resource_mod": resource_mod_config,
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
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    # Run multiple times to test probabilistic rounding
    env = MettaGrid(cpp_config, game_map, 42)
    observations = np.zeros((1, 100, 3), dtype=dtype_observations)
    terminals = np.zeros(1, dtype=dtype_terminals)
    truncations = np.zeros(1, dtype=dtype_truncations)
    rewards = np.zeros(1, dtype=dtype_rewards)
    env.set_buffers(observations, terminals, truncations, rewards)

    # Track results over multiple runs
    energy_consumed = []
    health_gained = []

    for seed in range(10):
        env = MettaGrid(cpp_config, game_map, seed)
        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        objects = env.grid_objects()
        agent_id = next(oid for oid, obj in objects.items() if obj.get("type") == 0)
        initial_energy = objects[agent_id].get("inventory", {}).get(0, 0)
        initial_health = objects[agent_id].get("inventory", {}).get(1, 0)

        action_idx = env.action_names().index("resource_mod")
        actions = np.zeros((1, 2), dtype=np.int32)
        actions[0] = [action_idx, 0]
        env.step(actions)

        objects = env.grid_objects()
        final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
        final_health = objects[agent_id].get("inventory", {}).get(1, 0)

        energy_consumed.append(initial_energy - final_energy)
        health_gained.append(final_health - initial_health)

    # Check that we get both 2 and 3 for consumption (2.3 rounds probabilistically)
    assert 2 in energy_consumed, "Should sometimes round down to 2"
    assert 3 in energy_consumed, "Should sometimes round up to 3"

    # Health gain should be 1 or 2, due to probabilistic rounding of 1.7
    assert 1 in health_gained, "Should sometimes round down to 1"
    assert 2 in health_gained, "Should sometimes round up to 2"

    print(f"Energy consumed over 10 runs: {energy_consumed}")
    print(f"Health gained over 10 runs: {health_gained}")

def test_resource_mod_negative_modifications():
    """Test that ResourceMod can apply negative modifications (damage/drain)."""
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Create ResourceModActionConfig that drains health from nearby agents
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 3.0},  # Consume 3 energy to cast
        modifies={1: -8.0},  # Remove 8 health from targets
        scales=False,
        agent_radius=1,  # Affect agents within radius 1
        converter_radius=0,
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
            "resource_mod": resource_mod_config,
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
                initial_inventory={0: 50, 1: 30},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=1,
                group_name="blue",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 30},
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

    # Get initial states
    objects = env.grid_objects()
    agents = []
    for oid, obj in objects.items():
        if obj.get("type") == 0:
            agents.append({
                "id": oid,
                "pos": (obj["r"], obj["c"]),
                "health_before": obj.get("inventory", {}).get(1, 0),
            })
    agents.sort(key=lambda a: a["pos"])

    print("Initial states:")
    for agent in agents:
        print(f"  Agent at {agent['pos']}: health={agent['health_before']}")

    # Red agent uses damaging resource_mod
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]  # Red agent uses resource_mod
    actions[1] = [0, 0]  # Blue agent does noop

    env.step(actions)

    # Check final states
    objects = env.grid_objects()
    for agent in agents:
        obj = objects[agent["id"]]
        agent["health_after"] = obj.get("inventory", {}).get(1, 0)

    print("\nFinal states:")
    for agent in agents:
        health_change = agent["health_after"] - agent["health_before"]
        print(f"  Agent at {agent['pos']}: health {agent['health_before']}→{agent['health_after']} ({health_change:+d})")

    # Both agents should lose 8 health (red affects self and blue)
    for agent in agents:
        assert agent["health_after"] == agent["health_before"] - 8, (
            f"Agent at {agent['pos']} should lose 8 health"
        )


def test_resource_mod_insufficient_resources():
    """Test that ResourceMod behaves correctly when actor has insufficient resources."""
    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "wall"],
        ["wall", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    # Create ResourceModActionConfig that requires more than agent has
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 15.0},  # Try to consume 15 energy
        modifies={1: 5.0},
        scales=False,
        agent_radius=1,  # Include self for modifies to work
        converter_radius=0,
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
            "resource_mod": resource_mod_config,
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
                initial_inventory={0: 10, 1: 50},  # Only 10 energy
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

    # Get initial state
    objects = env.grid_objects()
    agent_id = next(oid for oid, obj in objects.items() if obj.get("type") == 0)
    initial_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    initial_health = objects[agent_id].get("inventory", {}).get(1, 0)

    print(f"Initial: energy={initial_energy}, health={initial_health}")

    # Try to use resource_mod with insufficient energy
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]
    env.step(actions)

    # Check final state
    objects = env.grid_objects()
    final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    final_health = objects[agent_id].get("inventory", {}).get(1, 0)

    print(f"Final: energy={final_energy}, health={final_health}")

    # Based on implementation, when update_inventory returns less than requested,
    # the action returns false and doesn't apply modifies.
    # The actor can only provide 10 energy, so it consumes those 10
    # but the action fails and doesn't apply the health modification
    assert final_energy == 0, f"Energy should be consumed down to 0, but is {final_energy}"
    assert final_health == initial_health, f"Health should be unchanged when action fails, but went from {initial_health} to {final_health}"


def test_resource_mod_nearby_converters():
    """Test that ResourceMod affects nearby converters within radius.

    Note: Converters with empty inputs/outputs are used here as inventory-holding
    objects that can be targeted by ResourceMod.
    """
    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "converter.a", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "converter.b", "empty", "empty", "converter.c", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    resource_mod_config = ResourceModActionConfig(
        consumes={0: 5.0},
        modifies={1: 10.0},
        scales=False,
        agent_radius=0,
        converter_radius=2,
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=1,
        episode_truncates=False,
        obs_width=6,
        obs_height=6,
        num_observation_tokens=100,
        resource_names=["energy", "mana"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(),
            "resource_mod": resource_mod_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                initial_inventory={0: 20, 1: 0},
            ),
            "converter.a": CppConverterConfig(
                type_id=2,
                type_name="converter",
                input_resources={},  # Empty - used as storage container
                output_resources={},  # Empty - not performing conversions
                max_output=-1,
                max_conversions=-1,
                conversion_ticks=0,
                cooldown=0,
            ),
            "converter.b": CppConverterConfig(
                type_id=2,
                type_name="converter",
                input_resources={},
                output_resources={},
                max_output=-1,
                max_conversions=-1,
                conversion_ticks=0,
                cooldown=0,
            ),
            "converter.c": CppConverterConfig(
                type_id=2,
                type_name="converter",
                input_resources={},
                output_resources={},
                max_output=-1,
                max_conversions=-1,
                conversion_ticks=0,
                cooldown=0,
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
    converters_info = []
    for oid, obj in objects.items():
        if obj.get("type_name") == "converter":
            converters_info.append({
                "id": oid,
                "pos": (obj["r"], obj["c"]),
                "mana_before": obj.get("inventory", {}).get(1, 0),
            })
    converters_info.sort(key=lambda c: c["pos"])

    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]
    env.step(actions)

    objects = env.grid_objects()
    for conv in converters_info:
        obj = objects[conv["id"]]
        conv["mana_after"] = obj.get("inventory", {}).get(1, 0)
        distance = abs(conv["pos"][0] - 1) + abs(conv["pos"][1] - 1)
        conv["distance"] = distance

    # Converter A at (1,3), dist 2, should be affected
    assert converters_info[0]["distance"] == 2
    assert converters_info[0]["mana_after"] == converters_info[0]["mana_before"] + 10

    # Converter B at (3,1), dist 2, should be affected
    assert converters_info[1]["distance"] == 2
    assert converters_info[1]["mana_after"] == converters_info[1]["mana_before"] + 10

    # Converter C at (3,4), dist 5, should NOT be affected
    assert converters_info[2]["distance"] == 5
    assert converters_info[2]["mana_after"] == converters_info[2]["mana_before"]


def test_resource_mod_scaling_with_converters():
    """Test that ResourceMod scales modifications with agents and converters.

    Uses converters as inventory holders to test that ResourceMod correctly
    scales its effects across both agent and converter targets.
    """
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", "converter.a", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    resource_mod_config = ResourceModActionConfig(
        consumes={0: 1.0},
        modifies={1: 12.0},
        scales=True,
        agent_radius=1,
        converter_radius=2,
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=2,
        episode_truncates=False,
        obs_width=5,
        obs_height=5,
        num_observation_tokens=100,
        resource_names=["energy", "mana"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(),
            "resource_mod": resource_mod_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0, type_name="agent", group_id=0, group_name="red",
                initial_inventory={0: 20, 1: 10},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0, type_name="agent", group_id=1, group_name="blue",
                initial_inventory={0: 20, 1: 10},
            ),
            "converter.a": CppConverterConfig(
                type_id=2, type_name="converter",
                input_resources={},
                output_resources={},
                max_output=-1,
                max_conversions=-1,
                conversion_ticks=0,
                cooldown=0,
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
    initial_states = {}
    for oid, obj in objects.items():
        initial_states[oid] = {
            "type_name": obj.get("type_name", obj.get("type")),
            "pos": (obj["r"], obj["c"]),
            "mana_before": obj.get("inventory", {}).get(1, 0),
        }

    # Red agent at (1,1) performs the action
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0] = [action_idx, 0] # red agent
    actions[1] = [0, 0] # blue agent noop
    env.step(actions)

    objects = env.grid_objects()

    # Total affected should be 3: agent.red, agent.blue, converter.a
    # Modification should be 12.0 / 3 = 4.0 for each
    for oid, obj in objects.items():
        final_mana = obj.get("inventory", {}).get(1, 0)
        initial_mana = initial_states[oid]["mana_before"]

        type_name = initial_states[oid]["type_name"]
        if type_name in ["agent", "converter"]:
            pos = initial_states[oid]["pos"]
            distance = abs(pos[0] - 1) + abs(pos[1] - 1)

            # agent.red at (1,1), dist 0, radius 1 -> affected
            # agent.blue at (1,2), dist 1, radius 1 -> affected
            # converter.a at (1,3), dist 2, radius 2 -> affected
            assert final_mana == initial_mana + 4, f"{type_name} at {pos} should have gained 4 mana"

def test_resource_mod_required_consumed_resources():
    """Test that ResourceMod respects required_resources and consumed_resources from base ActionConfig."""
    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "wall"],
        ["wall", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    # Create ResourceModActionConfig with required and consumed resources
    resource_mod_config = ResourceModActionConfig(
        required_resources={0: 5},  # Require at least 5 energy to use
        consumed_resources={2: 2},  # Also consume 2 of resource 2
        consumes={0: 3.0},  # Additionally consume 3 energy via float system
        modifies={1: 10.0},  # Add 10 health
        scales=False,
        agent_radius=0,
        converter_radius=0,
    )

    cpp_config = CppGameConfig(
        max_steps=10,
        num_agents=1,
        episode_truncates=False,
        obs_width=3,
        obs_height=3,
        num_observation_tokens=100,
        resource_names=["energy", "health", "mana"],
        track_movement_metrics=False,
        actions={
            "noop": CppActionConfig(required_resources={}, consumed_resources={}),
            "resource_mod": resource_mod_config,
        },
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=10,
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100, 2: 100},
                initial_inventory={0: 20, 1: 50, 2: 10},
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

    # Get initial state
    objects = env.grid_objects()
    agent_id = next(oid for oid, obj in objects.items() if obj.get("type") == 0)
    initial_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    initial_health = objects[agent_id].get("inventory", {}).get(1, 0)
    initial_mana = objects[agent_id].get("inventory", {}).get(2, 0)

    print(f"Initial: energy={initial_energy}, health={initial_health}, mana={initial_mana}")

    # Use resource_mod
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]
    env.step(actions)

    # Check final state
    objects = env.grid_objects()
    final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    final_health = objects[agent_id].get("inventory", {}).get(1, 0)
    final_mana = objects[agent_id].get("inventory", {}).get(2, 0)

    print(f"Final: energy={final_energy}, health={final_health}, mana={final_mana}")

    # Verify resources were consumed correctly
    # Based on test output: energy went from 20 to 17 (consumed 3), mana from 10 to 8 (consumed 2)
    # This shows consumed_resources from ActionConfig are applied separately from consumes
    assert final_energy == initial_energy - 3, f"Energy should be reduced by 3 from consumes, but went from {initial_energy} to {final_energy}"
    assert final_mana == initial_mana - 2, f"Mana should be reduced by 2 from consumed_resources, but went from {initial_mana} to {final_mana}"
    # Health should increase by 10, as agent_radius=0 now affects self
    assert final_health == initial_health + 10, f"Health should increase by 10, but went from {initial_health} to {final_health}"
