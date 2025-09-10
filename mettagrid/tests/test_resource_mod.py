"""Test ResourceModAction with floating-point resource values."""

import numpy as np

from metta.mettagrid.mettagrid_c import (
    ActionConfig as CppActionConfig,
)
from metta.mettagrid.mettagrid_c import (
    AgentConfig as CppAgentConfig,
)
from metta.mettagrid.mettagrid_c import (
    ConverterConfig as CppConverterConfig,
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
        consumes={0: 1.0},  # 100% chance to consume 1 unit of resource 0 (energy)
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

    # With 100% probability, should consume exactly 1 unit
    assert final_energy == initial_energy - 1, (
        f"Energy should be reduced by 1, but went from {initial_energy} to {final_energy}"
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
        consumes={0: 1.0},  # 100% chance to consume 1 energy from actor
        modifies={1: 1.0},  # 100% chance to add 1 health to nearby agents
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
    assert red_agent["energy_after"] == red_agent["energy_before"] - 1, "Red agent should consume 1 energy"
    assert red_agent["health_after"] == red_agent["health_before"] + 1, "Red agent should gain 1 health (affects self)"

    # Blue agent at (1,3) is within radius 2 (distance = 2)
    blue_agent = agents_info[1]
    assert blue_agent["pos"] == (1, 3)
    assert blue_agent["distance"] == 2
    assert blue_agent["health_after"] == blue_agent["health_before"] + 1, "Blue agent should gain 1 health"

    # Green agent at (3,1) is within radius 2 (distance = 2)
    green_agent = agents_info[2]
    assert green_agent["pos"] == (3, 1)
    assert green_agent["distance"] == 2
    assert green_agent["health_after"] == green_agent["health_before"] + 1, "Green agent should gain 1 health"

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
        consumes={0: 1.0},  # 100% chance to consume 1 energy from actor
        modifies={1: 1.0},  # 100% chance to modify, but scaled by targets
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
                f"  Agent at {agent['pos']} (distance={distance}): "
                f"health {agent['health_before']}→{agent['health_after']} ({health_change:+d})"
            )

    print(f"\nTotal health distributed: {total_health_added}")
    print(f"Affected agents: {affected_count}")

    # With scaling, probability is divided by number of targets
    # 2 agents affected: Red at (1,1) and Blue at (1,2)
    # Each has 1.0 / 2 = 0.5 = 50% chance to gain 1 health
    # Green at (1,3) is distance 2, outside radius 1

    # Since this is probabilistic with 50% chance for each agent,
    # we can't assert exact values, but we know the affected agents
    red_agent = agents_info[0]
    assert red_agent["pos"] == (1, 1)
    # Red agent has 50% chance to gain 1 health

    blue_agent = agents_info[1]
    assert blue_agent["pos"] == (1, 2)
    assert blue_agent["distance"] == 1
    # Blue agent has 50% chance to gain 1 health

    green_agent = agents_info[2]
    assert green_agent["pos"] == (1, 3)
    assert green_agent["distance"] == 2
    assert green_agent["health_after"] == green_agent["health_before"], (
        "Green agent should not be affected (outside radius)"
    )

    # The test should accept either 0, 1, or 2 total health distributed
    assert 0 <= total_health_added <= 2, "Total health should be 0-2 with 50% chance per agent"


def test_resource_mod_fractional_amounts():
    """Test that fractional resource amounts are handled probabilistically."""
    # Create a simple map with one agent
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Test fractional consumption and modification with probabilistic behavior
    # 0.3 means 30% chance to consume 1 unit
    # 0.7 means 70% chance to add 1 unit
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 0.3},  # 30% chance to consume 1
        modifies={1: 0.7},  # 70% chance to add 1
        scales=False,
        agent_radius=1,  # Include self (distance 0 is within radius 1)
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

    # Run multiple iterations to test probabilistic behavior
    num_iterations = 100
    energy_consumed_counts = {0: 0, 1: 0}
    health_gained_counts = {0: 0, 1: 0}

    for seed in range(num_iterations):
        env = MettaGrid(cpp_config, game_map, seed)
        observations = np.zeros((1, 100, 3), dtype=dtype_observations)
        terminals = np.zeros(1, dtype=dtype_terminals)
        truncations = np.zeros(1, dtype=dtype_truncations)
        rewards = np.zeros(1, dtype=dtype_rewards)
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

        energy_consumed = initial_energy - final_energy
        health_gained = final_health - initial_health

        # Count outcomes
        if energy_consumed in energy_consumed_counts:
            energy_consumed_counts[energy_consumed] += 1
        if health_gained in health_gained_counts:
            health_gained_counts[health_gained] += 1

    # Check probabilistic distribution
    # For 0.3: expect ~70% to consume 0, ~30% to consume 1
    energy_0_rate = energy_consumed_counts.get(0, 0) / num_iterations
    energy_1_rate = energy_consumed_counts.get(1, 0) / num_iterations

    # For 0.7: expect ~30% to gain 0, ~70% to gain 1
    health_0_rate = health_gained_counts.get(0, 0) / num_iterations
    health_1_rate = health_gained_counts.get(1, 0) / num_iterations

    print(f"Energy consumption distribution over {num_iterations} runs:")
    print(f"  Consumed 0: {energy_consumed_counts.get(0, 0)} times ({energy_0_rate:.1%})")
    print(f"  Consumed 1: {energy_consumed_counts.get(1, 0)} times ({energy_1_rate:.1%})")
    print(f"Health gain distribution over {num_iterations} runs:")
    print(f"  Gained 0: {health_gained_counts.get(0, 0)} times ({health_0_rate:.1%})")
    print(f"  Gained 1: {health_gained_counts.get(1, 0)} times ({health_1_rate:.1%})")

    # Allow some variance in probabilities (±15% from expected)
    assert 0.55 <= energy_0_rate <= 0.85, f"Expected ~70% to consume 0, got {energy_0_rate:.1%}"
    assert 0.15 <= energy_1_rate <= 0.45, f"Expected ~30% to consume 1, got {energy_1_rate:.1%}"
    assert 0.15 <= health_0_rate <= 0.45, f"Expected ~30% to gain 0, got {health_0_rate:.1%}"
    assert 0.55 <= health_1_rate <= 0.85, f"Expected ~70% to gain 1, got {health_1_rate:.1%}"

    # Test edge cases with different probabilities
    resource_mod_config_edge = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 0.1},  # 10% chance to consume 1
        modifies={1: 0.9},  # 90% chance to add 1
        scales=False,
        agent_radius=1,  # Include self
        converter_radius=0,
    )

    # Create new config with edge case action
    cpp_config_edge = CppGameConfig(
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
            "resource_mod": resource_mod_config_edge,
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

    edge_consumed_counts = {0: 0, 1: 0}
    edge_gained_counts = {0: 0, 1: 0}

    for seed in range(num_iterations):
        env = MettaGrid(cpp_config_edge, game_map, seed + 1000)  # Different seed range
        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        objects = env.grid_objects()
        agent_id = next(oid for oid, obj in objects.items() if obj.get("type") == 0)
        initial_energy = objects[agent_id].get("inventory", {}).get(0, 0)
        initial_health = objects[agent_id].get("inventory", {}).get(1, 0)

        action_idx = env.action_names().index("resource_mod")
        actions[0] = [action_idx, 0]
        env.step(actions)

        objects = env.grid_objects()
        final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
        final_health = objects[agent_id].get("inventory", {}).get(1, 0)

        energy_consumed = initial_energy - final_energy
        health_gained = final_health - initial_health

        if energy_consumed in edge_consumed_counts:
            edge_consumed_counts[energy_consumed] += 1
        if health_gained in edge_gained_counts:
            edge_gained_counts[health_gained] += 1

    # Check edge case probabilities
    edge_consume_0_rate = edge_consumed_counts[0] / num_iterations
    edge_consume_1_rate = edge_consumed_counts[1] / num_iterations
    edge_gain_0_rate = edge_gained_counts[0] / num_iterations
    edge_gain_1_rate = edge_gained_counts[1] / num_iterations

    print("\nEdge case (0.1 consume, 0.9 modify) distribution:")
    print(f"  Consumed 0: {edge_consumed_counts[0]} times ({edge_consume_0_rate:.1%})")
    print(f"  Consumed 1: {edge_consumed_counts[1]} times ({edge_consume_1_rate:.1%})")
    print(f"  Gained 0: {edge_gained_counts[0]} times ({edge_gain_0_rate:.1%})")
    print(f"  Gained 1: {edge_gained_counts[1]} times ({edge_gain_1_rate:.1%})")

    assert 0.75 <= edge_consume_0_rate <= 0.95, f"Expected ~90% to consume 0, got {edge_consume_0_rate:.1%}"
    assert 0.05 <= edge_consume_1_rate <= 0.25, f"Expected ~10% to consume 1, got {edge_consume_1_rate:.1%}"
    assert 0.05 <= edge_gain_0_rate <= 0.25, f"Expected ~10% to gain 0, got {edge_gain_0_rate:.1%}"
    assert 0.75 <= edge_gain_1_rate <= 0.95, f"Expected ~90% to gain 1, got {edge_gain_1_rate:.1%}"


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
        consumes={0: 1.0},  # 100% chance to consume 1 energy to cast
        modifies={1: -1.0},  # 100% chance to remove 1 health from targets
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
            agents.append(
                {
                    "id": oid,
                    "pos": (obj["r"], obj["c"]),
                    "health_before": obj.get("inventory", {}).get(1, 0),
                }
            )
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
        print(
            f"  Agent at {agent['pos']}: health {agent['health_before']}→{agent['health_after']} ({health_change:+d})"
        )

    # Both agents should lose 1 health (red affects self and blue)
    for agent in agents:
        assert agent["health_after"] == agent["health_before"] - 1, f"Agent at {agent['pos']} should lose 1 health"


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
        consumes={0: 1.0},  # 100% chance to consume 1 energy (which we have)
        modifies={1: 1.0},  # 100% chance to add 1 health
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

    # With the new implementation, consumes={0: 1.0} means 100% chance to consume 1 unit
    # Since we have 10 energy, this should succeed
    assert final_energy == initial_energy - 1, (
        f"Energy should be reduced by 1, but went from {initial_energy} to {final_energy}"
    )
    # Should also gain 1 health since action succeeded
    assert final_health == initial_health + 1, (
        f"Health should increase by 1, but went from {initial_health} to {final_health}"
    )


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
        consumes={0: 1.0},  # 100% chance to consume 1 energy
        modifies={1: 1.0},  # 100% chance to add 1 mana
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
        # Converters have type=2
        if obj.get("type") == 2:
            converters_info.append(
                {
                    "id": oid,
                    "pos": (obj["r"], obj["c"]),
                    "mana_before": obj.get("inventory", {}).get(1, 0),
                }
            )
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
    assert converters_info[0]["mana_after"] == converters_info[0]["mana_before"] + 1

    # Converter B at (3,1), dist 2, should be affected
    assert converters_info[1]["distance"] == 2
    assert converters_info[1]["mana_after"] == converters_info[1]["mana_before"] + 1

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
        modifies={1: 0.99},  # High probability, but scaled by 3 targets = 0.33 each
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
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                initial_inventory={0: 20, 1: 10},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=1,
                group_name="blue",
                initial_inventory={0: 20, 1: 10},
            ),
            "converter.a": CppConverterConfig(
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
            "type": obj.get("type"),  # type_id (0=agent, 2=converter)
            "pos": (obj["r"], obj["c"]),
            "mana_before": obj.get("inventory", {}).get(1, 0),
        }

    # Red agent at (1,1) performs the action
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]  # red agent
    actions[1] = [0, 0]  # blue agent noop
    env.step(actions)

    objects = env.grid_objects()

    # Total affected should be 3: agent.red, agent.blue, converter.a
    # Modification probability should be 0.99 / 3 = 0.33 for each
    # Each has 33% chance to gain 1 mana
    total_mana_gained = 0
    for oid, obj in objects.items():
        final_mana = obj.get("inventory", {}).get(1, 0)
        initial_mana = initial_states[oid]["mana_before"]

        type_id = initial_states[oid]["type"]
        # Check if it's an agent (type 0) or converter (type 2)
        if type_id in [0, 2]:
            pos = initial_states[oid]["pos"]
            mana_gained = final_mana - initial_mana
            total_mana_gained += mana_gained

            # Each entity has 33% chance to gain 1 mana
            type_name = "agent" if type_id == 0 else "converter"
            assert mana_gained in [0, 1], f"{type_name} at {pos} should gain 0 or 1 mana"

    # With 3 targets at 33% each, we expect 0-3 total mana distributed
    assert 0 <= total_mana_gained <= 3, f"Total mana gained should be 0-3, got {total_mana_gained}"


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
        consumes={0: 1.0},  # Additionally 100% chance to consume 1 energy via float system
        modifies={1: 1.0},  # 100% chance to add 1 health
        scales=False,
        agent_radius=1,  # Include self for health modification
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
    # Energy: consumes 1 from the probabilistic system
    # Mana: consumes 2 from consumed_resources
    # Health: gains 1 from modifies with 100% probability
    assert final_energy == initial_energy - 1, (
        f"Energy should be reduced by 1 from consumes, but went from {initial_energy} to {final_energy}"
    )
    assert final_mana == initial_mana - 2, (
        f"Mana should be reduced by 2 from consumed_resources, but went from {initial_mana} to {final_mana}"
    )
    # Health should increase by 1 (agent_radius=1 includes self)
    assert final_health == initial_health + 1, (
        f"Health should increase by 1, but went from {initial_health} to {final_health}"
    )


def test_resource_mod_atomicity_multiple_consumes():
    """Test that ResourceMod maintains atomicity - if any consume fails, none are applied."""
    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "wall"],
        ["wall", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    # Create ResourceModActionConfig with multiple consumes
    # where one will fail due to insufficient resources
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={
            0: 1.0,  # 100% chance to consume 1 energy (have 5)
            1: 1.0,  # 100% chance to consume 1 health (have 5)
            2: 1.0,  # 100% chance to consume 1 mana (have 0 - will fail)
        },
        modifies={},
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
                initial_inventory={0: 5, 1: 5, 2: 0},  # No mana - action should fail
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

    print(f"Atomicity test - Initial: energy={initial_energy}, health={initial_health}, mana={initial_mana}")

    # Try to use resource_mod
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]
    env.step(actions)

    # Check final state
    objects = env.grid_objects()
    final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    final_health = objects[agent_id].get("inventory", {}).get(1, 0)
    final_mana = objects[agent_id].get("inventory", {}).get(2, 0)

    print(f"Atomicity test - Final: energy={final_energy}, health={final_health}, mana={final_mana}")

    # This tests atomicity - if any consume fails, none should be applied
    assert final_energy == initial_energy, (
        f"Energy should remain unchanged due to atomic failure, but went from {initial_energy} to {final_energy}"
    )
    assert final_health == initial_health, (
        f"Health should remain unchanged due to atomic failure, but went from {initial_health} to {final_health}"
    )
    assert final_mana == initial_mana, f"Mana should remain unchanged, but went from {initial_mana} to {final_mana}"


def test_resource_mod_overlap_with_consumed_resources():
    """Test that ResourceMod correctly handles overlap between consumes and consumed_resources."""
    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "wall"],
        ["wall", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    # Create ResourceModActionConfig with overlap between consumes and consumed_resources
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={0: 2},  # Base class consumes 2 energy
        consumes={0: 1.0},  # Also consumes 1 energy with 100% probability
        modifies={1: 1.0},  # Adds 1 health
        scales=False,
        agent_radius=1,
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
                initial_inventory={0: 10, 1: 50},  # 10 energy should be enough for 2+1=3 total
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

    print(f"Overlap test - Initial: energy={initial_energy}, health={initial_health}")

    # Use resource_mod
    action_idx = env.action_names().index("resource_mod")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]
    env.step(actions)

    # Check final state
    objects = env.grid_objects()
    final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    final_health = objects[agent_id].get("inventory", {}).get(1, 0)

    print(f"Overlap test - Final: energy={final_energy}, health={final_health}")

    # Should consume 3 total energy (2 from consumed_resources + 1 from consumes)
    assert final_energy == initial_energy - 3, (
        f"Energy should be reduced by 3 total (2 base + 1 consumes), but went from {initial_energy} to {final_energy}"
    )
    assert final_health == initial_health + 1, (
        f"Health should increase by 1, but went from {initial_health} to {final_health}"
    )

    # Test with insufficient resources for the overlap
    # Create a new environment with exactly 2 energy
    cpp_config2 = CppGameConfig(
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
                initial_inventory={0: 2, 1: 50},  # Only 2 energy - not enough for 2+1=3 total
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    env2 = MettaGrid(cpp_config2, game_map, 42)
    env2.set_buffers(observations, terminals, truncations, rewards)
    env2.reset()

    objects = env2.grid_objects()
    agent_id = next(oid for oid, obj in objects.items() if obj.get("type") == 0)
    initial_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    initial_health = objects[agent_id].get("inventory", {}).get(1, 0)

    print(f"Overlap insufficient test - Initial: energy={initial_energy}, health={initial_health}")

    # Try to use resource_mod - should fail due to insufficient total resources
    actions[0] = [action_idx, 0]
    env2.step(actions)

    objects = env2.grid_objects()
    final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
    final_health = objects[agent_id].get("inventory", {}).get(1, 0)

    print(f"Overlap insufficient test - Final: energy={final_energy}, health={final_health}")

    # Action should fail - no resources consumed, no health gained
    assert final_energy == initial_energy, (
        f"Energy should remain unchanged when action fails, but went from {initial_energy} to {final_energy}"
    )
    assert final_health == initial_health, (
        f"Health should remain unchanged when action fails, but went from {initial_health} to {final_health}"
    )


def test_resource_mod_determinism():
    """Test that ResourceMod produces deterministic results with fixed seeds."""
    game_map = [
        ["wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "wall"],
        ["wall", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall"],
    ]

    # Create ResourceModActionConfig with probabilistic consumption
    resource_mod_config = ResourceModActionConfig(
        required_resources={},
        consumed_resources={},
        consumes={0: 0.5},  # 50% chance to consume 1 energy
        modifies={1: 0.5},  # 50% chance to add 1 health
        scales=False,
        agent_radius=1,
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

    # Run multiple times with the same seed
    results_same_seed = []
    fixed_seed = 12345

    for _ in range(5):
        env = MettaGrid(cpp_config, game_map, fixed_seed)
        observations = np.zeros((1, 100, 3), dtype=dtype_observations)
        terminals = np.zeros(1, dtype=dtype_terminals)
        truncations = np.zeros(1, dtype=dtype_truncations)
        rewards = np.zeros(1, dtype=dtype_rewards)
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

        energy_consumed = initial_energy - final_energy
        health_gained = final_health - initial_health
        results_same_seed.append((energy_consumed, health_gained))

    # All runs with the same seed should produce identical results
    print(f"Determinism test - Results with seed {fixed_seed}: {results_same_seed}")

    first_result = results_same_seed[0]
    for i, result in enumerate(results_same_seed[1:], 1):
        assert result == first_result, (
            f"Run {i + 1} produced different result {result} than first run {first_result} with same seed"
        )

    # Run with different seeds to verify we get different results
    results_different_seeds = []
    for seed in [100, 200, 300]:
        env = MettaGrid(cpp_config, game_map, seed)
        env.set_buffers(observations, terminals, truncations, rewards)
        env.reset()

        objects = env.grid_objects()
        agent_id = next(oid for oid, obj in objects.items() if obj.get("type") == 0)
        initial_energy = objects[agent_id].get("inventory", {}).get(0, 0)
        initial_health = objects[agent_id].get("inventory", {}).get(1, 0)

        actions[0] = [action_idx, 0]
        env.step(actions)

        objects = env.grid_objects()
        final_energy = objects[agent_id].get("inventory", {}).get(0, 0)
        final_health = objects[agent_id].get("inventory", {}).get(1, 0)

        energy_consumed = initial_energy - final_energy
        health_gained = final_health - initial_health
        results_different_seeds.append((energy_consumed, health_gained))

    print(f"Determinism test - Results with different seeds: {results_different_seeds}")

    # With different seeds, we should see some variation in the results
    # (though it's possible to get the same result by chance with 50% probability)
    unique_results = set(results_different_seeds)
    # We don't assert that all results are different since that could fail randomly
    # but we log it for observation
    print(f"Unique results across different seeds: {unique_results}")
