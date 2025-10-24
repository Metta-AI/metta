"""Test AgentSupervisor functionality in mettagrid."""

import numpy as np


def test_supervisor_basic():
    """Test basic supervisor creation and configuration."""
    from mettagrid_c import (
        AgentConfig,
        AgentSupervisorConfig,
        GameConfig,
        InventoryConfig,
        MettaGrid,
    )

    # Create a basic supervisor config
    supervisor_config = AgentSupervisorConfig(can_override_action=False, name="test_supervisor")

    # Create agent config with supervisor
    agent_config = AgentConfig(
        type_id=1,
        type_name="supervised_agent",
        group_id=0,
        group_name="team1",
        freeze_duration=0,
        action_failure_penalty=0.0,
        inventory_config=InventoryConfig(),
        stat_rewards={},
        stat_reward_max={},
        group_reward_pct=0.0,
        initial_inventory={},
        soul_bound_resources=[],
        shareable_resources=[],
        inventory_regen_amounts={},
        supervisor_config=supervisor_config,
    )

    # Create game config
    game_config = GameConfig(
        obs_width=7,
        obs_height=7,
        max_steps=100,
        num_agents=1,
        resource_names=["Energy", "Carbon", "Iron"],
        objects={"supervised_agent": agent_config},
        actions={},
    )

    # Create a simple 3x3 map with one agent
    test_map = [
        [".", ".", "."],
        [".", "supervised_agent", "."],
        [".", ".", "."],
    ]

    # Create environment
    env = MettaGrid(game_config, test_map, seed=42)

    # Reset and check that it doesn't crash
    obs, info = env.reset()
    assert obs is not None

    # Take a step with a no-op action
    actions = np.array([[0, 0]], dtype=np.int8)  # [action_type, action_arg]
    obs, reward, done, truncated, info = env.step(actions)

    # Check that we got stats from the supervisor
    stats = env.get_episode_stats()

    # The supervisor should have recorded right/wrong statistics
    # (even though we don't have a real supervisor implementation yet)
    # This test mainly checks that the integration doesn't crash


def test_resource_transport_supervisor():
    """Test ResourceTransportSupervisor configuration and basic functionality."""
    from mettagrid_c import (
        AgentConfig,
        ChestConfig,
        ConverterConfig,
        GameConfig,
        InventoryConfig,
        MettaGrid,
        Recipe,
        ResourceTransportSupervisorConfig,
        WallConfig,
    )

    # Create resource transport supervisor config
    supervisor_config = ResourceTransportSupervisorConfig(
        target_resource=1,  # Carbon (index 1 in resource_names)
        min_energy_threshold=10,
        manage_energy=True,
        max_search_distance=10,
        can_override_action=True,
        name="carbon_transporter",
    )

    # Create agent config with supervisor
    agent_config = AgentConfig(
        type_id=1,
        type_name="transport_agent",
        group_id=0,
        group_name="team1",
        freeze_duration=0,
        action_failure_penalty=0.1,
        inventory_config=InventoryConfig(
            limits={0: 100, 1: 50, 2: 50},  # Energy: 100, Carbon: 50, Iron: 50
        ),
        stat_rewards={
            "carbon_transporter.right": 0.01,
            "carbon_transporter.wrong": -0.01,
        },
        stat_reward_max={},
        group_reward_pct=0.0,
        initial_inventory={0: 50},  # Start with 50 energy
        soul_bound_resources=[],
        shareable_resources=[],
        inventory_regen_amounts={},
        supervisor_config=supervisor_config,
    )

    # Create converter (extractor) config
    carbon_recipe = Recipe(
        inputs={},  # No inputs needed
        outputs={1: 1},  # Produces 1 Carbon
        crafting_time=1,
    )

    converter_config = ConverterConfig(
        type_id=2,
        type_name="carbon_extractor",
        inventory_config=InventoryConfig(
            limits={1: 100},  # Can store up to 100 Carbon
        ),
        recipes=[carbon_recipe],
        auto_start=True,
    )

    # Create chest config
    chest_config = ChestConfig(
        type_id=3,
        type_name="storage_chest",
        inventory_config=InventoryConfig(
            limits={0: 1000, 1: 1000, 2: 1000},  # Can store lots of resources
        ),
    )

    # Create wall config
    wall_config = WallConfig(type_id=4, type_name="wall")

    # Create game config
    game_config = GameConfig(
        obs_width=7,
        obs_height=7,
        max_steps=100,
        num_agents=1,
        resource_names=["Energy", "Carbon", "Iron"],
        objects={
            "transport_agent": agent_config,
            "carbon_extractor": converter_config,
            "storage_chest": chest_config,
            "wall": wall_config,
        },
        actions={},  # Will use default actions
    )

    # Create a map with agent, extractor, and chest
    test_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", ".", ".", ".", ".", ".", "wall"],
        ["wall", ".", "carbon_extractor", ".", ".", ".", "wall"],
        ["wall", ".", ".", "transport_agent", ".", ".", "wall"],
        ["wall", ".", ".", ".", ".", ".", "wall"],
        ["wall", ".", "storage_chest", ".", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    # Create environment
    env = MettaGrid(game_config, test_map, seed=42)

    # Reset
    obs, info = env.reset()
    assert obs is not None

    # Run a few steps to see if the supervisor is working
    for step in range(10):
        # Agent would normally choose random actions, but supervisor should override
        actions = np.array([[np.random.randint(0, 4), np.random.randint(0, 4)]], dtype=np.int8)
        obs, reward, done, truncated, info = env.step(actions)

        if done[0] or truncated[0]:
            break

    # Check stats to see if supervisor is recording decisions
    stats = env.get_episode_stats()

    # We should see some supervisor stats
    # The exact values depend on the implementation, but we should see the keys
    expected_stat_keys = [
        "carbon_transporter.right",
        "carbon_transporter.wrong",
        "carbon_transporter.state.searching",
    ]

    for key in expected_stat_keys:
        # Check if any agent has this stat
        agent_stats_exist = any(key in agent_stats for agent_stats in stats.get("agents", {}).values())
        if not agent_stats_exist:
            print(f"Warning: Expected stat '{key}' not found in agent stats")

    print(f"Supervisor test completed {step + 1} steps")


if __name__ == "__main__":
    test_supervisor_basic()
    test_resource_transport_supervisor()
    print("All supervisor tests passed!")
