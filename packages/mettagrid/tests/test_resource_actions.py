"""Consolidated tests for resource actions, inventory bounds, and ModifyTarget functionality."""

import numpy as np
import pytest

from mettagrid.config.mettagrid_c_config import convert_to_cpp_game_config
from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    GameConfig,
    ModifyTargetActionConfig,
)
from mettagrid.mettagrid_c import (
    ActionConfig as CppActionConfig,
)
from mettagrid.mettagrid_c import (
    AgentConfig as CppAgentConfig,
)
from mettagrid.mettagrid_c import (
    ConverterConfig as CppConverterConfig,
)
from mettagrid.mettagrid_c import (
    GameConfig as CppGameConfig,
)
from mettagrid.mettagrid_c import (
    GlobalObsConfig as CppGlobalObsConfig,
)
from mettagrid.mettagrid_c import (
    MettaGrid,
    ModifyTargetConfig,
    dtype_observations,
    dtype_rewards,
    dtype_terminals,
    dtype_truncations,
)
from mettagrid.mettagrid_c import (
    WallConfig as CppWallConfig,
)

# ============================================================================
# Inventory Bounds Tests
# ============================================================================


def test_inventory_overflow_protection():
    """Test that inventory values are clamped at 255 (max uint8_t)."""

    # Create a game config with modify_target that can add resources
    game_config = CppGameConfig(
        num_agents=1,
        obs_width=5,
        obs_height=5,
        max_steps=100,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=True,
        resource_names=["resource"],
        actions=[
            ("move", CppActionConfig()),
            (
                "modify_target",
                ModifyTargetConfig(
                    required_resources={},
                    consumed_resources={},
                    modifies={0: 20.0},  # Add 20 resources per action
                    agent_radius=1,
                    converter_radius=1,
                    scales=False,
                ),
            ),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                resource_limits={0: 255},
                initial_inventory={0: 250},  # Start near max
            ),
            "converter": CppConverterConfig(
                type_id=2,
                type_name="converter",
                input_resources={},
                output_resources={},
                max_output=-1,
                max_conversions=-1,
                conversion_ticks=1,
                cooldown=0,
                initial_resource_count=240,  # Start near max for converter
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    # Create a map with agent and converter
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "converter", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    env = MettaGrid(game_config, game_map, 0)
    obs, _ = env.reset()

    # Get initial state
    grid_objects = env.grid_objects()
    agent_id = [k for k, v in grid_objects.items() if v.get("type") == 0][0]
    converter_id = [k for k, v in grid_objects.items() if v.get("type") == 2][0]

    initial_agent_inventory = grid_objects[agent_id]["inventory"].get(0, 0)
    assert initial_agent_inventory == 250, f"Expected initial agent inventory 250, got {initial_agent_inventory}"

    # Try to add 20 resources when agent already has 250
    # Should clamp to 255, not overflow
    modify_target_idx = env.action_names().index("modify_target")
    actions = np.array([[modify_target_idx, 0]], dtype=np.int32)  # modify_target facing forward
    obs, rewards, dones, truncs, infos = env.step(actions)

    # Verify both agent and converter inventories are clamped at 255
    grid_objects = env.grid_objects()
    final_agent_inventory = grid_objects[agent_id]["inventory"].get(0, 0)
    final_converter_inventory = grid_objects[converter_id].get("inventory", {}).get(0, 0)

    assert final_agent_inventory == 255, f"Expected agent inventory clamped at 255, got {final_agent_inventory}"
    # Converter inventory might be modified if it's a target, verify it's also clamped
    assert final_converter_inventory <= 255, f"Converter inventory exceeded 255: {final_converter_inventory}"


def test_inventory_underflow_protection():
    """Test that inventory values are clamped at 0 (min for uint8_t)."""

    # Create a game config with modify_target that can subtract resources
    game_config = CppGameConfig(
        num_agents=1,
        obs_width=5,
        obs_height=5,
        max_steps=100,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=True,
        resource_names=["resource"],
        actions=[
            ("move", CppActionConfig()),
            (
                "modify_target",
                ModifyTargetConfig(
                    required_resources={},
                    consumed_resources={},
                    modifies={0: -10.0},  # Subtract 10 resources per action
                    agent_radius=1,
                    converter_radius=1,
                    scales=False,
                ),
            ),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                resource_limits={0: 255},
                initial_inventory={0: 5},  # Start with only 5
            ),
            "converter": CppConverterConfig(
                type_id=2,
                type_name="converter",
                input_resources={},
                output_resources={},
                max_output=-1,
                max_conversions=-1,
                conversion_ticks=1,
                cooldown=0,
                initial_resource_count=3,  # Start with only 3
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    # Create a map with agent and converter
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "converter", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    env = MettaGrid(game_config, game_map, 0)
    obs, _ = env.reset()

    # Get initial inventory state
    grid_objects = env.grid_objects()
    agent_id = [k for k, v in grid_objects.items() if v.get("type") == 0][0]  # Agent type_id is 0
    initial_inventory = grid_objects[agent_id]["inventory"].get(0, 0)
    assert initial_inventory == 5  # Verify starting condition
    # Try to subtract 10 resources when agent only has 5
    # Should clamp to 0, not underflow to negative or wrap around
    action_names = env.action_names()
    modify_target_idx = action_names.index("modify_target") if "modify_target" in action_names else 0
    actions = np.array([[modify_target_idx, 0]], dtype=np.int32)  # modify_target with arg 0
    obs, rewards, dones, truncs, infos = env.step(actions)

    # Check that value was properly clamped to 0
    grid_objects = env.grid_objects()
    final_inventory = grid_objects[agent_id].get("inventory", {}).get(0, 0)
    assert final_inventory == 0, f"Expected inventory to be clamped at 0, got {final_inventory}"


def test_inventory_edge_cases():
    """Test inventory behavior at exact boundaries."""

    game_config = CppGameConfig(
        num_agents=2,
        obs_width=5,
        obs_height=5,
        max_steps=100,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=True,
        resource_names=["resource"],
        actions=[
            ("move", CppActionConfig()),
            (
                "modify_target",
                ModifyTargetConfig(
                    required_resources={},
                    consumed_resources={},
                    modifies={0: 1.0},  # Add 1 resource
                    agent_radius=1,
                    converter_radius=1,
                    scales=False,
                ),
            ),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                resource_limits={0: 255},
                initial_inventory={0: 254},  # One below max
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                group_id=1,
                group_name="blue",
                resource_limits={0: 255},
                initial_inventory={0: 255},  # At max
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", "agent.blue", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    env = MettaGrid(game_config, game_map, 0)
    obs, _ = env.reset()

    # Get initial state
    grid_objects = env.grid_objects()
    agent_ids = [k for k, v in grid_objects.items() if v.get("type") == 0]
    assert len(agent_ids) == 2, f"Expected 2 agents, found {len(agent_ids)}"

    # Find which agent is which based on inventory
    red_agent_id = None
    blue_agent_id = None
    for agent_id in agent_ids:
        inventory = grid_objects[agent_id]["inventory"].get(0, 0)
        if inventory == 254:
            red_agent_id = agent_id
        elif inventory == 255:
            blue_agent_id = agent_id

    assert red_agent_id is not None, "Could not find red agent with 254 resources"
    assert blue_agent_id is not None, "Could not find blue agent with 255 resources"

    # Both agents try to add 1 resource
    # Agent.red: 254 + 1 = 255 (should succeed)
    # Agent.blue: 255 + 1 = 255 (should clamp, already at max)
    modify_target_idx = env.action_names().index("modify_target")
    actions = np.array([[modify_target_idx, 0], [modify_target_idx, 0]], dtype=np.int32)
    obs, rewards, dones, truncs, infos = env.step(actions)

    # Verify edge cases handled correctly
    grid_objects = env.grid_objects()
    red_final = grid_objects[red_agent_id]["inventory"].get(0, 0)
    blue_final = grid_objects[blue_agent_id]["inventory"].get(0, 0)

    assert red_final == 255, f"Red agent should have 255 (254+1), got {red_final}"
    assert blue_final == 255, f"Blue agent should remain at 255 (clamped), got {blue_final}"


# ============================================================================
# ModifyTarget Configuration Tests
# ============================================================================


def test_modify_target_config_creation():
    """Test that ModifyTargetActionConfig can be created properly."""
    config = ModifyTargetActionConfig(
        enabled=True,
        required_resources={"mana": 5},
        consumed_resources={"mana": 3.0},
        modifies={"health": 10.0, "gold": -5.0},
    )

    assert config.enabled is True
    assert config.required_resources == {"mana": 5}
    assert config.consumed_resources == {"mana": 3.0}
    assert config.modifies == {"health": 10.0, "gold": -5.0}


def test_modify_target_in_actions_config():
    """Test that modify_target can be added to ActionsConfig."""
    actions = ActionsConfig(
        move=ActionConfig(enabled=True), modify_target=ModifyTargetActionConfig(enabled=True, modifies={"health": 10.0})
    )

    assert actions.modify_target.enabled is True
    assert actions.modify_target.modifies == {"health": 10.0}


def test_modify_target_conversion_to_cpp():
    """Test that ModifyTargetActionConfig converts properly to C++ config."""
    # Create a GameConfig with modify_target action
    game_config = GameConfig(
        resource_names=["health", "mana", "gold"],
        num_agents=2,
        actions=ActionsConfig(
            move=ActionConfig(enabled=True),
            modify_target=ModifyTargetActionConfig(
                enabled=True,
                required_resources={"mana": 5},
                consumed_resources={"mana": 3.0},
                modifies={"health": 10.0, "gold": -5.0},
            ),
        ),
    )

    # Convert to C++ config - this should not raise an exception
    cpp_config = convert_to_cpp_game_config(game_config)

    # The conversion succeeded if we got here without exception
    assert cpp_config is not None


def test_modify_target_disabled_by_default():
    """Test that modify_target is disabled by default in ActionsConfig."""
    actions = ActionsConfig()
    assert actions.modify_target.enabled is False


def test_modify_target_end_to_end():
    """Test that modify_target action works correctly in MettaGrid."""
    import numpy as np

    import mettagrid.mettagrid_c as mc

    # Create game config with modify_target action
    game_config = mc.GameConfig(
        num_agents=1,
        obs_width=5,
        obs_height=5,
        max_steps=100,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=False,
        resource_names=["energy", "health"],
        actions=[
            ("move", mc.ActionConfig()),
            (
                "modify_target",
                mc.ModifyTargetConfig(
                    required_resources={0: 1},  # Requires 1 energy
                    consumed_resources={0: 0.5},  # Consumes 0.5 energy
                    modifies={1: 1.0},  # Modifies health by 1.0
                    agent_radius=1,
                    converter_radius=1,
                    scales=False,
                ),
            ),
        ],
        objects={
            "agent.red": mc.AgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                resource_limits={0: 255, 1: 255},
                initial_inventory={0: 10, 1: 50},  # 10 energy, 50 health
            ),
            "wall": mc.WallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=mc.GlobalObsConfig(),
    )

    # Create a simple 5x5 map
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Create MettaGrid directly
    env = mc.MettaGrid(game_config, game_map, 0)
    obs, _ = env.reset()

    # Verify modify_target action is present
    action_names = env.action_names()
    assert "modify_target" in action_names, f"modify_target not found in actions: {action_names}"
    modify_target_idx = action_names.index("modify_target")

    # Verify action effects when applied
    actions = np.array([[modify_target_idx, 0]], dtype=np.int32)
    obs, rewards, dones, truncs, infos = env.step(actions)

    # Check that the action was registered (even if no targets in range)
    assert env.action_success() is not None, "action_success should return results"


def test_modify_target_with_radius_parameters():
    """Test that ModifyTargetConfig can use agent_radius and converter_radius parameters."""
    import mettagrid.mettagrid_c as mc

    # Create ModifyTargetConfig with explicit radius parameters
    config = mc.ModifyTargetConfig(
        required_resources={0: 5},
        consumed_resources={0: 1.0},
        modifies={1: 2.0},
        agent_radius=2,
        converter_radius=1,
        scales=False,
    )

    assert config.agent_radius == 2
    assert config.converter_radius == 1
    assert config.scales is False
    assert config.modifies == {1: 2.0}


def test_modify_target_scales_true():
    """Test ModifyTargetConfig with scales=True for splitting effects."""
    import mettagrid.mettagrid_c as mc

    # Create config with scales=True
    config = mc.ModifyTargetConfig(
        required_resources={},
        consumed_resources={},
        modifies={0: 10.0},  # 10 points to distribute
        agent_radius=3,
        converter_radius=3,
        scales=True,  # Effect will be divided among targets
    )

    assert config.scales is True
    assert config.modifies == {0: 10.0}
    # With scales=True, the 10.0 effect will be divided by number of targets


def test_modify_target_different_radii():
    """Test ModifyTargetConfig with different agent and converter radii."""
    import mettagrid.mettagrid_c as mc

    # Test with agent_radius=0, converter_radius>0 (only affects converters)
    config = mc.ModifyTargetConfig(
        required_resources={},
        consumed_resources={},
        modifies={0: 1.0},
        agent_radius=0,  # No agents affected
        converter_radius=2,  # Converters within radius 2
        scales=False,
    )

    assert config.agent_radius == 0
    assert config.converter_radius == 2

    # Test with agent_radius>0, converter_radius=0 (only affects agents)
    config2 = mc.ModifyTargetConfig(
        required_resources={},
        consumed_resources={},
        modifies={0: 1.0},
        agent_radius=3,  # Agents within radius 3
        converter_radius=0,  # No converters affected
        scales=False,
    )

    assert config2.agent_radius == 3
    assert config2.converter_radius == 0


def test_modify_target_resource_consumption_on_success():
    """Test that resources are only consumed when action succeeds."""
    import numpy as np

    import mettagrid.mettagrid_c as mc

    # Phase 1: Test with radius=0 (no targets, should not consume resources)
    game_config_no_targets = mc.GameConfig(
        num_agents=2,
        obs_width=5,
        obs_height=5,
        max_steps=100,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=False,
        resource_names=["energy", "health"],
        actions=[
            ("move", mc.ActionConfig()),
            (
                "modify_target",
                mc.ModifyTargetConfig(
                    required_resources={0: 10},  # Requires 10 energy
                    consumed_resources={0: 5.0},  # Consumes 5 energy on success
                    modifies={1: 1.0},  # Modifies health
                    agent_radius=0,  # No agents in range
                    converter_radius=0,  # No converters in range
                    scales=False,
                ),
            ),
        ],
        objects={
            "agent.red": mc.AgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                resource_limits={0: 255, 1: 255},
                initial_inventory={0: 20, 1: 50},  # 20 energy, 50 health
            ),
            "agent.blue": mc.AgentConfig(
                type_id=0,
                group_id=1,
                group_name="blue",
                resource_limits={0: 255, 1: 255},
                initial_inventory={0: 15, 1: 40},  # 15 energy, 40 health
            ),
            "wall": mc.WallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=mc.GlobalObsConfig(),
    )

    # Map with agents next to each other
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "agent.blue", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    env_no_targets = mc.MettaGrid(game_config_no_targets, game_map, 0)
    obs, _ = env_no_targets.reset()

    # Get initial inventories
    grid_objects = env_no_targets.grid_objects()
    # Find agents by type (0 = agent)
    agent_ids = [k for k, v in grid_objects.items() if v.get("type") == 0]
    assert len(agent_ids) >= 1, f"No agents found in grid_objects: {grid_objects}"
    red_agent_id = agent_ids[0]  # First agent should be red
    initial_energy_red = grid_objects[red_agent_id]["inventory"].get(0, 0)

    # Try modify_target with radius=0 (no targets)
    modify_target_idx = env_no_targets.action_names().index("modify_target")
    actions = np.array([[modify_target_idx, 0], [0, 0]], dtype=np.int32)  # Red tries modify_target, Blue does nothing
    obs, rewards, dones, truncs, infos = env_no_targets.step(actions)

    # Check that resources were NOT consumed (no targets)
    grid_objects = env_no_targets.grid_objects()
    final_energy_red = grid_objects[red_agent_id]["inventory"].get(0, 0)
    assert final_energy_red == initial_energy_red, (
        f"Energy should not be consumed with no targets: {initial_energy_red} -> {final_energy_red}"
    )
    assert not env_no_targets.action_success()[0], "Action should fail with no targets in range"

    # Phase 2: Test with radius>0 (targets exist, should consume resources)
    game_config_with_targets = mc.GameConfig(
        num_agents=2,
        obs_width=5,
        obs_height=5,
        max_steps=100,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=False,
        resource_names=["energy", "health"],
        actions=[
            ("move", mc.ActionConfig()),
            (
                "modify_target",
                mc.ModifyTargetConfig(
                    required_resources={0: 10},  # Requires 10 energy
                    consumed_resources={0: 5.0},  # Consumes 5 energy on success
                    modifies={1: 1.0},  # Modifies health
                    agent_radius=2,  # Agents within radius 2
                    converter_radius=2,  # Converters within radius 2
                    scales=False,
                ),
            ),
        ],
        objects={
            "agent.red": mc.AgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                resource_limits={0: 255, 1: 255},
                initial_inventory={0: 20, 1: 50},  # 20 energy, 50 health
            ),
            "agent.blue": mc.AgentConfig(
                type_id=0,
                group_id=1,
                group_name="blue",
                resource_limits={0: 255, 1: 255},
                initial_inventory={0: 15, 1: 40},  # 15 energy, 40 health
            ),
            "wall": mc.WallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=mc.GlobalObsConfig(),
    )

    env_with_targets = mc.MettaGrid(game_config_with_targets, game_map, 0)
    obs, _ = env_with_targets.reset()

    # Get initial inventories
    grid_objects = env_with_targets.grid_objects()
    # Find agents by type (0 = agent)
    agent_ids = [k for k, v in grid_objects.items() if v.get("type") == 0]
    assert len(agent_ids) == 2, f"Expected 2 agents, found {len(agent_ids)}"

    # Identify which is which by checking inventory values
    red_agent_id = None
    blue_agent_id = None
    for agent_id in agent_ids:
        energy = grid_objects[agent_id]["inventory"].get(0, 0)
        if energy == 20:
            red_agent_id = agent_id
        elif energy == 15:
            blue_agent_id = agent_id

    assert red_agent_id is not None, "Could not find red agent"
    assert blue_agent_id is not None, "Could not find blue agent"

    initial_energy_red = grid_objects[red_agent_id]["inventory"].get(0, 0)
    initial_health_blue = grid_objects[blue_agent_id]["inventory"].get(1, 0)

    # Try modify_target with radius>0 (blue agent is in range)
    modify_target_idx = env_with_targets.action_names().index("modify_target")
    actions = np.array([[modify_target_idx, 0], [0, 0]], dtype=np.int32)  # Red tries modify_target, Blue does nothing
    obs, rewards, dones, truncs, infos = env_with_targets.step(actions)

    # Check that resources WERE consumed (target exists)
    grid_objects = env_with_targets.grid_objects()
    final_energy_red = grid_objects[red_agent_id]["inventory"].get(0, 0)
    final_health_blue = grid_objects[blue_agent_id]["inventory"].get(1, 0)

    assert final_energy_red == initial_energy_red - 5, (
        f"Energy should be consumed with targets: {initial_energy_red} -> {final_energy_red}"
    )
    assert final_health_blue == initial_health_blue + 1, (
        f"Target health should increase: {initial_health_blue} -> {final_health_blue}"
    )
    assert env_with_targets.action_success()[0], "Action should succeed with targets in range"


# ============================================================================
# Negative Action Argument Tests
# ============================================================================


def test_negative_action_arg_validation():
    """Test that negative action arguments are rejected without crashes."""

    # Create a simple game config with modify_target action
    game_config = CppGameConfig(
        num_agents=1,
        obs_width=5,
        obs_height=5,
        max_steps=100,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=True,
        resource_names=["energy", "health"],
        actions=[
            ("move", CppActionConfig()),
            (
                "modify_target",
                ModifyTargetConfig(
                    required_resources={0: 1},  # Need 1 energy
                    consumed_resources={0: 1.0},  # Consume 1 energy on success
                    modifies={1: 1.0},  # Modify health by 1
                    agent_radius=1,
                    converter_radius=1,
                    scales=False,
                ),
            ),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                resource_limits={0: 10, 1: 10},  # 10 max for energy and health
                initial_inventory={0: 5},  # Start with 5 energy
            ),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    # Create a simple map
    game_map = [
        ["wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", ".", ".", ".", "wall"],
        ["wall", "wall", "wall", "wall", "wall"],
    ]

    # Create the environment
    env = MettaGrid(game_config, game_map, 0)

    # Reset and get initial state
    obs, _ = env.reset()

    # Get initial agent state
    grid_objects = env.grid_objects()
    agent_id = [k for k, v in grid_objects.items() if v.get("type") == 0][0]
    initial_inventory = grid_objects[agent_id]["inventory"].copy()
    # Location is returned as tuple (r, c, layer)
    loc_tuple = grid_objects[agent_id]["location"]
    initial_location = (loc_tuple[0], loc_tuple[1])

    # Try to perform modify_target action with negative arg (-1)
    # Look up the action index dynamically
    action_names = env.action_names()
    modify_target_idx = action_names.index("modify_target") if "modify_target" in action_names else 0
    actions = np.array([[modify_target_idx, -1]], dtype=np.int32)  # modify_target with arg=-1 (invalid)

    # Step with invalid negative arg
    obs, rewards, dones, truncs, infos = env.step(actions)

    # Verify action failed
    assert not env.action_success()[0], "Action with negative arg should fail"

    # Verify state unchanged
    grid_objects = env.grid_objects()
    final_inventory = grid_objects[agent_id]["inventory"]
    # Location is returned as tuple (r, c, layer)
    loc_tuple = grid_objects[agent_id]["location"]
    final_location = (loc_tuple[0], loc_tuple[1])

    assert final_inventory == initial_inventory, (
        f"Inventory should remain unchanged: {initial_inventory} != {final_inventory}"
    )
    assert final_location == initial_location, (
        f"Location should remain unchanged: {initial_location} != {final_location}"
    )
    assert rewards[0] <= 0, f"Should not get positive reward for invalid action: {rewards[0]}"


def test_negative_action_arg_no_crash():
    """Test that negative action arguments don't cause crashes or undefined behavior."""

    game_config = CppGameConfig(
        num_agents=1,
        obs_width=5,
        obs_height=5,
        max_steps=10,
        episode_truncates=False,
        num_observation_tokens=200,
        track_movement_metrics=True,
        resource_names=["resource"],
        actions=[
            ("move", CppActionConfig()),
            (
                "modify_target",
                ModifyTargetConfig(
                    required_resources={},
                    consumed_resources={},
                    modifies={},
                    agent_radius=1,
                    converter_radius=1,
                    scales=False,
                ),
            ),
        ],
        objects={
            "agent.red": CppAgentConfig(type_id=0, group_id=0, group_name="red", resource_limits={0: 255}),
            "wall": CppWallConfig(type_id=1, type_name="wall", swappable=False),
        },
        global_obs=CppGlobalObsConfig(),
    )

    game_map = [
        ["wall", "wall", "wall"],
        ["wall", "agent.red", "wall"],
        ["wall", "wall", "wall"],
    ]

    env = MettaGrid(game_config, game_map, 42)
    obs, _ = env.reset()

    # Get initial agent state
    grid_objects = env.grid_objects()
    agent_id = [k for k, v in grid_objects.items() if v.get("type") == 0][0]
    # Try various negative arguments
    action_names = env.action_names()
    modify_target_idx = action_names.index("modify_target") if "modify_target" in action_names else 0
    test_cases = [
        (np.array([[0, -1]], dtype=np.int32), "move with -1"),
        (np.array([[modify_target_idx, -10]], dtype=np.int32), "modify_target with -10"),
        (np.array([[modify_target_idx, -128]], dtype=np.int32), "modify_target with min int8"),
        (np.array([[0, -255]], dtype=np.int32), "move with large negative"),
    ]

    for actions, description in test_cases:
        # Get state before action
        grid_objects_before = env.grid_objects()
        loc_tuple_before = grid_objects_before[agent_id]["location"]
        location_before = (loc_tuple_before[0], loc_tuple_before[1])

        # Should not crash or raise exception
        obs, rewards, dones, truncs, infos = env.step(actions)

        # Verify action failed due to negative arg
        assert not env.action_success()[0], f"Action should fail for {description}"

        # Verify agent didn't move or change state
        grid_objects_after = env.grid_objects()
        loc_tuple_after = grid_objects_after[agent_id]["location"]
        location_after = (loc_tuple_after[0], loc_tuple_after[1])
        assert location_after == location_before, f"Agent should not move with negative arg ({description})"


# ============================================================================
# Resource Modification Tests
# ============================================================================


def test_modify_target_config():
    """Test that ModifyTargetConfig can be created directly."""
    config = ModifyTargetConfig(
        required_resources={0: 5},
        consumed_resources={0: 1.5},
        modifies={1: 2.0, 2: -0.5},
        agent_radius=3,
        converter_radius=2,
        scales=False,
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
        modifies={1: 1.0},  # Heal self for 1 health (so action succeeds)
        agent_radius=1,
        converter_radius=1,
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
            ("modify_target", modify_target_config),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=0,  # Agent must not be frozen to execute actions
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
    actions[0] = [action_idx, 0]  # No arg needed for radius-based AoE

    env.step(actions)

    # Check that energy was consumed
    objects = env.grid_objects()
    final_energy = objects[agent_id].get("inventory", {}).get(0, 0)

    # With 100% probability, should consume exactly 1 unit
    assert final_energy == initial_energy - 1, f"Energy not consumed: initial={initial_energy}, final={final_energy}"


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
        agent_radius=1,
        converter_radius=1,
        scales=False,
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
        actions=[
            ("noop", CppActionConfig(required_resources={}, consumed_resources={})),
            ("modify_target", modify_config),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=0,  # Agent must not be frozen to execute actions
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
    # Execute modify_target action - AoE affects all agents within radius
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]  # Red agent uses AoE with radius=1
    actions[1] = [0, 0]  # Blue agent does noop

    env.step(actions)

    # Check results
    objects = env.grid_objects()
    red_health_after = objects[red_agent_id].get("inventory", {}).get(1, 0)
    blue_health_after = objects[blue_agent_id].get("inventory", {}).get(1, 0)
    red_energy_after = objects[red_agent_id].get("inventory", {}).get(0, 0)

    # Red should have consumed energy
    assert red_energy_after == red_energy_before - 1

    # With AoE radius=1, both agents gain health (red at (1,1), blue at (1,2) are within Manhattan distance 1)
    assert blue_health_after == blue_health_before + 5
    assert red_health_after == red_health_before + 5  # Red also gains health from its own AoE


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
        agent_radius=1,
        converter_radius=1,
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
            ("modify_target", modify_config),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=0,  # Agent must not be frozen to execute actions
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

    # Agent uses AoE to target converter
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]  # AoE with radius=1 affects adjacent converter

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
        agent_radius=1,
        converter_radius=1,
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
            ("modify_target", modify_config),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=0,  # Agent must not be frozen to execute actions
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

    # Use AoE - agent will affect itself since it's within its own radius
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((1, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]  # AoE includes self

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
        agent_radius=1,
        converter_radius=1,
        scales=False,
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
        actions=[
            ("noop", CppActionConfig(required_resources={}, consumed_resources={})),
            ("modify_target", modify_config),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=0,
                group_name="red",
                freeze_duration=0,  # Agent must not be frozen to execute actions
                action_failure_penalty=0,
                resource_limits={0: 100, 1: 100},
                initial_inventory={0: 50, 1: 50},
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                type_name="agent",
                group_id=1,
                group_name="blue",
                freeze_duration=0,  # Agent must not be frozen to execute actions
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

    # Red uses AoE damage action
    action_idx = env.action_names().index("modify_target")
    actions = np.zeros((2, 2), dtype=np.int32)
    actions[0] = [action_idx, 0]  # AoE with radius=1 affects adjacent agents
    actions[1] = [0, 0]

    env.step(actions)

    objects = env.grid_objects()
    blue_health_after = objects[blue_agent_id].get("inventory", {}).get(1, 0)

    # Blue should have lost health (AoE affects both agents)
    assert blue_health_after == blue_health_before - 3


def test_modify_target_invalid_magnitude():
    """Test that ModifyTargetConfig throws on invalid magnitudes."""
    # Test NaN value
    with pytest.raises(ValueError):
        ModifyTargetConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: float("nan")},
            agent_radius=1,
            converter_radius=1,
            scales=False,
        )

    # Test value too large (>255)
    with pytest.raises(ValueError):
        ModifyTargetConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: 1000.0},
            agent_radius=1,
            converter_radius=1,
            scales=False,
        )

    # Test negative value too large (abs > 255)
    with pytest.raises(ValueError):
        ModifyTargetConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: -500.0},
            agent_radius=1,
            converter_radius=1,
            scales=False,
        )

    # Test infinity
    with pytest.raises(ValueError):
        ModifyTargetConfig(
            required_resources={},
            consumed_resources={},
            modifies={0: float("inf")},
            agent_radius=1,
            converter_radius=1,
            scales=False,
        )

    # Test that valid values work
    config = ModifyTargetConfig(
        required_resources={},
        consumed_resources={},
        modifies={0: 255.0, 1: -255.0, 2: 0.5},
        agent_radius=1,
        converter_radius=1,
        scales=False,
    )
    assert config.modifies[0] == 255.0
    assert config.modifies[1] == -255.0
    assert config.modifies[2] == 0.5
