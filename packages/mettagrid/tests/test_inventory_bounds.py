"""Test that inventory properly handles overflow and underflow."""

import numpy as np

from mettagrid.mettagrid_c import (
    ActionConfig as CppActionConfig,
)
from mettagrid.mettagrid_c import (
    AgentConfig as CppAgentConfig,
)
from mettagrid.mettagrid_c import (
    GameConfig as CppGameConfig,
)
from mettagrid.mettagrid_c import (
    GlobalObsConfig as CppGlobalObsConfig,
)
from mettagrid.mettagrid_c import (
    InventoryConfig as CppInventoryConfig,
)
from mettagrid.mettagrid_c import (
    MettaGrid,
    ResourceModConfig,
)
from mettagrid.mettagrid_c import (
    WallConfig as CppWallConfig,
)


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
                "resource_mod",
                ResourceModConfig(
                    required_resources={},
                    consumed_resources={},
                    modifies={0: 1.0},  # Add 1 resource
                    agent_radius=1,  # Affects self (distance 0)
                    converter_radius=0,
                    scales=False,
                ),
            ),
        ],
        objects={
            "agent.red": CppAgentConfig(
                type_id=0,
                group_id=0,
                group_name="red",
                inventory_config=CppInventoryConfig(limits=[[[0], 255]]),
                initial_inventory={0: 254},  # One below max
            ),
            "agent.blue": CppAgentConfig(
                type_id=0,
                group_id=1,
                group_name="blue",
                inventory_config=CppInventoryConfig(limits=[[[0], 255]]),
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

    # Both agents try to add 1 resource to themselves
    # With ResourceMod and agent_radius=1, the action affects the acting agent (at distance 0)
    # The agents are spaced apart so they don't affect each other
    actions = np.array([[1, 0], [1, 0]], dtype=np.int32)  # resource_mod with arg=0 (unused)
    obs, rewards, dones, truncs, infos = env.step(actions)

    # Verify action success
    action_success = env.action_success()

    # Get both agents' inventories from grid_objects()
    # grid_objects() returns a dictionary of object ID -> object
    objects = env.grid_objects()
    agent_inventories = []

    # Find agents (type 0) and get their inventories
    agent_objects = []
    for oid, obj in objects.items():
        if obj.get("type") == 0:  # Agent type
            agent_objects.append((oid, obj))

    # Sort by group_id to ensure consistent order (red=0, blue=1)
    agent_objects.sort(key=lambda x: x[1].get("group", 0))

    for _oid, obj in agent_objects[:2]:  # Get first two agents
        agent_inventory = obj.get("inventory", {}).get(0, 0)  # resource is at index 0
        agent_inventories.append(agent_inventory)

    # Note: resource_mod with radius=1 affects the agent itself (manhattan distance 0).
    # Agent.red at 254 + 1 = 255 (at max)
    # Agent.blue at 255 + 1 = 255 (already at max, stays at max)
    # The test verifies that inventories are capped at 255.
    assert agent_inventories[0] == 255, (
        f"Agent.red inventory should be 255 after adding 1 to 254, got {agent_inventories[0]}"
    )
    assert agent_inventories[1] == 255, f"Agent.blue inventory should remain at 255 (max), got {agent_inventories[1]}"
    assert action_success[0], "Agent.red action should succeed"
    assert action_success[1], "Agent.blue action should succeed (even though already at max)"
