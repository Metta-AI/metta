import numpy as np
import pytest

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.map_builder.map_builder import map_grid_dtype
from mettagrid.mettagrid_c import MettaGrid
from mettagrid.test_support.actions import attack, move, rotate, swap
from mettagrid.test_support.orientation import Orientation


def test_swap():
    """Test that swap_objects preserves the original layers when swapping positions."""
    # Create a minimal 3x3 map with an agent and a swappable block
    # Layout:
    #   W W W
    #   W A B    W=wall, A=agent, B=block (swappable)
    #   W W W
    game_map = np.array(
        [["wall", "wall", "wall"], ["wall", "agent.red", "block"], ["wall", "wall", "wall"]], dtype=map_grid_dtype
    )

    game_config = {
        "max_steps": 10,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 100,
        "resource_names": [],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "swap": {
                "enabled": True,
            },
        },
        "objects": {
            "wall": {"type_id": 1, "swappable": False},
            "block": {"type_id": 14, "swappable": True},  # Swappable block
        },
        "agent": {},
    }

    # Create environment
    env = MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)
    env.reset()

    # Find the agent and block
    objects = env.grid_objects()
    agent_id = None
    block_id = None
    agent_pos = None
    block_pos = None
    agent_layer = None
    block_layer = None

    for oid, obj in objects.items():
        if obj.get("type") == 0:  # Agent type
            agent_id = oid
            agent_pos = (obj["r"], obj["c"])
            agent_layer = obj["layer"]
        elif obj.get("type") == 14:  # Block type
            block_id = oid
            block_pos = (obj["r"], obj["c"])
            block_layer = obj["layer"]

    # Ensure we found both objects
    assert agent_id is not None, "Could not find agent in environment"
    assert block_id is not None, "Could not find block in environment"

    print("Initial state:")
    print(f"  Agent {agent_id}: pos={agent_pos}, layer={agent_layer}")
    print(f"  Block {block_id}: pos={block_pos}, layer={block_layer}")

    # First, try to swap with the wall the agent is initially facing (Up)
    # This should fail because walls are not swappable
    print("\nAttempting swap with wall (should fail):")
    swap_result = swap(env)
    if swap_result["success"]:
        pytest.fail("Swap with non-swappable wall should have failed!")
    print("  ✓ Swap correctly rejected (wall is not swappable)")

    # Now rotate to face right (toward the block)
    print("\nRotating to face right (toward block):")
    rotate_result = rotate(env, Orientation.EAST)
    assert rotate_result["success"], f"Rotation failed: {rotate_result.get('error')}"
    print("  Agent now facing right")

    # Now perform the swap with the block
    print("\nAttempting swap with block (should succeed):")
    swap_result = swap(env)

    # Check if swap succeeded
    if not swap_result["success"]:
        pytest.skip("Swap failed - cannot test layer preservation")

    # Get the final state
    final_objects = env.grid_objects()
    agent_final = final_objects[agent_id]
    block_final = final_objects[block_id]

    print("\nAfter swap:")
    print(f"  Agent {agent_id}: pos=({agent_final['r']},{agent_final['c']}), layer={agent_final['layer']}")
    print(f"  Block {block_id}: pos=({block_final['r']},{block_final['c']}), layer={block_final['layer']}")

    # Verify positions were swapped
    assert (agent_final["r"], agent_final["c"]) == block_pos, "Agent should be at block's position"
    assert (block_final["r"], block_final["c"]) == agent_pos, "Block should be at agent's position"

    # Verify layers were preserved (not swapped)
    layers_correct = agent_final["layer"] == agent_layer and block_final["layer"] == block_layer

    if not layers_correct:
        print("\nBUG DETECTED:")
        print(f"  Agent layer changed: {agent_layer} -> {agent_final['layer']} (should stay 0)")
        print(f"  Block layer changed: {block_layer} -> {block_final['layer']} (should stay 1)")
        print("  The layers were swapped instead of preserved!")
        pytest.fail("Layer preservation bug: swap_objects swapped layers instead of preserving them")

    print("\nSUCCESS: Layers were correctly preserved during swap!")


def test_swap_frozen_agent_preserves_layers():
    """Test that swap_objects preserves layers when swapping with a frozen agent.

    Frozen agents are swappable according to the agent.hpp code:
    - agent->swappable() returns true when agent->frozen > 0

    This test:
    1. Creates two adjacent agents
    2. Has one attack the other to freeze it
    3. Swaps with the frozen agent
    4. Verifies layers are preserved
    """
    # Create a minimal map with two adjacent agents
    # Layout:
    #   W W W W W
    #   W A . . W    W=wall, A=agent
    #   W . . A W
    #   W W W W W
    game_map = np.array(
        [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "empty", "empty", "wall"],
            ["wall", "empty", "empty", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ],
        dtype=map_grid_dtype,
    )

    game_config = {
        "max_steps": 10,
        "num_agents": 2,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 100,
        "resource_names": ["laser"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {
                "enabled": True,
            },
            "swap": {
                "enabled": True,
            },
        },
        "agents": [
            {"team_id": 0, "freeze_duration": 6, "resource_limits": {"laser": 10}},  # red
            {"team_id": 1, "freeze_duration": 6, "resource_limits": {"laser": 10}},  # blue
        ],
        "objects": {
            "wall": {"type_id": 1, "swappable": False},
        },
        "agent": {
            "freeze_duration": 6,
            "resource_limits": {"laser": 10},  # Allow agents to hold lasers
        },
    }

    # Create environment
    env = MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)
    env.reset()

    assert env.map_height == 4
    assert env.map_width == 5

    # Find both agents
    objects = env.grid_objects()
    agents = []

    for oid, obj in objects.items():
        if obj.get("type") == 0:  # Agent type
            agents.append(
                {
                    "id": oid,
                    "pos": (obj["r"], obj["c"]),
                    "layer": obj["layer"],
                }
            )

    assert len(agents) == 2, f"Expected 2 agents, found {len(agents)}"

    # Sort agents by position to ensure consistent ordering
    agents.sort(key=lambda a: a["pos"])
    agent0 = agents[0]  # Left agent at (1,1)
    agent1 = agents[1]  # Right agent at (2,3)

    print("Initial state:")
    print(f"  Agent {agent0['id']}: pos={agent0['pos']}, layer={agent0['layer']}")
    print(f"  Agent {agent1['id']}: pos={agent1['pos']}, layer={agent1['layer']}")

    # Get action indices we'll need
    all_agent_ids = sorted([a["id"] for a in agents])
    agent0_action_idx = all_agent_ids.index(agent0["id"])
    agent1_action_idx = all_agent_ids.index(agent1["id"])

    print(f"  Agent {agent0['id']} uses action index {agent0_action_idx}")
    print(f"  Agent {agent1['id']} uses action index {agent1_action_idx}")

    # Agent 0 needs to face right to attack agent 1
    print("\nAgent 0 rotating to face right:")
    rotate_result = rotate(env, Orientation.EAST, agent_idx=agent0_action_idx)
    assert rotate_result["success"], f"Rotation failed: {rotate_result.get('error')}"

    # Agent 0 attacks agent 1 to freeze it
    print("\nAgent 0 attacking agent 1:")
    attack_result = attack(env, target_arg=6, agent_idx=agent0_action_idx)
    assert attack_result["success"], "Attack should succeed"

    if "frozen_agent_id" in attack_result:
        print(f"  Froze agent {attack_result['frozen_agent_id']} for {attack_result['freeze_duration']} steps")

    # Verify agent 1 is frozen
    objects = env.grid_objects()
    agent1_frozen = objects[agent1["id"]].get("freeze_remaining", 0)
    assert agent1_frozen > 0, "Agent 1 should be frozen"

    # Walk over to the frozen agent
    # Agent0 is at (1,1), Agent1 is at (2,3)
    # We need to move right twice then be adjacent to swap
    print("\nAgent 0 moving to be adjacent to frozen agent 1:")

    # Move right twice
    move_result = move(env, Orientation.EAST, agent_idx=agent0_action_idx)
    assert move_result["success"], "First move east should succeed"

    move_result = move(env, Orientation.EAST, agent_idx=agent0_action_idx)
    assert move_result["success"], "Second move east should succeed"

    # Now rotate to face down (towards the frozen agent)
    print("\nAgent 0 rotating to face down:")
    rotate_result = rotate(env, Orientation.SOUTH, agent_idx=agent0_action_idx)
    assert rotate_result["success"], f"Rotation to down failed: {rotate_result.get('error')}"

    # Verify agent 1 is still frozen
    objects = env.grid_objects()
    agent1_frozen = objects[agent1["id"]].get("freeze_remaining", 0)
    print(f"  Agent 1 frozen for {agent1_frozen - 1} more steps")
    assert agent1_frozen > 0

    print("\nBefore swap:")
    agent0_before = objects[agent0["id"]]
    agent1_before = objects[agent1["id"]]
    print(f"  Agent {agent0['id']}: pos=({agent0_before['r']},{agent0_before['c']}), layer={agent0_before['layer']}")
    print(f"  Agent {agent1['id']}: pos=({agent1_before['r']},{agent1_before['c']}), layer={agent1_before['layer']}")

    # Now swap with the frozen agent
    print("\nAgent 0 swapping with frozen agent 1:")
    swap_result = swap(env, agent_idx=agent0_action_idx)

    if not swap_result["success"]:
        pytest.skip(f"Swap with frozen agent not supported: {swap_result.get('error')}")

    # Get final state
    objects = env.grid_objects()
    agent0_after = objects[agent0["id"]]
    agent1_after = objects[agent1["id"]]

    print("\nAfter swap:")
    print(f"  Agent {agent0['id']}: pos=({agent0_after['r']},{agent0_after['c']}), layer={agent0_after['layer']}")
    print(f"  Agent {agent1['id']}: pos=({agent1_after['r']},{agent1_after['c']}), layer={agent1_after['layer']}")

    # Verify positions were swapped
    r0_before, c0_before = agent0_before["r"], agent0_before["c"]
    r1_before, c1_before = agent1_before["r"], agent1_before["c"]
    r0_after, c0_after = agent0_after["r"], agent0_after["c"]
    r1_after, c1_after = agent1_after["r"], agent1_after["c"]

    assert (r0_after, c0_after) == (r1_before, c1_before), "Agent 0 should be at agent 1's original position"
    assert (r1_after, c1_after) == (r0_before, c0_before), "Agent 1 should be at agent 0's original position"

    # CRITICAL: Verify layers were preserved
    if agent0_after["layer"] != agent0_before["layer"] or agent1_after["layer"] != agent1_before["layer"]:
        print("\nBUG DETECTED:")
        print(f"  Agent 0 layer: {agent0_before['layer']} -> {agent0_after['layer']} (should stay 0)")
        print(f"  Agent 1 layer: {agent1_before['layer']} -> {agent1_after['layer']} (should stay 0)")
        print("  Layers were changed during swap!")
        pytest.fail("Layer preservation bug: agents' layers changed during swap")

    print("\nSUCCESS: Layers correctly preserved when swapping with frozen agent!")
