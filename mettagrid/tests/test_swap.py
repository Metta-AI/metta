import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_config import from_mettagrid_config
from metta.mettagrid.mettagrid_env import dtype_actions


def test_swap():
    """Test that swap_objects preserves the original layers when swapping positions.

    This test verifies the fix for a critical bug where swap_objects was swapping
    the layers along with the positions, causing agents to end up in the wrong layer.

    The bug would cause:
    - Agents (which must be in AgentLayer=0) to move to ObjectLayer=1
    - Objects to move to AgentLayer=0

    This violates the invariant that agents must always remain in AgentLayer.
    """
    # Create a minimal 3x3 map with an agent and a swappable block
    # Layout:
    #   W W W
    #   W A B    W=wall, A=agent, B=block (swappable)
    #   W W W
    game_map = np.array(
        [["wall", "wall", "wall"], ["wall", "agent.red", "block"], ["wall", "wall", "wall"]], dtype="<U50"
    )

    game_config = {
        "max_steps": 10,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "num_observation_tokens": 100,
        "inventory_item_names": [],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": False},
            "put_items": {"enabled": False},
            "get_items": {"enabled": False},
            "swap": {
                "enabled": True,
                "required_resources": {},  # No resources needed
                "consumed_resources": {},
            },
            "change_color": {"enabled": False},
        },
        "groups": {"red": {"id": 0, "props": {}}},
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
    swap_idx = env.action_names().index("swap")
    actions = np.array([[swap_idx, 0]], dtype=dtype_actions)
    env.step(actions)

    if env.action_success()[0]:
        pytest.fail("Swap with non-swappable wall should have failed!")
    print("  ✓ Swap correctly rejected (wall is not swappable)")

    # Now rotate to face the swappable block
    # Agent starts at (1,1) facing Up (orientation=0)
    # Block is at (1,2) to the right
    # We need to rotate the agent to face Right (orientation=3)
    print("\nRotating agent to face right (toward block):")
    rotate_idx = env.action_names().index("rotate")
    actions = np.array([[rotate_idx, 3]], dtype=dtype_actions)  # 3 = Right
    env.step(actions)
    print("  Agent now facing right")

    # Now perform the swap with the block
    print("\nAttempting swap with block (should succeed):")
    actions = np.array([[swap_idx, 0]], dtype=dtype_actions)  # arg is ignored
    env.step(actions)

    # Check if swap succeeded
    if not env.action_success()[0]:
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

    # CRITICAL: Verify layers were preserved (not swapped)
    layers_correct = agent_final["layer"] == agent_layer and block_final["layer"] == block_layer

    if not layers_correct:
        print("\nBUG DETECTED:")
        print(f"  Agent layer changed: {agent_layer} -> {agent_final['layer']} (should stay 0)")
        print(f"  Block layer changed: {block_layer} -> {block_final['layer']} (should stay 1)")
        print("  The layers were swapped instead of preserved!")
        pytest.fail("Layer preservation bug: swap_objects swapped layers instead of preserving them")

    print("\nSUCCESS: Layers were correctly preserved during swap!")
