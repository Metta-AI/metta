"""Debug version of the swap test to understand why it's failing."""

import numpy as np

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.map_builder.map_builder import map_grid_dtype
from mettagrid.mettagrid_c import MettaGrid, dtype_actions


def debug_swap_frozen_agent():
    """Debug version to understand why swap with frozen agent is failing."""
    # Create a minimal map with two adjacent agents
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
            "attack": {"enabled": True},
            "swap": {"enabled": True},
        },
        "agents": [
            {"team_id": 0, "freeze_duration": 6, "resource_limits": {"laser": 10}, "initial_inventory": {"laser": 5}},
            {"team_id": 1, "freeze_duration": 6, "resource_limits": {"laser": 10}, "initial_inventory": {"laser": 5}},
        ],
        "objects": {
            "wall": {"type_id": 1, "swappable": False},
        },
        "agent": {
            "freeze_duration": 6,
            "resource_limits": {"laser": 10},
            "initial_inventory": {"laser": 5},  # Start with lasers
        },
    }

    env = MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)
    env.reset()

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

    agents.sort(key=lambda a: a["pos"])
    agent0 = agents[0]  # Left agent
    agent1 = agents[1]  # Right agent

    print("Initial positions:")
    print(f"  Agent {agent0['id']}: {agent0['pos']}")
    print(f"  Agent {agent1['id']}: {agent1['pos']}")

    # Get action indices
    attack_idx = env.action_names().index("attack")
    swap_idx = env.action_names().index("swap")
    noop_idx = env.action_names().index("noop")
    move_idx = env.action_names().index("move")
    rotate_idx = env.action_names().index("rotate")

    # First, let's verify the agents are where we think they are
    print("\nAgent positions from initial map:")
    print("  Red agent should be at (1,1)")
    print("  Blue agent should be at (2,3)")

    # Let's check if we need to move differently to reach the frozen agent
    print("\nTrying simpler approach - agents start closer:")

    # Create a simpler map with agents adjacent
    simple_map = np.array(
        [
            ["wall", "wall", "wall", "wall"],
            ["wall", "agent.red", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall"],
        ],
        dtype=map_grid_dtype,
    )

    env2 = MettaGrid(from_mettagrid_config(game_config), simple_map.tolist(), 42)
    env2.reset()

    # Find agents in new environment
    objects2 = env2.grid_objects()
    agents2 = []
    for oid, obj in objects2.items():
        if obj.get("type") == 0:
            agents2.append(
                {
                    "id": oid,
                    "pos": (obj["r"], obj["c"]),
                    "orientation": obj.get("orientation", -1),
                }
            )

    agents2.sort(key=lambda a: a["pos"])
    ag0 = agents2[0]
    ag1 = agents2[1]

    print("\nSimpler setup - agents are adjacent:")
    print(f"  Agent {ag0['id']}: pos={ag0['pos']}, orientation={ag0['orientation']}")
    print(f"  Agent {ag1['id']}: pos={ag1['pos']}, orientation={ag1['orientation']}")

    # Get rotate action index
    rotate_idx = env2.action_names().index("rotate")

    # Make agent 0 face right to attack agent 1
    # Orientation mapping: 0=Up, 1=Down, 2=Left, 3=Right
    actions = np.zeros((2, 2), dtype=dtype_actions)
    actions[0] = [rotate_idx, 3]  # Rotate to face Right
    actions[1] = [noop_idx, 0]
    env2.step(actions)

    # Check orientation after turning
    objects2 = env2.grid_objects()
    ag0_orient = objects2[ag0["id"]].get("orientation", -1)
    print(f"  Agent 0 orientation after rotate Right: {ag0_orient}")

    # Check resources before attack
    ag0_laser = objects2[ag0["id"]].get("resources", {}).get("laser", 0)
    print(f"  Agent 0 laser count: {ag0_laser}")

    # Attack to freeze
    print("\nAttacking to freeze agent 1:")
    actions[0] = [attack_idx, 0]  # Attack in front
    actions[1] = [noop_idx, 0]
    env2.step(actions)

    if not env2.action_success()[0]:
        print("  Attack failed!")

        # Debug why attack failed
        print("\nDebugging attack failure:")

        # Check if agents have lasers
        ag0_laser = objects2[ag0["id"]].get("resources", {}).get("laser", 0)
        print(f"  Agent 0 laser count: {ag0_laser}")

        # Try giving agent a laser first
        print("\nTrying a different approach - checking if attack needs resources...")

        # Let's check the attack configuration
        print("  Attack might require resources or have other conditions")

        # Let's try a different attack argument
        print("\nTrying attack with different arguments:")
        for arg in range(5):
            actions[0] = [attack_idx, arg]
            actions[1] = [noop_idx, 0]
            env2.step(actions)
            if env2.action_success()[0]:
                print(f"  Attack succeeded with arg={arg}")
                break
        else:
            print("  Attack failed with all arguments 0-4")
            return

    # Check if frozen
    objects2 = env2.grid_objects()
    frozen = objects2[ag1["id"]].get("freeze_remaining", 0)
    print(f"  Agent 1 frozen for {frozen} steps")

    # Try to swap
    print("\nAttempting swap with frozen agent:")
    actions[0] = [swap_idx, 0]
    actions[1] = [noop_idx, 0]
    env2.step(actions)

    success = env2.action_success()[0]
    print(f"  Swap success: {success}")

    if not success:
        print("\nPossible reasons for failure:")
        print("  1. Frozen agents might not actually be swappable")
        print("  2. The swap action might require different arguments")
        print("  3. There might be other conditions preventing the swap")

        # Let's check what happens with a non-frozen swappable object
        print("\nTesting swap with a swappable block instead:")
        block_map = np.array(
            [
                ["wall", "wall", "wall", "wall"],
                ["wall", "agent.red", "block", "wall"],
                ["wall", "wall", "wall", "wall"],
            ],
            dtype=map_grid_dtype,
        )

        config_with_block = game_config.copy()
        config_with_block["objects"]["block"] = {"type_id": 14, "swappable": True}

        env3 = MettaGrid(from_mettagrid_config(config_with_block), block_map.tolist(), 42)
        env3.reset()

        # Face the block
        actions = np.array([[move_idx, 2]], dtype=dtype_actions)
        env3.step(actions)

        # Try to swap
        actions = np.array([[swap_idx, 0]], dtype=dtype_actions)
        env3.step(actions)

        block_swap_success = env3.action_success()[0]
        print(f"  Swap with block success: {block_swap_success}")


if __name__ == "__main__":
    debug_swap_frozen_agent()
