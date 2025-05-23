"""
Heart Collection Test
"""

import numpy as np

from mettagrid.mettagrid_c import MettaGrid
from mettagrid.tests.actions import _get_agent_orientation, _get_agent_position, move, rotate


def test_heart_collection():
    """Test heart collection."""

    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "agent.red", "empty", "altar", "empty", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    config = {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "actions": {
            "noop": {"enabled": True},
            "get_items": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "attack": {"enabled": True},
            "swap": {"enabled": True},
            "change_color": {"enabled": True},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1, "hp": 100},
            "altar": {
                "type_id": 4,
                "hp": 100,
                "output_heart": 1,
                "initial_items": 0,
                "max_output": 50,
                "conversion_ticks": 0,
                "cooldown": -1,
            },
        },
        "agent": {
            "inventory_size": 10,
            "hp": 100,
            "rewards": {"heart": 1.0},
        },
    }

    try:
        env_config = {"game": config}
        env = MettaGrid(env_config, game_map)

        # Set up buffers
        np_actions_type = np.dtype(env.get_numpy_type_name("actions"))
        np_observations_type = np.dtype(env.get_numpy_type_name("observations"))
        np_terminals_type = np.dtype(env.get_numpy_type_name("terminals"))
        np_truncations_type = np.dtype(env.get_numpy_type_name("truncations"))
        np_rewards_type = np.dtype(env.get_numpy_type_name("rewards"))

        num_features = len(env.grid_features())
        observations = np.zeros((1, 3, 3, num_features), dtype=np_observations_type)
        terminals = np.zeros(1, dtype=np_terminals_type)
        truncations = np.zeros(1, dtype=np_truncations_type)
        rewards = np.zeros(1, dtype=np_rewards_type)
        env.set_buffers(observations, terminals, truncations, rewards)

        obs, info = env.reset()

        print("🔧 HEART COLLECTION - RELIABLE MOVEMENT")
        print("=" * 60)
        print(f"Available actions: {env.action_names()}")
        print(f"Max action args: {env.max_action_args()}")
        print()

        def get_agent_hearts(obs):
            grid_features = env.grid_features()
            if "agent:inv:heart" in grid_features:
                feature_idx = grid_features.index("agent:inv:heart")
                return int(np.sum(obs[:, :, feature_idx]))
            return 0

        def perform_action(action_name, arg=0):
            action_idx = env.action_names().index(action_name)
            action = np.zeros((1, 2), dtype=np_actions_type)
            action[0] = [action_idx, arg]
            obs, rewards, _terminals, _truncations, _info = env.step(action)
            return obs, float(rewards[0]), env.action_success()[0]

        def show_state():
            """Show current state using our utility functions."""
            agent_pos = _get_agent_position(env, 0)
            agent_orientation = _get_agent_orientation(env, 0)

            # Get altar info from grid objects
            grid_objects = env.grid_objects()
            altar_hearts = 0
            altar_pos = None
            for _obj_id, obj_data in grid_objects.items():
                if "altar" in obj_data:
                    altar_pos = (obj_data["r"], obj_data["c"])
                    altar_hearts = obj_data.get("inv:heart", 0)
                    break

            print(f"  Agent at {agent_pos} facing {agent_orientation}")
            print(f"  Altar at {altar_pos} with {altar_hearts} hearts")

            return agent_pos, altar_pos, agent_orientation, altar_hearts

        # Show initial state
        print("INITIAL STATE:")
        agent_pos, altar_pos, agent_orientation, altar_hearts = show_state()
        print()

        # Wait for heart production
        print("Step 1: Wait for heart production...")
        for _i in range(3):
            obs, reward, success = perform_action("noop")
        print()

        # Show state after waiting
        print("AFTER WAITING:")
        agent_pos, altar_pos, _agent_orientation, altar_hearts = show_state()
        print()

        # Use reliable movement to get adjacent to altar
        print("Step 2: Move right to get adjacent to altar...")
        move_result = move(env, 3)  # Move right (orientation 3)

        if not move_result["success"]:
            print(f"❌ Movement failed: {move_result['error']}")
            return False

        print("✅ Movement successful!")
        print()

        # Show final positions
        print("FINAL POSITIONS:")
        agent_pos, altar_pos, _agent_orientation, altar_hearts = show_state()

        # Verify adjacency
        if agent_pos and altar_pos:
            distance = abs(agent_pos[0] - altar_pos[0]) + abs(agent_pos[1] - altar_pos[1])
            print(f"  Distance to altar: {distance}")

            if distance != 1:
                print(f"❌ Agent not adjacent! Expected distance 1, got {distance}")
                # Try moving one more time
                print("\nTrying one more move...")
                move_result = move(env, 3)  # Move right again
                if move_result["success"]:
                    agent_pos, altar_pos, _agent_orientation, altar_hearts = show_state()
                    distance = abs(agent_pos[0] - altar_pos[0]) + abs(agent_pos[1] - altar_pos[1])
                    print(f"  New distance: {distance}")
        print()

        # Try heart collection with perfect positioning
        print("Step 3: Try get_output with current positioning...")
        hearts_before = get_agent_hearts(obs[0])
        obs, reward, success = perform_action("get_output", 0)
        hearts_after = get_agent_hearts(obs[0])
        hearts_gained = hearts_after - hearts_before

        print(f"  Hearts before: {hearts_before}")
        print(f"  Hearts after: {hearts_after}")
        print(f"  Hearts gained: {hearts_gained}")
        print(f"  Action success: {success}")
        print(f"  Reward: {reward}")

        if success and hearts_gained > 0:
            print("\n🎉 SUCCESS! Heart collection works with reliable movement!")
            return True
        elif success:
            print("\n⚠️ Action succeeded but no hearts gained")
        else:
            print("\n❌ get_output failed - trying all orientations...")

        # Try all orientations while maintaining position
        print("\nStep 4: Testing all orientations while adjacent...")
        orientation_names = {0: "up", 1: "down", 2: "left", 3: "right"}

        for orientation in range(4):
            direction_name = orientation_names[orientation]
            print(f"\nTrying orientation {orientation} ({direction_name}):")

            # Use reliable rotate function
            rotate_result = rotate(env, orientation)

            if not rotate_result["success"]:
                print(f"  ❌ Rotation failed: {rotate_result['error']}")
                continue

            # Verify we're still in the same position
            current_pos = _get_agent_position(env, 0)
            if current_pos != agent_pos:
                print(f"  ⚠️ Position changed during rotation: {agent_pos} → {current_pos}")

            # Try get_output with this orientation
            hearts_before = get_agent_hearts(obs[0])
            obs, reward, success = perform_action("get_output", 0)
            hearts_after = get_agent_hearts(obs[0])
            hearts_gained = hearts_after - hearts_before

            print(f"  Result: success={success}, hearts_gained={hearts_gained}, reward={reward}")

            if success and hearts_gained > 0:
                print(f"  🎉 SUCCESS! get_output works when facing {direction_name}!")
                return True

        print("\n❌ get_output failed in all orientations")
        print("   This confirms the issue is in the get_output action logic itself")
        print("   Next step: Add C++ debugging to see exact failure reason")

        # Final debug info
        print("\nFINAL DEBUG INFO:")
        final_pos = _get_agent_position(env, 0)
        final_orientation = _get_agent_orientation(env, 0)
        print(f"  Agent final position: {final_pos}")
        print(f"  Agent final orientation: {final_orientation}")
        print(f"  Altar position: {altar_pos}")
        print(f"  Altar hearts available: {altar_hearts}")
        print(f"  Agent adjacent: {distance == 1}")

        return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_heart_collection()
