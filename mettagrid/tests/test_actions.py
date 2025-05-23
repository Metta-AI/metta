import numpy as np

from mettagrid.mettagrid_c import MettaGrid
from mettagrid.tests.actions import _get_agent_position, move

# Rebuild the NumPy types using the exposed function
np_actions_type = np.dtype(MettaGrid.get_numpy_type_name("actions"))
np_observations_type = np.dtype(MettaGrid.get_numpy_type_name("observations"))
np_terminals_type = np.dtype(MettaGrid.get_numpy_type_name("terminals"))
np_truncations_type = np.dtype(MettaGrid.get_numpy_type_name("truncations"))
np_rewards_type = np.dtype(MettaGrid.get_numpy_type_name("rewards"))


def test_move():
    """Test the move function."""

    game_map = [
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "empty", "empty", "agent.red", "empty", "empty", "wall"],  # Agent in center
        ["wall", "empty", "empty", "empty", "empty", "empty", "wall"],
        ["wall", "wall", "wall", "wall", "wall", "wall", "wall"],
    ]

    config = {
        "max_steps": 50,
        "num_agents": 1,
        "obs_width": 3,
        "obs_height": 3,
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "get_items": {"enabled": True},  # maps to get_output
            "attack": {"enabled": False},
            "put_items": {"enabled": False},  # maps to get_recipe_items
            "swap": {"enabled": False},
            "change_color": {"enabled": False},
        },
        "groups": {"red": {"id": 0, "props": {}}},
        "objects": {
            "wall": {"type_id": 1, "hp": 100},
        },
        "agent": {
            "inventory_size": 10,
            "hp": 100,
        },
    }

    try:
        env_config = {"game": config}
        env = MettaGrid(env_config, game_map)

        # Set up buffers
        num_features = len(env.grid_features())
        observations = np.zeros((1, 3, 3, num_features), dtype=np_observations_type)
        terminals = np.zeros(1, dtype=np_terminals_type)
        truncations = np.zeros(1, dtype=np_truncations_type)
        rewards = np.zeros(1, dtype=np_rewards_type)
        env.set_buffers(observations, terminals, truncations, rewards)

        _obs, _info = env.reset()

        print("🔧 TESTING SIMPLIFIED MOVE FUNCTION")
        print("=" * 50)

        initial_pos = _get_agent_position(env)
        print(f"Agent initial position: {initial_pos}")

        # Test all 4 directions
        directions = [(0, "up"), (3, "right"), (1, "down"), (2, "left")]

        for orientation, direction_name in directions:
            print(f"\n--- Testing move {direction_name} (orientation {orientation}) ---")
            result = move(env, orientation)

            status = "✅" if result["success"] else "❌"
            print(f"{status} Move {direction_name}: {result['success']}")

            if result["error"]:
                print(f"    Error: {result['error']}")

            pos_change = f"{result['position_before']} → {result['position_after']}"
            print(f"    Position: {pos_change}")
            print(f"    Moved correctly: {result['moved_correctly']}")
            print(f"    Observations changed: {result['obs_changed']}")

        print("\n✅ Simplified move tests complete!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_move()
