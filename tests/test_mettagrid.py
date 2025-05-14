import numpy as np

from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611


def create_minimal_mettagrid_env(max_steps=10, width=5, height=5):
    """Helper function to create a MettaGrid environment with minimal config."""
    # Define a simple map: empty with walls around perimeter
    game_map = np.full((height, width), "empty", dtype="<U50")
    game_map[0, :] = "wall"
    game_map[-1, :] = "wall"
    game_map[:, 0] = "wall"
    game_map[:, -1] = "wall"

    # Place first agent in upper left
    game_map[1, 1] = "agent.red"

    # Place second agent in middle
    mid_y = height // 2
    mid_x = width // 2
    game_map[mid_y, mid_x] = "agent.red"

    env_config = {
        "game": {
            "max_steps": max_steps,
            "num_agents": 2,
            "obs_width": 3,
            "obs_height": 3,
            "actions": {
                # don't really care about the actions for this test
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": False},
                "put_items": {"enabled": False},
                "get_items": {"enabled": False},
                "swap": {"enabled": False},
                "change_color": {"enabled": False},
            },
            "groups": {"red": {"id": 0, "props": {}}},
            "objects": {
                "wall": {"type_id": 1, "hp": 100},
                "block": {"type_id": 2, "hp": 100},
            },
            "agent": {
                "inventory_size": 0,
            },
        }
    }

    return MettaGrid(env_config, game_map)


def test_truncation_at_max_steps():
    max_steps = 5
    env = create_minimal_mettagrid_env(max_steps=max_steps)
    obs, info = env.reset()

    # Noop until time runs out
    noop_action_idx = env.action_names().index("noop")
    actions = np.full((2, 2), [noop_action_idx, 0], dtype=np.int64)

    for step_num in range(1, max_steps + 1):
        obs, rewards, terminals, truncations, info = env.step(actions)
        if step_num < max_steps:
            assert not np.any(truncations), f"Truncations should be False before max_steps at step {step_num}"
            assert not np.any(terminals), f"Terminals should be False before max_steps at step {step_num}"
        else:
            assert np.all(truncations), f"Truncations should be True at max_steps (step {step_num})"
            # As per current C++ code, terminals are not explicitly set true on truncation.
            assert not np.any(terminals), f"Terminals should remain False at max_steps (step {step_num})"


def test_observation():
    env = create_minimal_mettagrid_env()
    obs, info = env.reset()
    print(obs)
    # Agent 0 starts at (1,1) and should see walls above and to the left
    # for now we treat the walls as "something non-empty"
    assert obs[0, :, 0, 1].any(), "Expected wall above agent 0"
    assert obs[0, :, 1, 0].any(), "Expected wall to left of agent 0"
    assert not obs[0, :, 2, 1].any(), "Expected empty space below agent 0"
    assert not obs[0, :, 1, 2].any(), "Expected empty space to right of agent 0"
