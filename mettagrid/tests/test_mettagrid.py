import numpy as np
import pytest

from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611

NUM_AGENTS = 2
OBS_HEIGHT = 3
OBS_WIDTH = 3

# Rebuild the NumPy types using the exposed function
np_observations_type = np.dtype(MettaGrid.get_numpy_type_name("observations"))
np_terminals_type = np.dtype(MettaGrid.get_numpy_type_name("terminals"))
np_truncations_type = np.dtype(MettaGrid.get_numpy_type_name("truncations"))
np_rewards_type = np.dtype(MettaGrid.get_numpy_type_name("rewards"))
np_actions_type = np.dtype(MettaGrid.get_numpy_type_name("actions"))
np_masks_type = np.dtype(MettaGrid.get_numpy_type_name("masks"))
np_success_type = np.dtype(MettaGrid.get_numpy_type_name("success"))


def create_minimal_mettagrid_c_env(max_steps=10, width=5, height=5):
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
            "num_agents": NUM_AGENTS,
            "obs_width": OBS_WIDTH,
            "obs_height": OBS_HEIGHT,
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

    c_env = MettaGrid(env_config, game_map)
    num_features = len(c_env.grid_features())
    observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
    terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
    truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
    rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)
    c_env.set_buffers(observations, terminals, truncations, rewards)

    return c_env


def test_truncation_at_max_steps():
    max_steps = 5
    c_env = create_minimal_mettagrid_c_env(max_steps=max_steps)
    obs, info = c_env.reset()

    # Noop until time runs out
    noop_action_idx = c_env.action_names().index("noop")
    actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=np.int64)

    for step_num in range(1, max_steps + 1):
        obs, rewards, terminals, truncations, info = c_env.step(actions)
        if step_num < max_steps:
            assert not np.any(truncations), f"Truncations should be False before max_steps at step {step_num}"
            assert not np.any(terminals), f"Terminals should be False before max_steps at step {step_num}"
        else:
            assert np.all(truncations), f"Truncations should be True at max_steps (step {step_num})"
            # As per current C++ code, terminals are not explicitly set true on truncation.
            assert not np.any(terminals), f"Terminals should remain False at max_steps (step {step_num})"


def test_observation():
    env = create_minimal_mettagrid_c_env()
    wall_feature_idx = env.grid_features().index("wall")
    obs, info = env.reset()
    # Agent 0 starts at (1,1) and should see walls above and to the left
    # for now we treat the walls as "something non-empty"
    assert obs[0, 0, 1, wall_feature_idx] == 1, "Expected wall above agent 0"
    assert obs[0, 1, 0, wall_feature_idx] == 1, "Expected wall to left of agent 0"
    assert not obs[0, 2, 1, :].any(), "Expected empty space below agent 0"
    assert not obs[0, 1, 2, :].any(), "Expected empty space to right of agent 0"


class TestSetBuffers:
    def test_set_buffers_wrong_shape(self):
        env = create_minimal_mettagrid_c_env()
        num_features = len(env.grid_features())
        terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
        truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
        rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)

        # Wrong number of agents
        observations = np.zeros((3, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
        with pytest.raises(RuntimeError, match="observations"):
            env.set_buffers(observations, terminals, truncations, rewards)

        # Wrong observation height
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT + 1, OBS_WIDTH, num_features), dtype=np_observations_type)
        with pytest.raises(RuntimeError, match="observations"):
            env.set_buffers(observations, terminals, truncations, rewards)

        # Wrong observation width
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH - 1, num_features), dtype=np_observations_type)
        with pytest.raises(RuntimeError, match="observations"):
            env.set_buffers(observations, terminals, truncations, rewards)

        # Wrong number of features
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features + 1), dtype=np_observations_type)
        with pytest.raises(RuntimeError, match="observations"):
            env.set_buffers(observations, terminals, truncations, rewards)

    def test_set_buffers_wrong_dtype(self):
        env = create_minimal_mettagrid_c_env()
        num_features = len(env.grid_features())
        wrong_type = np.float32
        assert wrong_type != np_observations_type
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=wrong_type)
        terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
        truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
        rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)

        with pytest.raises(TypeError):
            env.set_buffers(observations, terminals, truncations, rewards)

    def test_set_buffers_non_contiguous(self):
        env = create_minimal_mettagrid_c_env()
        num_features = len(env.grid_features())
        observations = np.asfortranarray(
            np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
        )
        terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
        truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
        rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)

        with pytest.raises(TypeError):
            env.set_buffers(observations, terminals, truncations, rewards)

    def test_set_buffers_happy_path(self):
        env = create_minimal_mettagrid_c_env()
        num_features = len(env.grid_features())
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np_observations_type)
        terminals = np.zeros(NUM_AGENTS, dtype=np_terminals_type)
        truncations = np.zeros(NUM_AGENTS, dtype=np_truncations_type)
        rewards = np.zeros(NUM_AGENTS, dtype=np_rewards_type)

        env.set_buffers(observations, terminals, truncations, rewards)
        observations_from_env, _info = env.reset()
        np.testing.assert_array_equal(observations_from_env, observations)
