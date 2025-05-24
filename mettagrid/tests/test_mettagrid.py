import numpy as np
import pytest

from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611

NUM_AGENTS = 2
OBS_HEIGHT = 3
OBS_WIDTH = 3


def create_minimal_mettagrid_env(max_steps=10, width=5, height=5, use_observation_tokens=False):
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
            "use_observation_tokens": use_observation_tokens,
            "num_observation_tokens": 100,
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
                "hp": 100,
            },
        }
    }

    return MettaGrid(env_config, game_map.tolist())


def test_truncation_at_max_steps():
    max_steps = 5
    env = create_minimal_mettagrid_env(max_steps=max_steps)
    obs, info = env.reset()

    # Noop until time runs out
    noop_action_idx = env.action_names().index("noop")
    actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=np.int64)

    for step_num in range(1, max_steps + 1):
        obs, rewards, terminals, truncations, info = env.step(actions)
        if step_num < max_steps:
            assert not np.any(truncations), f"Truncations should be False before max_steps at step {step_num}"
            assert not np.any(terminals), f"Terminals should be False before max_steps at step {step_num}"
        else:
            assert np.all(truncations), f"Truncations should be True at max_steps (step {step_num})"
            # As per current C++ code, terminals are not explicitly set true on truncation.
            assert not np.any(terminals), f"Terminals should remain False at max_steps (step {step_num})"


class TestObservations:
    def test_observation_tokens(self):
        env = create_minimal_mettagrid_env(use_observation_tokens=True)
        # These come from constants in the C++ code, and are fragile.
        TYPE_ID_FEATURE = 1
        WALL_TYPE_ID = 1
        obs, info = env.reset()
        # Agent 0 starts at (1,1) and should see walls above and to the left
        # for now we treat the walls as "something non-empty"
        for x, y in [(0, 1), (1, 0)]:
            location = x << 4 | y
            token_matches = obs[0, :, :] == [location, TYPE_ID_FEATURE, WALL_TYPE_ID]
            assert token_matches.all(axis=1).any(), f"Expected wall at location {x}, {y}"
        for x, y in [(2, 1), (1, 2)]:
            location = x << 4 | y
            token_matches = obs[0, :, 0] == location
            assert not token_matches.any(), f"Expected no tokens at location {x}, {y}"

    def test_observations(self):
        env = create_minimal_mettagrid_env()
        wall_feature_idx = env.grid_features().index("wall")
        obs, info = env.reset()
        # Agent 0 starts at (1,1) and should see walls above and to the left
        assert obs[0, 0, 1, wall_feature_idx] == 1, "Expected wall above agent 0"
        assert obs[0, 1, 0, wall_feature_idx] == 1, "Expected wall to left of agent 0"
        assert not obs[0, 2, 1, :].any(), "Expected empty space below agent 0"
        assert not obs[0, 1, 2, :].any(), "Expected empty space to right of agent 0"


def test_grid_objects():
    env = create_minimal_mettagrid_env()
    objects = env.grid_objects()

    # Test that we have the expected number of objects
    # 4 walls on each side (minus corners) + 2 agents
    expected_walls = 2 * (env.map_width() + env.map_height() - 2)
    expected_agents = 2
    assert len(objects) == expected_walls + expected_agents, "Wrong number of objects"

    common_properties = {"r", "c", "layer", "type", "id"}

    for obj in objects.values():
        if obj.get("wall"):
            assert set(obj) == {"wall", "hp", "swappable"} | common_properties
            assert obj["wall"] == 1, "Wall should have type 1"
            assert obj["hp"] == 100, "Wall should have 100 hp"
        if obj.get("agent"):
            # agents will also have various inventory, which we don't list here
            assert set(obj).issuperset(
                {"agent", "agent:group", "hp", "agent:frozen", "agent:orientation", "agent:color", "inv:heart"}
                | common_properties
            )
            assert obj["agent"] == 1, "Agent should have type 1"
            assert obj["agent:group"] == 0, "Agent should be in group 0"
            assert obj["hp"] == 100, "Agent should have 100 hp"
            assert obj["agent:frozen"] == 0, "Agent should not be frozen"


class TestSetBuffers:
    def test_default_buffers(self):
        env = create_minimal_mettagrid_env()
        env.reset()

        noop_action_idx = env.action_names().index("noop")
        actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=np.int64)
        obs, rewards, terminals, truncations, info = env.step(actions)
        episode_rewards = env.get_episode_rewards()

        # Check strides. We've had issues where we've not correctly initialized the buffers, and have had
        # strides of zero.
        assert rewards.strides == (4,)  # float32
        assert terminals.strides == (1,)  # bool, tracked as a byte
        assert truncations.strides == (1,)  # bool, tracked as a byte
        assert episode_rewards.strides == (4,)  # float32
        assert obs.strides[-1] == 1  # uint8

        # This is a more brute force way to check that the buffers are behaving correctly by changing a single
        # element and making sure the correct update is reflected. Given that the strides are correct, these tests
        # are probably superfluous; but we've been surprised by what can fail in the past, so we're aiming for
        # overkill.
        assert (rewards == [0, 0]).all()
        assert (terminals == [False, False]).all()
        assert (truncations == [False, False]).all()
        assert (episode_rewards == [0, 0]).all()

        rewards[0] = 1
        terminals[0] = True
        truncations[0] = True
        episode_rewards[0] = 1

        assert (rewards == [1, 0]).all()
        assert (terminals == [True, False]).all()
        assert (truncations == [True, False]).all()
        assert (episode_rewards == [1, 0]).all()

        # Obs is non-empty, so we treat it differently than the others.
        initial_obs_sum = obs.sum()
        obs[0, 0, 0, 0] += 1
        assert obs.sum() == initial_obs_sum + 1

    def test_set_buffers_wrong_shape(self):
        env = create_minimal_mettagrid_env()
        num_features = len(env.grid_features())
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        # Wrong number of agents
        observations = np.zeros((3, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="observations"):
            env.set_buffers(observations, terminals, truncations, rewards)

        # Wrong observation height
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT + 1, OBS_WIDTH, num_features), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="observations"):
            env.set_buffers(observations, terminals, truncations, rewards)

        # Wrong observation width
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH - 1, num_features), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="observations"):
            env.set_buffers(observations, terminals, truncations, rewards)

        # Wrong number of features
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features + 1), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="observations"):
            env.set_buffers(observations, terminals, truncations, rewards)

    def test_set_buffers_wrong_dtype(self):
        env = create_minimal_mettagrid_env()
        num_features = len(env.grid_features())
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np.float32)
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        with pytest.raises(TypeError):
            env.set_buffers(observations, terminals, truncations, rewards)

    def test_set_buffers_non_contiguous(self):
        env = create_minimal_mettagrid_env()
        num_features = len(env.grid_features())
        observations = np.asfortranarray(np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np.uint8))
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        with pytest.raises(TypeError):
            env.set_buffers(observations, terminals, truncations, rewards)

    def test_set_buffers_happy_path(self):
        env = create_minimal_mettagrid_env()
        num_features = len(env.grid_features())
        observations = np.zeros((NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, num_features), dtype=np.uint8)
        terminals = np.zeros(NUM_AGENTS, dtype=bool)
        truncations = np.zeros(NUM_AGENTS, dtype=bool)
        rewards = np.zeros(NUM_AGENTS, dtype=np.float32)

        env.set_buffers(observations, terminals, truncations, rewards)
        observations_from_env, info = env.reset()
        np.testing.assert_array_equal(observations_from_env, observations)
