import numpy as np
import pytest

from mettagrid.mettagrid_c import MettaGrid  # pylint: disable=E0611

NUM_AGENTS = 2
OBS_HEIGHT = 3
OBS_WIDTH = 3


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

    return MettaGrid(env_config, game_map)


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


def test_observation():
    env = create_minimal_mettagrid_env()
    wall_feature_idx = env.grid_features().index("wall")
    obs, info = env.reset()
    # Agent 0 starts at (1,1) and should see walls above and to the left
    # for now we treat the walls as "something non-empty"
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
    
    for obj_id, obj in objects.items():
        print(obj)
        if obj.get("wall"):
            assert set(obj) == {"wall", "hp", "swappable"} | common_properties
            # assert obj["hp"] == 100, "Wall should have 100 hp"
            # assert "swappable" in obj, "Wall should have swappable property"
    # # Test wall properties
    # wall_id = None
    # for obj_id, obj in objects.items():
    #     if obj["wall"] == 1:  # Wall type
    #         wall_id = obj_id
    #         break
    
    # assert wall_id is not None, "No wall found"
    # wall = objects[wall_id]
    # assert wall["hp"] == 100, "Wall should have 100 hp"
    # assert "swappable" in wall, "Wall should have swappable property"
    
    # # Test agent properties
    # agent_ids = []
    # for obj_id, obj in objects.items():
    #     if obj["type"] == 0:  # Agent type
    #         agent_ids.append(obj_id)
    
    # assert len(agent_ids) == 2, "Should have 2 agents"
    
    # # Test first agent (upper left)
    # agent1 = objects[agent_ids[0]]
    # assert agent1["r"] == 1, "First agent should be at row 1"
    # assert agent1["c"] == 1, "First agent should be at column 1"
    # assert agent1["group"] == 0, "Agent should be in group 0"
    # assert agent1["agent_id"] in [0, 1], "Agent should have an agent_id"
    
    # # Test second agent (middle)
    # agent2 = objects[agent_ids[1]]
    # mid_y = env.map_height() // 2
    # mid_x = env.map_width() // 2
    # assert agent2["r"] == mid_y, "Second agent should be in middle row"
    # assert agent2["c"] == mid_x, "Second agent should be in middle column"
    # assert agent2["group"] == 0, "Agent should be in group 0"
    # assert agent2["agent_id"] in [0, 1], "Agent should have an agent_id"
    # assert agent2["agent_id"] != agent1["agent_id"], "Agents should have different agent_ids"


class TestSetBuffers:
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
