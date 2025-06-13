import numpy as np

from mettagrid.mettagrid_c import MettaGrid
from mettagrid.mettagrid_env import dtype_actions

NUM_AGENTS = 2
OBS_HEIGHT = 3
OBS_WIDTH = 3
NUM_OBS_TOKENS = 100
OBS_TOKEN_SIZE = 3


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
            "use_observation_tokens": True,
            "num_observation_tokens": NUM_OBS_TOKENS,
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
    """Test that environments properly truncate at max_steps."""
    max_steps = 5
    c_env = create_minimal_mettagrid_c_env(max_steps=max_steps)
    _obs, _info = c_env.reset()

    # Noop until time runs out
    noop_action_idx = c_env.action_names().index("noop")
    actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=dtype_actions)

    for step_num in range(1, max_steps + 1):
        _obs, _rewards, terminals, truncations, _info = c_env.step(actions)
        if step_num < max_steps:
            assert not np.any(truncations), f"Truncations should be False before max_steps at step {step_num}"
            assert not np.any(terminals), f"Terminals should be False before max_steps at step {step_num}"
        else:
            assert np.all(truncations), f"Truncations should be True at max_steps (step {step_num})"
            # As per current C++ code, terminals are not explicitly set true on truncation.
            assert not np.any(terminals), f"Terminals should remain False at max_steps (step {step_num})"


class TestObservations:
    """Test observation functionality and formats."""

    def test_observation_tokens(self):
        """Test observation token format and content."""
        c_env = create_minimal_mettagrid_c_env()
        # These come from constants in the C++ code, and are fragile.
        TYPE_ID_FEATURE = 0
        WALL_TYPE_ID = 1
        obs, _info = c_env.reset()
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

    def test_observation_token_order(self):
        """Test observation token order."""
        c_env = create_minimal_mettagrid_c_env()
        obs, _info = c_env.reset()
        distances = []
        # skip the first 4 (global) tokens
        for location in obs[0, 4:, 0]:
            # cast as ints to avoid numpy uint8 underflow
            x = int(location >> 4)
            y = int(location & 0xF)
            distances.append(abs(x - 1) + abs(y - 1))  # 1,1 is the agent's location
        assert distances == sorted(distances), f"Distances should be increasing: {distances}"


def test_grid_objects():
    """Test grid object representation and properties."""
    c_env = create_minimal_mettagrid_c_env()
    objects = c_env.grid_objects()

    # Test that we have the expected number of objects
    # 4 walls on each side (minus corners) + 2 agents
    expected_walls = 2 * (c_env.map_width + c_env.map_height - 2)
    expected_agents = 2
    assert len(objects) == expected_walls + expected_agents, "Wrong number of objects"

    common_properties = {"r", "c", "layer", "type", "id"}

    for obj in objects.values():
        if obj.get("type_id"):
            assert set(obj) == {"type_id", "hp", "swappable"} | common_properties
            assert obj["type_id"] == 1, "Wall should have type 1"
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


def test_environment_initialization():
    """Test basic environment initialization and configuration."""
    c_env = create_minimal_mettagrid_c_env()

    # Test basic properties
    assert c_env.map_width == 5, "Map width should be 5"
    assert c_env.map_height == 5, "Map height should be 5"
    assert len(c_env.action_names()) > 0, "Should have available actions"
    assert len(c_env.feature_normalizations()) > 0, "Should have feature normalizations"

    # Test reset functionality
    obs, info = c_env.reset()
    assert obs.shape == (NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), (
        "Observation shape should match expected dimensions"
    )
    assert isinstance(info, dict), "Info should be a dictionary"


def test_action_interface():
    """Test action interface and basic action execution."""
    c_env = create_minimal_mettagrid_c_env()
    c_env.reset()

    # Test action names
    action_names = c_env.action_names()
    assert "noop" in action_names, "Noop action should be available"
    assert "move" in action_names, "Move action should be available"
    assert "rotate" in action_names, "Rotate action should be available"

    # Test basic action execution
    noop_action_idx = action_names.index("noop")
    actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=dtype_actions)

    obs, rewards, terminals, truncations, info = c_env.step(actions)

    # Verify return types and shapes
    assert obs.shape == (NUM_AGENTS, NUM_OBS_TOKENS, OBS_TOKEN_SIZE), "Step observation shape should be correct"
    assert rewards.shape == (NUM_AGENTS,), "Rewards shape should match number of agents"
    assert terminals.shape == (NUM_AGENTS,), "Terminals shape should match number of agents"
    assert truncations.shape == (NUM_AGENTS,), "Truncations shape should match number of agents"
    assert isinstance(info, dict), "Step info should be a dictionary"

    # Test action success tracking
    action_success: list[bool] = c_env.action_success()
    assert len(action_success) == NUM_AGENTS, "Action success length should match number of agents"
    assert all(isinstance(x, bool) for x in action_success), "Action success should be boolean"


def test_environment_state_consistency():
    """Test that environment state remains consistent across operations."""
    c_env = create_minimal_mettagrid_c_env()

    # Initial state
    _obs1, _info1 = c_env.reset()
    initial_objects = c_env.grid_objects()

    # Take a noop action (should not change world state significantly)
    noop_action_idx = c_env.action_names().index("noop")
    actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=dtype_actions)

    _obs2, _rewards, _terminals, _truncations, _info2 = c_env.step(actions)
    post_step_objects = c_env.grid_objects()

    # Object count should remain the same
    assert len(initial_objects) == len(post_step_objects), "Object count should remain consistent after noop actions"

    # Map dimensions should remain the same
    assert c_env.map_width == 5, "Map width should remain consistent"
    assert c_env.map_height == 5, "Map height should remain consistent"

    # Action lists should remain consistent
    actions1 = c_env.action_names()

    # Take another step
    c_env.step(actions)

    actions2 = c_env.action_names()

    assert actions1 == actions2, "Action names should remain consistent"


class TestGlobalTokens:
    """Test global token functionality and formats."""

    def test_global_tokens(self):
        """Test global token format and content."""
        max_steps = 10
        c_env = create_minimal_mettagrid_c_env(max_steps=max_steps)
        obs, _info = c_env.reset()

        # These come from constants in the C++ code
        EPISODE_COMPLETION_PCT = 8
        LAST_ACTION = 9
        LAST_ACTION_ARG = 10
        LAST_REWARD = 11

        # Initial state checks
        assert obs[0, 0, 1] == EPISODE_COMPLETION_PCT, "First token should be episode completion percentage"
        assert obs[0, 0, 2] == 0, "Episode completion should start at 0%"
        assert obs[0, 1, 1] == LAST_ACTION, "Second token should be last action"
        assert obs[0, 1, 2] == 0, "Last action should be 0"
        assert obs[0, 2, 1] == LAST_ACTION_ARG, "Third token should be last action arg"
        assert obs[0, 2, 2] == 0, "Last action arg should start at 0"
        assert obs[0, 3, 1] == LAST_REWARD, "Fourth token should be last reward"
        assert obs[0, 3, 2] == 0, "Last reward should start at 0"

        # Take a step with a noop action
        noop_action_idx = c_env.action_names().index("noop")
        actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=dtype_actions)
        obs, rewards, terminals, truncations, _info = c_env.step(actions)

        # Check tokens after first step
        expected_completion = int(round((1 / max_steps) * 255))  # 10% completion
        assert obs[0, 0, 2] == expected_completion, (
            f"Episode completion should be {expected_completion} after first step"
        )
        assert obs[0, 1, 2] == noop_action_idx, "Last action should be noop action index"
        assert obs[0, 2, 2] == 0, "Last action arg should be 0 for noop"
        assert obs[0, 3, 2] == 0, "Last reward should be 0 for noop"

        # Take another step with a move action
        move_action_idx = c_env.action_names().index("move")
        actions = np.full((NUM_AGENTS, 2), [move_action_idx, 1], dtype=dtype_actions)  # Use arg 1 to move backwards
        obs, rewards, terminals, truncations, _info = c_env.step(actions)

        # Check tokens after second step
        expected_completion = int(round((2 / max_steps) * 255))  # 20% completion
        assert obs[0, 0, 2] == expected_completion, (
            f"Episode completion should be {expected_completion} after second step"
        )
        assert obs[0, 1, 2] == move_action_idx, "Last action should be move action index"
        assert obs[0, 2, 2] == 1, "Last action arg should be 1 for backwards move"
        assert obs[0, 3, 2] == 0, "Last reward should be 0 for failed move"
