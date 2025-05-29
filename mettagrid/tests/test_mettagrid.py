import numpy as np

from mettagrid.mettagrid_c import MettaGrid

NUM_AGENTS = 2
OBS_HEIGHT = 3
OBS_WIDTH = 3


def create_minimal_mettagrid_c_env(max_steps=10, width=5, height=5, use_observation_tokens=False):
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
    """Test that environments properly truncate at max_steps."""
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


class TestObservations:
    """Test observation functionality and formats."""

    def test_observation_tokens(self):
        c_env = create_minimal_mettagrid_c_env(use_observation_tokens=True)

        # These come from constants in the C++ code, and are fragile.
        TYPE_ID_FEATURE = 1
        WALL_TYPE_ID = 1

        obs, info = c_env.reset()

        # obs shape should be [num_agents, num_tokens, 3]
        # Each token consists of [location, feature_id, value]
        print(f"Observation shape: {obs.shape}")

        # The observation is already in the correct shape
        num_agents, num_tokens, _ = obs.shape
        obs_reshaped = obs  # No need to reshape

        print(f"Observation shape (already correct): {obs_reshaped.shape}")
        print(f"Agent 0 tokens:\n{obs_reshaped[0]}")

        # Agent 0 starts at (1,1) and should see walls above and to the left
        # The location encoding is: (obs_r << 4) | obs_c
        # where obs_r and obs_c are relative to the agent's observation window

        # For agent at (1,1), walls should be visible at relative positions:
        # - Wall above: relative position (0, 1) -> location = 0 << 4 | 1 = 1
        # - Wall to left: relative position (1, 0) -> location = 1 << 4 | 0 = 16

        expected_wall_locations = [1, 16]  # encoded locations for walls above and left

        found_walls = []
        for token_idx in range(num_tokens):
            token = obs_reshaped[0, token_idx]
            location, feature_id, value = token

            # Skip empty tokens (all zeros)
            if location == 0 and feature_id == 0 and value == 0:
                continue

            print(f"Token {token_idx}: location={location}, feature_id={feature_id}, value={value}")

            # Check if this is a wall token
            if feature_id == TYPE_ID_FEATURE and value == WALL_TYPE_ID:
                found_walls.append(location)

        print(f"Found walls at encoded locations: {found_walls}")
        print(f"Expected wall locations: {expected_wall_locations}")

        # Check that we found walls at the expected locations
        for expected_location in expected_wall_locations:
            assert expected_location in found_walls, f"Expected wall at encoded location {expected_location}"

        # Check that no tokens exist at positions where there shouldn't be walls
        # For example, positions (2,1) and (1,2) relative to agent should be empty
        # These would be encoded as: (2 << 4 | 1) = 33 and (1 << 4 | 2) = 18
        unexpected_locations = [33, 18]  # encoded locations where we don't expect objects

        all_token_locations = []
        for token_idx in range(num_tokens):
            token = obs_reshaped[0, token_idx]
            location, feature_id, value = token

            # Skip empty tokens
            if location == 0 and feature_id == 0 and value == 0:
                continue

            all_token_locations.append(location)

        for unexpected_location in unexpected_locations:
            assert unexpected_location not in all_token_locations, (
                f"Unexpected token at encoded location {unexpected_location}"
            )

        print("Test passed: observation tokens are correctly formatted and positioned")

    def test_observations(self):
        """Test standard observation format and content."""
        c_env = create_minimal_mettagrid_c_env()
        wall_feature_idx = c_env.grid_features().index("wall")
        obs, info = c_env.reset()
        # Agent 0 starts at (1,1) and should see walls above and to the left
        assert obs[0, 0, 1, wall_feature_idx] == 1, "Expected wall above agent 0"
        assert obs[0, 1, 0, wall_feature_idx] == 1, "Expected wall to left of agent 0"
        assert not obs[0, 2, 1, :].any(), "Expected empty space below agent 0"
        assert not obs[0, 1, 2, :].any(), "Expected empty space to right of agent 0"


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


def test_environment_initialization():
    """Test basic environment initialization and configuration."""
    c_env = create_minimal_mettagrid_c_env()

    # Test basic properties
    assert c_env.map_width == 5, "Map width should be 5"
    assert c_env.map_height == 5, "Map height should be 5"
    assert len(c_env.action_names()) > 0, "Should have available actions"
    assert len(c_env.grid_features()) > 0, "Should have grid features"

    # Test reset functionality
    obs, info = c_env.reset()
    assert obs.shape == (NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, len(c_env.grid_features())), (
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
    actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=np.int64)

    obs, rewards, terminals, truncations, info = c_env.step(actions)

    # Verify return types and shapes
    assert obs.shape == (NUM_AGENTS, OBS_HEIGHT, OBS_WIDTH, len(c_env.grid_features())), (
        "Step observation shape should be correct"
    )
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
    actions = np.full((NUM_AGENTS, 2), [noop_action_idx, 0], dtype=np.int64)

    _obs2, _rewards, _terminals, _truncations, _info2 = c_env.step(actions)
    post_step_objects = c_env.grid_objects()

    # Object count should remain the same
    assert len(initial_objects) == len(post_step_objects), "Object count should remain consistent after noop actions"

    # Map dimensions should remain the same
    assert c_env.map_width == 5, "Map width should remain consistent"
    assert c_env.map_height == 5, "Map height should remain consistent"

    # Feature and action lists should remain consistent
    features1 = c_env.grid_features()
    actions1 = c_env.action_names()

    # Take another step
    c_env.step(actions)

    features2 = c_env.grid_features()
    actions2 = c_env.action_names()

    assert features1 == features2, "Grid features should remain consistent"
    assert actions1 == actions2, "Action names should remain consistent"
