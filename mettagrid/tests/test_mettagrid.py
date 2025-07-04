import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate
from metta.mettagrid.mettagrid_c_config import cpp_config_dict
from metta.mettagrid.mettagrid_env import dtype_actions

NUM_AGENTS = 2
OBS_HEIGHT = 3
OBS_WIDTH = 3
NUM_OBS_TOKENS = 100
OBS_TOKEN_SIZE = 3


def create_minimal_mettagrid_c_env(max_steps=10, width=8, height=4):
    """Helper function to create a MettaGrid environment with minimal config.

    Creates a 4x8 (height x width) grid to test for width/height confusion.
    Grid layout:
        W W W W W W W W
        W A . . . . . W
        W . . . A . . W
        W W W W W W W W
    """
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

    game_config = {
        "max_steps": max_steps,
        "num_agents": NUM_AGENTS,
        "obs_width": OBS_WIDTH,
        "obs_height": OBS_HEIGHT,
        "num_observation_tokens": NUM_OBS_TOKENS,
        "inventory_item_names": ["laser", "armor"],
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
            "wall": {"type_id": 1},
        },
        "agent": {},
    }

    return MettaGrid(cpp_config_dict(game_config), game_map.tolist(), 42)


def test_grid_hash():
    """Test grid object representation and properties."""
    c_env = create_minimal_mettagrid_c_env()
    assert c_env.initial_grid_hash == 9437127895318323250


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
        """Test observation token format and content for both agents."""
        c_env = create_minimal_mettagrid_c_env()
        # These come from constants in the C++ code, and are fragile.
        TYPE_ID_FEATURE = 0
        WALL_TYPE_ID = 1
        obs, _info = c_env.reset()

        # The environment creates a 4x8 grid (height=4, width=8):
        #   0 1 2 3 4 5 6 7  (x/width)
        # 0 W W W W W W W W
        # 1 W A . . . . . W
        # 2 W . . . A . . W
        # 3 W W W W W W W W
        # (y/height)
        # Where: W=wall, A=agent, .=empty
        # Agent 0 is at grid position (1,1), Agent 1 is at (4,2)

        # Helper function to decode and print tokens
        def inspect_tokens(agent_obs, agent_name, n=10):
            """Decode and print the first n tokens of an agent's observation."""
            print(f"\n{agent_name} observation (first {n} tokens decoded):")

            # Known type IDs from the C++ code
            type_names = {
                0: "TYPE_ID_FEATURE",
                8: "EPISODE_COMPLETION_PCT",
                9: "LAST_ACTION",
                10: "LAST_ACTION_ARG",
                11: "LAST_REWARD",
            }

            # Known feature values
            feature_names = {0: "empty/other", 1: "WALL"}

            for i in range(min(n, len(agent_obs))):
                token = agent_obs[i]
                location = token[0]
                type_id = token[1]
                value = token[2]

                # Decode location using C++ PackedCoordinate
                if PackedCoordinate.is_empty(location):
                    loc_str = "EMPTY"
                    x, y = None, None
                else:
                    coords = PackedCoordinate.unpack(location)
                    if coords is not None:
                        row, col = coords
                        # Display as (x, y) = (col, row) for user convenience
                        x, y = col, row
                        loc_str = f"({x},{y})"
                    else:
                        loc_str = "INVALID"
                        x, y = None, None

                # Get type name
                type_name = type_names.get(type_id, f"Unknown({type_id})")

                # Get value interpretation
                if type_id == 0 and value in feature_names:
                    value_str = feature_names[value]
                else:
                    value_str = str(value)

                print(
                    f"  [{location:3}, {type_id:3}, {value:3}] : "
                    + f"location={loc_str}, type={type_name}, value={value_str}"
                )

        # Helper function to check for specific tokens
        def check_token_exists(agent_obs, x, y, type_id, feature_id, msg=""):
            # Use C++ PackedCoordinate.pack (row=y, col=x)
            location = PackedCoordinate.pack(y, x)
            token_matches = agent_obs[:, :] == [location, type_id, feature_id]
            assert token_matches.all(axis=1).any(), (
                f"{msg} Expected token [{location}, {type_id}, {feature_id}] at ({x}, {y})"
            )

        # Test Agent 0 (at position 1,1)
        print("Testing Agent 0 observation (at position 1,1):")
        agent0_obs = obs[0]

        # Inspect Agent 0's tokens
        inspect_tokens(agent0_obs, "Agent 0", n=15)
        print("\nAgent 0 wall tokens:")
        wall_mask = agent0_obs[:, 2] == WALL_TYPE_ID
        wall_indices = np.where(wall_mask)[0]
        for idx in wall_indices[:10]:  # Show first 10 walls
            token = agent0_obs[idx, :]
            location = token[0]
            coords = PackedCoordinate.unpack(location)
            if coords is not None:
                row, col = coords
                print(f"  Wall at relative ({col}, {row}), token: {token}")

        # Agent 0 is at grid position (1,1)
        # Agent 0 should see walls at these relative positions:
        #
        #   W W W
        #   W A .
        #   W . .
        #
        # The bottom wall is outside the 3x3 observation window
        wall_positions_agent0 = [
            (0, 0),  # top-left
            (1, 0),  # top-center
            (2, 0),  # top-right
            (0, 1),  # middle-left
            (0, 2),  # bottom-left
        ]

        all_positions = {(x, y) for x in range(3) for y in range(3)}
        no_wall_positions_agent0 = all_positions - set(wall_positions_agent0)

        for x, y in wall_positions_agent0:
            check_token_exists(agent0_obs, x, y, TYPE_ID_FEATURE, WALL_TYPE_ID, "Agent 0:")

        # For empty positions, we check that there's no wall token at that location
        for x, y in no_wall_positions_agent0:
            location = PackedCoordinate.pack(y, x)
            wall_tokens = (agent0_obs[:, 0] == location) & (agent0_obs[:, 2] == WALL_TYPE_ID)
            assert not wall_tokens.any(), f"Agent 0: Expected no wall at ({x}, {y})"

        # Verify we see the expected number of wall tokens for Agent 0
        wall_count_agent0 = np.sum(agent0_obs[:, 2] == WALL_TYPE_ID)
        print(f"Agent 0 sees {wall_count_agent0} walls")
        assert wall_count_agent0 == 5, f"Agent 0 should see exactly 5 walls, but sees {wall_count_agent0}"

        # Test Agent 1 (at position 4,2)
        print("\nTesting Agent 1 observation (at position 4,2):")
        agent1_obs = obs[1]

        # Inspect Agent 1's tokens
        inspect_tokens(agent1_obs, "Agent 1", n=15)
        print("\nAgent 1 wall tokens:")
        wall_mask = agent1_obs[:, 2] == WALL_TYPE_ID
        wall_indices = np.where(wall_mask)[0]
        for idx in wall_indices[:10]:  # Show first 10 walls
            token = agent1_obs[idx, :]
            location = token[0]
            coords = PackedCoordinate.unpack(location)
            if coords is not None:
                row, col = coords
                print(f"  Wall at relative ({col}, {row}), token: {token}")

        # Agent 1 is at grid position (4,2)
        # Agent 1 should see walls at these relative positions:
        #
        #   . . .
        #   . A .
        #   W W W
        #
        wall_positions_agent1 = [
            (0, 2),  # bottom-left
            (1, 2),  # bottom-center
            (2, 2),  # bottom-right
        ]
        all_positions = {(x, y) for x in range(3) for y in range(3)}
        no_wall_positions_agent1 = all_positions - set(wall_positions_agent1)

        for x, y in wall_positions_agent1:
            check_token_exists(agent1_obs, x, y, TYPE_ID_FEATURE, WALL_TYPE_ID, "Agent 1:")

        for x, y in no_wall_positions_agent1:
            location = PackedCoordinate.pack(y, x)
            wall_tokens = (agent1_obs[:, 0] == location) & (agent1_obs[:, 2] == WALL_TYPE_ID)
            assert not wall_tokens.any(), f"Agent 1: Expected no wall at ({x}, {y})"

        # Verify wall count for Agent 1
        wall_count_agent1 = np.sum(agent1_obs[:, 2] == WALL_TYPE_ID)
        print(f"Agent 1 sees {wall_count_agent1} walls")
        assert wall_count_agent1 == 3, f"Agent 1 should see exactly 3 walls, but sees {wall_count_agent1}"

        # Verify both agents have the empty terminator tokens
        assert (obs[0, -1, :] == [PackedCoordinate.EMPTY, 0xFF, 0xFF]).all(), "Agent 0: Last token should be empty"
        assert (obs[1, -1, :] == [PackedCoordinate.EMPTY, 0xFF, 0xFF]).all(), "Agent 1: Last token should be empty"

        # Verify observation shape
        assert obs.shape[0] == 2, f"Expected 2 agents, got {obs.shape[0]}"
        assert obs.shape[2] == 3, f"Expected 3 values per token, got {obs.shape[2]}"

        # Additional structural checks
        # Check that both agents see themselves (their own position should have agent-related tokens)
        # Agent positions in their own view should be at (1,1)
        _agent_location = PackedCoordinate.pack(1, 1)  # row=1, col=1

        # Check global tokens are present (first 4 tokens)
        for agent_idx in range(2):
            # First 4 tokens should be global tokens with location 0x11 (17)
            for token_idx in range(4):
                assert obs[agent_idx, token_idx, 0] == 17, (
                    f"Agent {agent_idx}: Global token {token_idx} should have location 17"
                )

        # Verify no duplicate wall tokens at the same location
        for agent_idx, agent_obs in enumerate([agent0_obs, agent1_obs]):
            wall_locations = []
            wall_mask = agent_obs[:, 2] == WALL_TYPE_ID
            wall_indices = np.where(wall_mask)[0]
            for idx in wall_indices:
                location = agent_obs[idx, 0]
                assert location not in wall_locations, f"Agent {agent_idx}: Duplicate wall token at location {location}"
                wall_locations.append(location)

        print("\nAll observation assertions passed!")

    def test_observation_token_order(self):
        """Test observation token order."""
        c_env = create_minimal_mettagrid_c_env()
        obs, _info = c_env.reset()
        distances = []
        # skip the first 4 (global) tokens
        for location in obs[0, 4:, 0]:
            coords = PackedCoordinate.unpack(location)
            if coords is not None:
                row, col = coords
                # Manhattan distance from agent position (1,1)
                distances.append(abs(col - 1) + abs(row - 1))
            elif not PackedCoordinate.is_empty(location):
                # If not empty but also not unpacked, something is wrong
                raise ValueError(f"Invalid location byte: {location}")

        # Check that distances are non-decreasing (allowing for ties)
        assert distances == sorted(distances), f"Distances should be non-decreasing: {distances}"


def test_packed_coordinate():
    """Test the PackedCoordinate functionality directly."""
    # Test constants
    assert PackedCoordinate.EMPTY == 0xFF
    assert PackedCoordinate.MAX_PACKABLE_COORD == 15

    # Test all valid coordinates
    successfully_packed = 0
    failed_coordinates = []

    for row in range(16):
        for col in range(16):
            packed = PackedCoordinate.pack(row, col)
            unpacked = PackedCoordinate.unpack(packed)

            if unpacked is None:
                failed_coordinates.append((row, col, packed))
            else:
                assert unpacked == (row, col), f"Packing/unpacking mismatch for ({row}, {col})"
                successfully_packed += 1

    # Verify we can pack 255 out of 256 positions (all except (15,15))
    assert successfully_packed == 255, f"Expected 255 packable positions, got {successfully_packed}"
    assert len(failed_coordinates) == 1, f"Expected 1 unpackable position, got {len(failed_coordinates)}"
    assert failed_coordinates[0] == (15, 15, 0xFF), "Only (15,15) should fail to unpack"

    # Test the four corners explicitly
    corners = [
        ((0, 0), 0x00, True),  # Top-left: packable
        ((0, 15), 0x0F, True),  # Top-right: packable
        ((15, 0), 0xF0, True),  # Bottom-left: packable
        ((15, 15), 0xFF, False),  # Bottom-right: NOT packable (conflicts with EMPTY)
    ]

    print("\nTesting corner cases:")
    for (row, col), expected_packed, should_work in corners:
        packed = PackedCoordinate.pack(row, col)
        assert packed == expected_packed, f"({row},{col}) should pack to {expected_packed:#04x}, got {packed:#04x}"

        unpacked = PackedCoordinate.unpack(packed)
        if should_work:
            assert unpacked == (row, col), f"Corner ({row},{col}) should unpack correctly"
            print(f"  ✓ ({row:2},{col:2}) -> {packed:#04x} -> {unpacked}")
        else:
            assert unpacked is None, f"Corner ({row},{col}) should not unpack (conflicts with EMPTY)"
            assert PackedCoordinate.is_empty(packed), f"({row},{col}) packed value should be EMPTY"
            print(f"  ✗ ({row:2},{col:2}) -> {packed:#04x} -> None (EMPTY marker)")

    # Test some regular positions
    test_cases = [
        (0, 0),  # Origin
        (1, 1),  # Common position
        (7, 7),  # Middle of range
        (14, 14),  # Almost at limit
        (15, 14),  # Max row, not max col
        (14, 15),  # Max col, not max row
    ]

    print("\nTesting regular positions:")
    for row, col in test_cases:
        packed = PackedCoordinate.pack(row, col)
        unpacked = PackedCoordinate.unpack(packed)
        assert unpacked == (row, col), f"Failed for ({row}, {col})"
        print(f"  ✓ ({row:2},{col:2}) -> {packed:#04x} -> {unpacked}")

    # Test empty location
    assert PackedCoordinate.is_empty(0xFF)
    assert not PackedCoordinate.is_empty(0x00)
    assert not PackedCoordinate.is_empty(0xF0)
    assert not PackedCoordinate.is_empty(0x0F)
    assert PackedCoordinate.unpack(0xFF) is None

    # Test invalid coordinates
    invalid_coords = [(16, 0), (0, 16), (16, 16), (255, 0), (0, 255)]
    for row, col in invalid_coords:
        try:
            PackedCoordinate.pack(row, col)
            raise AssertionError(f"Should have raised exception for ({row}, {col})")
        except Exception as e:
            print(f"  ✓ ({row},{col}) correctly raised exception: {type(e).__name__}")

    print("\nPackedCoordinate tests passed!")
    print(f"Summary: Can pack {successfully_packed}/256 positions (99.6% coverage)")
    print("Limitation: Cannot represent coordinate (15,15) due to EMPTY marker conflict")


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
        if obj.get("type_id") == 1:
            # Walls
            assert set(obj) == {"type_id", "type_name"} | common_properties
        if obj.get("type_id") == 0:
            # Agents
            assert set(obj).issuperset(
                {"agent:group", "agent:frozen", "agent:orientation", "agent:color"} | common_properties
            )
            assert obj["agent:group"] == 0, "Agent should be in group 0"
            assert obj["agent:frozen"] == 0, "Agent should not be frozen"


def test_environment_initialization():
    """Test basic environment initialization and configuration."""
    c_env = create_minimal_mettagrid_c_env()

    # Test basic properties
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
    initial_width = c_env.map_width
    initial_height = c_env.map_height

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
    assert c_env.map_width == initial_width, "Map width should remain consistent"
    assert c_env.map_height == initial_height, "Map height should remain consistent"

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
