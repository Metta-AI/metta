import numpy as np
import pytest

from mettagrid.config.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    ChangeGlyphActionConfig,
    ConverterConfig,
    GameConfig,
    GlobalObsConfig,
    MettaGridConfig,
    WallConfig,
)
from mettagrid.core import MettaGridCore
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.mettagrid_c import PackedCoordinate, dtype_actions
from mettagrid.test_support import ObservationHelper, Orientation, TokenTypes
from mettagrid.test_support.actions import action_index

NUM_OBS_TOKENS = 50


@pytest.fixture
def basic_env() -> MettaGridCore:
    """Create a basic test environment."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=1000,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=NUM_OBS_TOKENS,
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
                rotate=ActionConfig(),
                get_items=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
            resource_names=["laser", "armor", "heart"],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#", "#", "#", "#"],
                    ["#", "@", ".", ".", ".", ".", ".", "#"],
                    ["#", ".", ".", ".", "@", ".", ".", "#"],
                    ["#", "#", "#", "#", "#", "#", "#", "#"],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )

    return MettaGridCore(cfg)


@pytest.fixture
def adjacent_agents_env() -> MettaGridCore:
    """Create an environment with adjacent agents."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=1000,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=NUM_OBS_TOKENS,
            actions=ActionsConfig(
                noop=ActionConfig(),
                move=ActionConfig(),
                rotate=ActionConfig(),
                get_items=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
            resource_names=["laser", "armor", "heart"],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "@", "@", ".", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ],
                char_to_name_map=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )

    return MettaGridCore(cfg)


class TestObservations:
    """Test observation functionality."""

    def test_observation_structure(self, basic_env):
        """Test basic observation structure."""
        obs, _ = basic_env.reset()

        # global token is always at the center of the observation window
        global_token_location = PackedCoordinate.pack(basic_env.obs_height // 2, basic_env.obs_width // 2)

        # Test global tokens (first 4 tokens)
        for agent_idx in range(basic_env.num_agents):
            for token_idx in range(4):
                assert obs[agent_idx, token_idx, 0] == global_token_location

        # Test empty terminator
        assert (obs[0, -1, :] == TokenTypes.EMPTY_TOKEN).all()
        assert (obs[1, -1, :] == TokenTypes.EMPTY_TOKEN).all()

    def test_detailed_wall_observations(self, basic_env):
        """Test detailed wall observations for both agents."""
        obs, _ = basic_env.reset()
        type_id_feature_id = basic_env.c_env.feature_spec()["type_id"]["id"]
        helper = ObservationHelper()

        # The environment creates a 4x8 grid (height=4, width=8):
        #   0 1 2 3 4 5 6 7  (x/width)
        # 0 W W W W W W W W
        # 1 W A . . . . . W
        # 2 W . . . A . . W
        # 3 W W W W W W W W
        # (y/height)
        # Where: W=wall, A=agent, .=empty
        # Agent 0 is at grid position (x/col = 1, y/row = 1), Agent 1 is at (4,2)

        # Test Agent 0 observation
        agent0_obs = obs[0]

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

        agent0_wall_tokens = helper.find_tokens(
            agent0_obs, feature_id=type_id_feature_id, value=TokenTypes.WALL_TYPE_ID
        )
        agent0_wall_positions = helper.get_positions_from_tokens(agent0_wall_tokens)
        assert set(agent0_wall_positions) == set(wall_positions_agent0), (
            f"Agent 0: Expected walls at {wall_positions_agent0}, got {agent0_wall_positions}"
        )

        # Verify wall count
        assert len(agent0_wall_tokens) == 5, "Agent 0 should see exactly 5 walls"

        # Test Agent 1 observation
        agent1_obs = obs[1]

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

        agent1_wall_tokens = helper.find_tokens(
            agent1_obs, feature_id=type_id_feature_id, value=TokenTypes.WALL_TYPE_ID
        )
        agent1_wall_positions = helper.get_positions_from_tokens(agent1_wall_tokens)
        assert set(agent1_wall_positions) == set(wall_positions_agent1), (
            f"Agent 1: Expected walls at {wall_positions_agent1}, got {agent1_wall_positions}"
        )

        # Verify wall count
        assert len(agent1_wall_tokens) == 3, "Agent 1 should see exactly 3 walls"

    def test_agents_see_each_other(self, adjacent_agents_env):
        """Test that adjacent agents can see each other."""
        obs, _ = adjacent_agents_env.reset()
        helper = ObservationHelper()

        # Debug: Let's check where agents actually are
        # Grid layout for adjacent_agents_env:
        #   0 1 2 3 4
        # 0 W W W W W
        # 1 W . . . W
        # 2 W A A . W  <- Agents at (1,2) and (2,2)
        # 3 W . . . W
        # 4 W W W W W

        # Agent 0 at (1,2) has observation window centered at (1,2)
        # Its 3x3 window covers grid positions (0,1) to (2,3)
        # Agent 1 at (2,2) is within this window

        # In Agent 0's relative coordinates:
        # Agent 0 is at center (1,1)
        # Agent 1 at grid (2,2) - Agent 0 at grid (1,2) = offset (1,0)
        # So Agent 1 should appear at observation position (1+1, 1+0) = (2,1)

        agent1_tokens = helper.find_tokens(obs[0], location=(2, 1))
        assert len(agent1_tokens) > 0, "Agent 0 should see Agent 1 at (2,1)"

        # Agent 1 at (2,2) has observation window centered at (2,2)
        # Its 3x3 window covers grid positions (1,1) to (3,3)
        # Agent 0 at (1,2) is within this window

        # In Agent 1's relative coordinates:
        # Agent 1 is at center (1,1)
        # Agent 0 at grid (1,2) - Agent 1 at grid (2,2) = offset (-1,0)
        # So Agent 0 should appear at observation position (1-1, 1+0) = (0,1)

        agent0_tokens = helper.find_tokens(obs[1], location=(0, 1))
        assert len(agent0_tokens) > 0, "Agent 1 should see Agent 0 at (0,1)"

    def test_observation_token_order(self, basic_env):
        """Test that observation tokens are ordered by distance."""
        obs, _ = basic_env.reset()

        distances = []
        # Skip global tokens (first 4)
        for location in obs[0, 4:, 0]:
            coords = PackedCoordinate.unpack(location)
            if coords:
                row, col = coords
                # Manhattan distance from agent position (1,1)
                distances.append(abs(col - 1) + abs(row - 1))

        assert distances == sorted(distances), "Tokens should be ordered by distance"


class TestGlobalTokens:
    """Test global token functionality."""

    def test_initial_global_tokens(self, basic_env):
        """Test initial global token values."""
        obs, _ = basic_env.reset()
        episode_completion_pct_feature_id = basic_env.c_env.feature_spec()["episode_completion_pct"]["id"]
        last_action_feature_id = basic_env.c_env.feature_spec()["last_action"]["id"]
        last_reward_feature_id = basic_env.c_env.feature_spec()["last_reward"]["id"]
        helper = ObservationHelper()

        # Global tokens are at the center of the observation window
        global_x = basic_env.obs_width // 2
        global_y = basic_env.obs_height // 2

        # Check token types and values
        assert helper.find_token_values(
            obs[0], location=(global_x, global_y), feature_id=episode_completion_pct_feature_id
        ) == [0]
        assert helper.find_token_values(obs[0], location=(global_x, global_y), feature_id=last_action_feature_id) == [0]
        assert helper.find_token_values(obs[0], location=(global_x, global_y), feature_id=last_reward_feature_id) == [0]

    def test_global_tokens_update(self):
        """Test that global tokens update correctly."""
        # Create basic 4x8 grid with walls around perimeter
        game_map = create_grid(4, 8, fill_value=".")
        game_map[0, :] = "#"
        game_map[-1, :] = "#"
        game_map[:, 0] = "#"
        game_map[:, -1] = "#"

        # Place agents
        game_map[1, 1] = "@"
        game_map[2, 4] = "@"

        # Create environment with max_steps=10 so that 1 step = 10% completion
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=2,
                max_steps=10,  # Important: 10 steps total so 1 step = 10%
                obs_width=3,
                obs_height=3,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move=ActionConfig(),
                    rotate=ActionConfig(),
                    get_items=ActionConfig(),
                ),
                objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
                global_obs=GlobalObsConfig(
                    episode_completion_pct=True,
                    last_action=True,
                    last_reward=True,
                ),
                resource_names=["laser", "armor", "heart"],
                map_builder=AsciiMapBuilder.Config(
                    map_data=game_map.tolist(),
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )
        env = MettaGridCore(cfg)
        episode_completion_pct_feature_id = env.c_env.feature_spec()["episode_completion_pct"]["id"]
        last_action_feature_id = env.c_env.feature_spec()["last_action"]["id"]
        obs, _ = env.reset()
        num_agents = env.num_agents
        helper = ObservationHelper()

        # Global tokens are at the center of the observation window
        global_x = env.obs_width // 2
        global_y = env.obs_height // 2

        # Take a noop action
        noop_idx = env.action_names.index("noop")
        actions = np.full(num_agents, noop_idx, dtype=dtype_actions)
        obs, _, _, _, _ = env.step(actions)

        # Check episode completion updated (1/10 = 10%)
        expected_completion = int(round(0.1 * 255))
        completion_values = helper.find_token_values(
            obs[0], location=(global_x, global_y), feature_id=episode_completion_pct_feature_id
        )
        assert completion_values == [expected_completion], (
            f"Expected completion {expected_completion}, got {completion_values}"
        )

        # Check last action
        last_action = helper.find_token_values(obs[0], location=(global_x, global_y), feature_id=last_action_feature_id)
        assert last_action == noop_idx, f"Expected last action {noop_idx}, got {last_action}"

        # Take a move action
        move_idx = action_index(env, "move", Orientation.SOUTH)
        actions = np.full(num_agents, move_idx, dtype=dtype_actions)
        obs, _, _, _, _ = env.step(actions)

        # Check updates
        expected_completion = int(round(0.2 * 255))
        completion_value = helper.find_token_values(
            obs[0], location=(global_x, global_y), feature_id=episode_completion_pct_feature_id
        )
        assert completion_value == expected_completion

        last_action = helper.find_token_values(obs[0], location=(global_x, global_y), feature_id=last_action_feature_id)
        assert last_action == move_idx

    def test_glyph_signaling(self):
        """Test that agents can signal using glyphs and observe each other's glyphs."""
        # Create a 5x5 environment with two adjacent agents
        game_map = create_grid(5, 5, fill_value=".")
        helper = ObservationHelper()

        # Add walls around perimeter
        game_map[0, :] = "#"
        game_map[-1, :] = "#"
        game_map[:, 0] = "#"
        game_map[:, -1] = "#"

        # Place two agents next to each other
        # Agent 0 at (1,2), Agent 1 at (2,2)
        game_map[2, 1] = "@"
        game_map[2, 2] = "@"

        # Create environment with change_glyph enabled and 8 glyphs
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=2,
                max_steps=10,
                obs_width=3,  # Use 3x3 observation window
                obs_height=3,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move=ActionConfig(),
                    rotate=ActionConfig(),
                    get_items=ActionConfig(),
                    change_glyph=ChangeGlyphActionConfig(enabled=True, number_of_glyphs=8),
                ),
                objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
                resource_names=["laser", "armor"],
                map_builder=AsciiMapBuilder.Config(
                    map_data=game_map.tolist(),
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )
        env = MettaGridCore(cfg)
        glyph_feature_id = env.c_env.feature_spec()["agent:glyph"]["id"]

        obs, _ = env.reset()

        # Check if we're seeing uninitialized memory issues
        agent0_self_glyph = helper.find_token_values(obs[0], location=(1, 1), feature_id=glyph_feature_id)
        agent0_sees_agent1_glyph = helper.find_token_values(obs[0], location=(2, 1), feature_id=glyph_feature_id)
        agent1_self_glyph = helper.find_token_values(obs[1], location=(1, 1), feature_id=glyph_feature_id)
        agent1_sees_agent0_glyph = helper.find_token_values(obs[1], location=(0, 1), feature_id=glyph_feature_id)

        # Initially, both agents should have glyph 0 (default)
        # Since glyph 0 is suppressed, we should NOT find any glyph tokens
        assert len(agent0_self_glyph) == 0, f"Agent 0 with glyph 0 should have no glyph token, got {agent0_self_glyph}"
        assert len(agent0_sees_agent1_glyph) == 0, (
            f"Agent 0 should see Agent 1 with no glyph token (glyph 0), got {agent0_sees_agent1_glyph}"
        )
        assert len(agent1_self_glyph) == 0, f"Agent 1 with glyph 0 should have no glyph token, got {agent1_self_glyph}"
        assert len(agent1_sees_agent0_glyph) == 0, (
            f"Agent 1 should see Agent 0 with no glyph token (glyph 0), got {agent1_sees_agent0_glyph}"
        )

        # Test changing glyphs
        def glyph_action(value: int) -> int:
            name = f"change_glyph_{value}"
            if name not in env.action_names:
                raise AssertionError(f"Missing expected action {name}")
            return env.action_names.index(name)

        # Test 1: Agent 0 changes to glyph 3, Agent 1 stays at 0
        actions = np.array(
            [
                glyph_action(3),
                glyph_action(5),
            ],
            dtype=dtype_actions,
        )

        obs, _, _, _, _ = env.step(actions)

        agent0_self_glyph = helper.find_token_values(obs[0], location=(1, 1), feature_id=glyph_feature_id)
        assert agent0_self_glyph == 3, f"Agent 0 should have glyph 3, got {agent0_self_glyph}"

        agent1_sees_agent0_glyph = helper.find_token_values(obs[1], location=(0, 1), feature_id=glyph_feature_id)
        assert agent1_sees_agent0_glyph == 3, f"Agent 1 should see Agent 0 with glyph 3, got {agent1_sees_agent0_glyph}"

        agent1_self_glyph = helper.find_token_values(obs[1], location=(1, 1), feature_id=glyph_feature_id)
        assert agent1_self_glyph == 5, f"Agent 1 should have glyph 5, got {agent1_self_glyph}"

        # Test 2: Invalid glyph values (should be no-op)
        assert "change_glyph_123" not in env.action_names, "Invalid glyph action should not exist"

        # Test 3: Changing back to glyph 0 removes the token

        # Change back to glyph 0
        actions = np.array(
            [
                glyph_action(0),
                glyph_action(0),
            ],
            dtype=dtype_actions,
        )
        obs, _, _, _, _ = env.step(actions)

        # Verify glyph tokens are gone
        agent0_glyph = helper.find_token_values(obs[0], location=(1, 1), feature_id=glyph_feature_id)
        agent1_glyph = helper.find_token_values(obs[1], location=(1, 1), feature_id=glyph_feature_id)

        assert len(agent0_glyph) == 0, f"Agent 0 changed to glyph 0 should have no token, got {agent0_glyph}"
        assert len(agent1_glyph) == 0, f"Agent 1 changed to glyph 0 should have no token, got {agent1_glyph}"


class TestEdgeObservations:
    """Test observation behavior near world edges."""

    def test_observation_off_edge_with_large_window(self):
        """Test observation window behavior when agent walks to corner of large map."""

        # Create a 15x10 grid (width=15, height=10) with 7x7 observation window
        game_map = create_grid(10, 15, fill_value=".")
        helper = ObservationHelper()

        # Add walls around perimeter
        game_map[0, :] = "#"
        game_map[-1, :] = "#"
        game_map[:, 0] = "#"
        game_map[:, -1] = "#"

        # Place agent near top-left at (2, 2)
        game_map[2, 2] = "@"

        # Place an altar at row=5, col=7 (which is position (7,5) in x,y coordinates)
        game_map[5, 7] = "_"

        # Create environment with 7x7 observation window
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=50,  # Enough steps to walk around
                obs_width=7,
                obs_height=7,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move=ActionConfig(),
                    rotate=ActionConfig(),
                    get_items=ActionConfig(),
                ),
                objects={
                    "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID),
                    "altar": ConverterConfig(
                        type_id=TokenTypes.ALTAR_TYPE_ID,
                        input_resources={"resource1": 1},
                        output_resources={"resource2": 1},
                        max_output=10,
                        conversion_ticks=5,
                        cooldown=[3],
                        initial_resource_count=0,
                    ),
                },
                resource_names=["laser", "resource1", "resource2"],  # laser required for attack action
                map_builder=AsciiMapBuilder.Config(
                    map_data=game_map.tolist(),
                    char_to_name_map=DEFAULT_CHAR_TO_NAME,
                ),
            )
        )
        env = MettaGridCore(cfg)
        type_id_feature_id = env.c_env.feature_spec()["type_id"]["id"]

        obs, _ = env.reset()

        # Get action indices
        move_east = action_index(env, "move", Orientation.EAST)

        # Verify initial position - agent should be at center of observation
        agent_tokens = helper.find_tokens(obs[0], location=(3, 3))
        assert len(agent_tokens) > 0, "Agent should see itself at center (3,3)"

        # The altar at grid position (row=5, col=7) should not be visible initially
        # Agent at (row=2, col=2) with 7x7 window sees:
        # - rows from (2-3) to (2+3) = -1 to 5 ✓ (altar at row 5 is at edge)
        # - cols from (2-3) to (2+3) = -1 to 5 ✗ (altar at col 7 is outside)
        altar_tokens = helper.find_tokens(obs[0], feature_id=type_id_feature_id, value=TokenTypes.ALTAR_TYPE_ID)
        altar_visible = len(altar_tokens) > 0
        assert not altar_visible, "Altar should not be visible initially"

        print("\nInitial state: Agent at (2,2), altar at (5,7) - not visible")

        # Move right (East) 3 steps
        for step in range(3):
            actions = np.array([move_east], dtype=dtype_actions)
            obs, _, _, _, _ = env.step(actions)

            # Calculate agent position after this step
            agent_col = 2 + step + 1  # Started at col 2, moved (step+1) times

            # Use helper to check if altar is actually visible
            altar_tokens = helper.find_tokens(obs[0], feature_id=type_id_feature_id, value=TokenTypes.ALTAR_TYPE_ID)
            altar_visible = len(altar_tokens) > 0

            # The altar becomes visible when agent reaches column 4 (after step 1)
            # At col 4: window covers cols 1-7, altar at col 7 is just visible
            if step >= 1:
                assert altar_visible, f"Altar should be visible after step {step} (agent at col {agent_col})"

                # Find altar in observation
                altar_tokens = helper.find_tokens(obs[0], feature_id=type_id_feature_id, value=TokenTypes.ALTAR_TYPE_ID)
                altar_positions = helper.get_positions_from_tokens(altar_tokens)
                assert len(altar_positions) == 1, "Should find exactly one altar"

                obs_col, obs_row = altar_positions[0]

                # Calculate expected observation position
                # Altar at grid (5,7), agent at grid (2,agent_col)
                # Relative position: (5-2, 7-agent_col) = (3, 7-agent_col)
                # In observation: relative + center = (7-agent_col+3, 3+3)
                expected_col = 7 - agent_col + 3
                expected_row = 3 + 3

                print(f"\nStep {step}: Agent at (2,{agent_col}), altar visible at obs ({obs_col},{obs_row})")
                assert obs_col == expected_col and obs_row == expected_row, (
                    f"Altar should be at obs ({expected_col},{expected_row}), found at ({obs_col},{obs_row})"
                )
            else:
                assert not altar_visible, f"Altar should not be visible yet at step {step}"
                print(f"\nStep {step}: Agent at (2,{agent_col}), altar not yet visible")

        # Continue moving right until altar leaves view
        for step in range(3, 9):
            actions = np.array([move_east], dtype=dtype_actions)
            obs, _, _, _, _ = env.step(actions)

            agent_col = 2 + step + 1

            # Check if altar is visible
            altar_tokens = helper.find_tokens(obs[0], feature_id=type_id_feature_id, value=TokenTypes.ALTAR_TYPE_ID)
            altar_found = len(altar_tokens) > 0

            if altar_found:
                altar_positions = helper.get_positions_from_tokens(altar_tokens)
                if altar_positions:
                    obs_col, obs_row = altar_positions[0]
                    print(f"\nStep {step}: Agent at (2,{agent_col}), altar at obs ({obs_col},{obs_row})")

            # Altar should leave view when agent reaches column 11 (after step 8)
            # At col 11: window covers cols 8-14, altar at col 7 is no longer visible
            if step < 8:
                assert altar_found, f"Altar should still be visible at step {step}"
            else:
                assert not altar_found, f"Altar should have left the view at step {step}"
                print(f"\nStep {step}: Agent at (2,{agent_col}), altar no longer visible")

        # Now walk to bottom-right corner
        # Move right to x=13
        for _ in range(5):
            actions = np.array([move_east], dtype=dtype_actions)
            obs, _, _, _, _ = env.step(actions)

        # Move down to y=8 using move (direction 4 = South)
        move_south = action_index(env, "move", Orientation.SOUTH)
        for _ in range(6):
            actions = np.array([move_south], dtype=dtype_actions)
            obs, _, _, _, _ = env.step(actions)

        # Verify agent is still at center of observation
        agent_tokens = helper.find_tokens(obs[0], location=(3, 3))
        assert len(agent_tokens) > 0, "Agent should still see itself at center (3,3)"

        # Check walls at edges of observation
        # Agent is now at (8, 13) in grid
        # Right wall at grid x=14 appears at obs x=(14-13+3)=4
        for obs_y in range(7):
            grid_y = 8 + obs_y - 3  # Convert obs y to grid y
            if 0 <= grid_y <= 9:  # Within grid bounds
                wall_tokens = helper.find_tokens(
                    obs[0], location=(4, obs_y), feature_id=type_id_feature_id, value=TokenTypes.WALL_TYPE_ID
                )
                assert len(wall_tokens) == 1, f"Should see right wall at obs (4, {obs_y})"

        # Bottom wall at grid y=9 appears at obs y=(9-8+3)=4
        for obs_x in range(7):
            grid_x = 13 + obs_x - 3  # Convert obs x to grid x
            if 0 <= grid_x <= 14:  # Within grid bounds
                wall_tokens = helper.find_tokens(
                    obs[0], location=(obs_x, 4), feature_id=type_id_feature_id, value=TokenTypes.WALL_TYPE_ID
                )
                assert len(wall_tokens) == 1, f"Should see bottom wall at obs ({obs_x}, 4)"

        # Verify padding areas (beyond walls) have no feature tokens
        # Areas beyond x=4 and y=4 should be empty
        for x in range(5, 7):
            for y in range(7):
                tokens = helper.find_tokens(obs[0], location=(x, y))
                # Check tokens beyond the first few (which might be global tokens)
                for i, token in enumerate(tokens):
                    if i >= 4:  # Skip potential global tokens
                        assert np.array_equal(token, TokenTypes.EMPTY_TOKEN), f"Expected empty token at obs ({x},{y})"

        print("\nSUCCESS: Watched altar move through field of view and verified edge behavior")
