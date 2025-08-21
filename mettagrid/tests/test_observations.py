import numpy as np
import pytest

from metta.mettagrid.core import MettaGridCore
from metta.mettagrid.map_builder.ascii import AsciiMapBuilder
from metta.mettagrid.map_builder.utils import create_grid
from metta.mettagrid.mettagrid_c import PackedCoordinate, dtype_actions
from metta.mettagrid.mettagrid_config import (
    ActionConfig,
    ActionsConfig,
    ChangeGlyphActionConfig,
    ConverterConfig,
    EnvConfig,
    GameConfig,
    GlobalObsConfig,
    GroupConfig,
    WallConfig,
)
from metta.mettagrid.test_support import ObservationHelper, TokenTypes

NUM_OBS_TOKENS = 50


@pytest.fixture
def basic_env() -> MettaGridCore:
    """Create a basic test environment."""
    cfg = EnvConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=1000,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=NUM_OBS_TOKENS,
            actions=ActionsConfig(
                noop=ActionConfig(),
                move_8way=ActionConfig(),
                rotate=ActionConfig(),
                get_items=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
            inventory_item_names=["laser", "armor", "heart"],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#", "#", "#", "#"],
                    ["#", "@", ".", ".", ".", ".", ".", "#"],
                    ["#", ".", ".", ".", "@", ".", ".", "#"],
                    ["#", "#", "#", "#", "#", "#", "#", "#"],
                ]
            ),
        )
    )

    return MettaGridCore(cfg)


@pytest.fixture
def adjacent_agents_env() -> MettaGridCore:
    """Create an environment with adjacent agents."""
    cfg = EnvConfig(
        game=GameConfig(
            num_agents=2,
            max_steps=1000,
            obs_width=3,
            obs_height=3,
            num_observation_tokens=NUM_OBS_TOKENS,
            actions=ActionsConfig(
                noop=ActionConfig(),
                move_8way=ActionConfig(),
                rotate=ActionConfig(),
                get_items=ActionConfig(),
            ),
            objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
            inventory_item_names=["laser", "armor", "heart"],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "@", "@", ".", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ]
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

        all_positions = {(x, y) for x in range(3) for y in range(3)}
        no_wall_positions_agent0 = all_positions - set(wall_positions_agent0)

        # Check expected wall positions
        for x, y in wall_positions_agent0:
            assert helper.has_wall_at(agent0_obs, x, y), f"Agent 0: Expected wall at ({x}, {y})"

        # Check no walls at empty positions
        for x, y in no_wall_positions_agent0:
            assert not helper.has_wall_at(agent0_obs, x, y), f"Agent 0: Expected no wall at ({x}, {y})"

        # Verify wall count
        assert helper.count_walls(agent0_obs) == 5, "Agent 0 should see exactly 5 walls"

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

        no_wall_positions_agent1 = all_positions - set(wall_positions_agent1)

        # Check expected wall positions
        for x, y in wall_positions_agent1:
            assert helper.has_wall_at(agent1_obs, x, y), f"Agent 1: Expected wall at ({x}, {y})"

        # Check no walls at empty positions
        for x, y in no_wall_positions_agent1:
            assert not helper.has_wall_at(agent1_obs, x, y), f"Agent 1: Expected no wall at ({x}, {y})"

        # Verify wall count
        assert helper.count_walls(agent1_obs) == 3, "Agent 1 should see exactly 3 walls"

    def test_agent_surrounded_by_altars(self):
        """Test agent observation when surrounded by colored altars."""
        # Create a 5x5 environment with agent in center surrounded by altars
        game_map = create_grid(5, 5, fill_value=".")
        helper = ObservationHelper()

        # Add walls around perimeter
        game_map[0, :] = "#"
        game_map[-1, :] = "#"
        game_map[:, 0] = "#"
        game_map[:, -1] = "#"

        # Place agent in center at grid position (2,2)
        game_map[2, 2] = "@"

        # Place 8 altars around the agent with different colors
        # Layout:
        #   0 1 2 3 4  (x)
        # 0 W W W W W
        # 1 W A B C W    A-H are altars with colors 1-8
        # 2 W D & E W    & is the agent
        # 3 W F G H W
        # 4 W W W W W
        # (y)

        altar_positions = [
            (1, 1, 1),  # A: top-left, color 1
            (2, 1, 2),  # B: top-center, color 2
            (3, 1, 3),  # C: top-right, color 3
            (1, 2, 4),  # D: middle-left, color 4
            (3, 2, 5),  # E: middle-right, color 5
            (1, 3, 6),  # F: bottom-left, color 6
            (2, 3, 7),  # G: bottom-center, color 7
            (3, 3, 8),  # H: bottom-right, color 8
        ]

        # Place altars on the map
        for x, y, _color in altar_positions:
            game_map[y, x] = "_"  # Use underscore character for altars

        # Create altar objects configuration - single altar type for simplicity
        objects = {
            "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID),
            "altar": ConverterConfig(
                type_id=TokenTypes.ALTAR_TYPE_ID,
                input_resources={"resource1": 1},
                output_resources={"resource2": 1},
                max_output=10,
                conversion_ticks=5,
                cooldown=3,
                initial_resource_count=0,
                color=42,  # Single color for all altars
            ),
        }

        # Create the environment using direct EnvConfig
        cfg = EnvConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=10,
                obs_width=3,  # Use 3x3 observation window for this test
                obs_height=3,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move_8way=ActionConfig(),
                    rotate=ActionConfig(),
                    get_items=ActionConfig(),
                ),
                objects=objects,
                groups={"agent": GroupConfig(id=0)},  # "@" maps to "agent.agent"
                inventory_item_names=["laser", "resource1", "resource2"],  # include laser to allow attack
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist()),
            )
        )

        env = MettaGridCore(cfg)

        obs, _ = env.reset()
        agent_obs = obs[0]

        # The agent at (2,2) should see all 8 altars in its 3x3 observation window
        # Expected altar positions in observation coordinates:
        #   A B C     (0,0) (1,0) (2,0)
        #   D & E  -> (0,1) (1,1) (2,1)
        #   F G H     (0,2) (1,2) (2,2)

        expected_altar_positions = [
            (0, 0),  # A: top-left
            (1, 0),  # B: top-center
            (2, 0),  # C: top-right
            (0, 1),  # D: middle-left
            (2, 1),  # E: middle-right (center 1,1 is agent)
            (0, 2),  # F: bottom-left
            (1, 2),  # G: bottom-center
            (2, 2),  # H: bottom-right
        ]

        # Check that we see altars at all expected positions
        for x, y in expected_altar_positions:
            # Check altar exists at location
            assert helper.has_feature_at(agent_obs, x, y, TokenTypes.ALTAR_TYPE_ID), f"Should have altar at ({x}, {y})"

            # Check color token
            color_value = helper.find_token_value_at_location(agent_obs, x, y, TokenTypes.COLOR)
            assert color_value == 42, f"Altar at ({x}, {y}) should have color 42, got {color_value}"

            # Check converter status token exists
            converter_value = helper.find_token_value_at_location(
                agent_obs, x, y, TokenTypes.CONVERTING_OR_COOLING_DOWN
            )
            assert converter_value is not None, f"Should have converter status token at ({x}, {y})"

        # Verify the agent sees itself at center (1,1)
        agent_tokens = helper.find_tokens_at_location(agent_obs, 1, 1)
        assert len(agent_tokens) > 0, "Agent should see itself at center position"

        # Count total altars
        altar_count = helper.count_features_by_type(agent_obs, TokenTypes.ALTAR_TYPE_ID)
        assert altar_count == 8, f"Should see 8 altars, got {altar_count}"

    def _check_token_exists(self, obs, x, y, type_id, feature_id, agent_name):
        """Helper to check if a specific token exists at a location."""
        helper = ObservationHelper()
        tokens = helper.find_tokens_at_location(obs, x, y)
        for token in tokens:
            if token[1] == type_id and token[2] == feature_id:
                return
        raise AssertionError(f"{agent_name}: Expected token with type {type_id} and feature {feature_id} at ({x}, {y})")

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

        agent1_tokens = helper.find_tokens_at_location(obs[0], 2, 1)
        assert len(agent1_tokens) > 0, "Agent 0 should see Agent 1 at (2,1)"

        # Agent 1 at (2,2) has observation window centered at (2,2)
        # Its 3x3 window covers grid positions (1,1) to (3,3)
        # Agent 0 at (1,2) is within this window

        # In Agent 1's relative coordinates:
        # Agent 1 is at center (1,1)
        # Agent 0 at grid (1,2) - Agent 1 at grid (2,2) = offset (-1,0)
        # So Agent 0 should appear at observation position (1-1, 1+0) = (0,1)

        agent0_tokens = helper.find_tokens_at_location(obs[1], 0, 1)
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
        helper = ObservationHelper()

        # Global tokens are at the center of the observation window
        global_x = basic_env.obs_width // 2
        global_y = basic_env.obs_height // 2

        # Check token types and values
        assert helper.find_token_value_at_location(obs[0], global_x, global_y, TokenTypes.EPISODE_COMPLETION_PCT) == 0
        assert helper.find_token_value_at_location(obs[0], global_x, global_y, TokenTypes.LAST_ACTION) == 0
        assert helper.find_token_value_at_location(obs[0], global_x, global_y, TokenTypes.LAST_ACTION_ARG) == 0
        assert helper.find_token_value_at_location(obs[0], global_x, global_y, TokenTypes.LAST_REWARD) == 0

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
        cfg = EnvConfig(
            game=GameConfig(
                num_agents=2,
                max_steps=10,  # Important: 10 steps total so 1 step = 10%
                obs_width=3,
                obs_height=3,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move_8way=ActionConfig(),
                    rotate=ActionConfig(),
                    get_items=ActionConfig(),
                ),
                objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
                global_obs=GlobalObsConfig(
                    episode_completion_pct=True,
                    last_action=True,
                    last_reward=True,
                    resource_rewards=False,
                ),
                inventory_item_names=["laser", "armor", "heart"],
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist()),
            )
        )
        env = MettaGridCore(cfg)
        obs, _ = env.reset()
        num_agents = env.num_agents
        helper = ObservationHelper()

        # Global tokens are at the center of the observation window
        global_x = env.obs_width // 2
        global_y = env.obs_height // 2

        # Take a noop action
        noop_idx = env.action_names.index("noop")
        actions = np.full((num_agents, 2), [noop_idx, 0], dtype=dtype_actions)
        obs, _, _, _, _ = env.step(actions)

        # Check episode completion updated (1/10 = 10%)
        expected_completion = int(round(0.1 * 255))
        completion_value = helper.find_token_value_at_location(
            obs[0], global_x, global_y, TokenTypes.EPISODE_COMPLETION_PCT
        )
        assert completion_value == expected_completion, (
            f"Expected completion {expected_completion}, got {completion_value}"
        )

        # Check last action
        last_action = helper.find_token_value_at_location(obs[0], global_x, global_y, TokenTypes.LAST_ACTION)
        assert last_action == noop_idx, f"Expected last action {noop_idx}, got {last_action}"

        # Check last action arg
        last_arg = helper.find_token_value_at_location(obs[0], global_x, global_y, TokenTypes.LAST_ACTION_ARG)
        assert last_arg == 0, f"Expected last action arg 0, got {last_arg}"

        # Take a move_8way action
        move_idx = env.action_names.index("move_8way")
        actions = np.full((num_agents, 2), [move_idx, 1], dtype=dtype_actions)
        obs, _, _, _, _ = env.step(actions)

        # Check updates
        expected_completion = int(round(0.2 * 255))
        completion_value = helper.find_token_value_at_location(
            obs[0], global_x, global_y, TokenTypes.EPISODE_COMPLETION_PCT
        )
        assert completion_value == expected_completion

        last_action = helper.find_token_value_at_location(obs[0], global_x, global_y, TokenTypes.LAST_ACTION)
        assert last_action == move_idx

        last_arg = helper.find_token_value_at_location(obs[0], global_x, global_y, TokenTypes.LAST_ACTION_ARG)
        assert last_arg == 1

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
        cfg = EnvConfig(
            game=GameConfig(
                num_agents=2,
                max_steps=10,
                obs_width=3,  # Use 3x3 observation window
                obs_height=3,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move_8way=ActionConfig(),
                    rotate=ActionConfig(),
                    get_items=ActionConfig(),
                    change_glyph=ChangeGlyphActionConfig(enabled=True, number_of_glyphs=8),
                ),
                objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
                groups={
                    "agent": GroupConfig(id=0),  # "@" maps to "agent.agent" for both agents
                },
                inventory_item_names=["laser", "armor"],
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist()),
            )
        )
        env = MettaGridCore(cfg)

        obs, _ = env.reset()

        print("\n=== Testing Initial Glyph Values ===")

        # Check if we're seeing uninitialized memory issues
        agent0_self_glyph = helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH)
        agent0_sees_agent1_glyph = helper.find_token_value_at_location(obs[0], 2, 1, TokenTypes.GLYPH)
        agent1_self_glyph = helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH)
        agent1_sees_agent0_glyph = helper.find_token_value_at_location(obs[1], 0, 1, TokenTypes.GLYPH)

        # If we see value 231 or other unexpected values, it's likely uninitialized memory
        unexpected_values = [agent0_self_glyph, agent0_sees_agent1_glyph, agent1_self_glyph, agent1_sees_agent0_glyph]
        unexpected_values = [v for v in unexpected_values if v is not None and v > 7]

        if unexpected_values:
            print(f"WARNING: Found unexpected glyph values: {unexpected_values}")
            print("This appears to be uninitialized memory. Attempting workaround...")

            # Workaround: explicitly set glyphs to 0 and step once
            change_glyph_idx = env.action_names.index("change_glyph")
            actions = np.array(
                [
                    [change_glyph_idx, 0],  # Agent 0 to glyph 0
                    [change_glyph_idx, 0],  # Agent 1 to glyph 0
                ],
                dtype=dtype_actions,
            )
            obs, _, _, _, _ = env.step(actions)

            # Re-check
            agent0_self_glyph = helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH)
            agent0_sees_agent1_glyph = helper.find_token_value_at_location(obs[0], 2, 1, TokenTypes.GLYPH)
            agent1_self_glyph = helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH)
            agent1_sees_agent0_glyph = helper.find_token_value_at_location(obs[1], 0, 1, TokenTypes.GLYPH)

        # Initially, both agents should have glyph 0 (default)
        # Since glyph 0 is suppressed, we should NOT find any glyph tokens
        assert agent0_self_glyph is None, f"Agent 0 with glyph 0 should have no glyph token, got {agent0_self_glyph}"
        assert agent0_sees_agent1_glyph is None, (
            f"Agent 0 should see Agent 1 with no glyph token (glyph 0), got {agent0_sees_agent1_glyph}"
        )
        assert agent1_self_glyph is None, f"Agent 1 with glyph 0 should have no glyph token, got {agent1_self_glyph}"
        assert agent1_sees_agent0_glyph is None, (
            f"Agent 1 should see Agent 0 with no glyph token (glyph 0), got {agent1_sees_agent0_glyph}"
        )

        print("✓ Both agents start with glyph 0 (no glyph token)")

        # Test changing glyphs
        print("\n=== Testing Glyph Changes ===")

        change_glyph_idx = env.action_names.index("change_glyph")
        noop_idx = env.action_names.index("noop")

        # Test 1: Agent 0 changes to glyph 3, Agent 1 stays at 0
        actions = np.array(
            [
                [change_glyph_idx, 3],  # Agent 0 changes to glyph 3
                [noop_idx, 0],  # Agent 1 does nothing
            ],
            dtype=dtype_actions,
        )

        obs, _, _, _, _ = env.step(actions)

        agent0_self_glyph = helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH)
        assert agent0_self_glyph == 3, f"Agent 0 should have glyph 3, got {agent0_self_glyph}"

        agent1_sees_agent0_glyph = helper.find_token_value_at_location(obs[1], 0, 1, TokenTypes.GLYPH)
        assert agent1_sees_agent0_glyph == 3, f"Agent 1 should see Agent 0 with glyph 3, got {agent1_sees_agent0_glyph}"

        agent1_self_glyph = helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH)
        assert agent1_self_glyph is None, f"Agent 1 should still have no glyph token (glyph 0), got {agent1_self_glyph}"

        print("✓ Agent 0 successfully changed to glyph 3")

        # Test 2: Both agents change glyphs simultaneously
        actions = np.array(
            [
                [change_glyph_idx, 5],  # Agent 0 changes to glyph 5
                [change_glyph_idx, 7],  # Agent 1 changes to glyph 7
            ],
            dtype=dtype_actions,
        )

        obs, _, _, _, _ = env.step(actions)

        agent0_self_glyph = helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH)
        agent1_self_glyph = helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH)

        assert agent0_self_glyph == 5, f"Agent 0 should have glyph 5, got {agent0_self_glyph}"
        assert agent1_self_glyph == 7, f"Agent 1 should have glyph 7, got {agent1_self_glyph}"

        # Verify they see each other's new glyphs
        agent0_sees_agent1 = helper.find_token_value_at_location(obs[0], 2, 1, TokenTypes.GLYPH)
        agent1_sees_agent0 = helper.find_token_value_at_location(obs[1], 0, 1, TokenTypes.GLYPH)

        assert agent0_sees_agent1 == 7, f"Agent 0 should see Agent 1 with glyph 7, got {agent0_sees_agent1}"
        assert agent1_sees_agent0 == 5, f"Agent 1 should see Agent 0 with glyph 5, got {agent1_sees_agent0}"

        print("✓ Both agents successfully changed glyphs simultaneously")

        # Test 3: Test all valid glyph values (0-7)
        print("\n=== Testing All Glyph Values ===")

        for glyph in range(8):
            actions = np.array(
                [
                    [change_glyph_idx, glyph],
                    [change_glyph_idx, (glyph + 4) % 8],  # Different glyph for agent 1
                ],
                dtype=dtype_actions,
            )

            obs, _, _, _, _ = env.step(actions)

            agent0_glyph = helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH)
            agent1_glyph = helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH)
            expected_agent1_glyph = (glyph + 4) % 8

            # Glyph 0 should not produce a token
            if glyph == 0:
                assert agent0_glyph is None, f"Agent 0 with glyph 0 should have no token, got {agent0_glyph}"
            else:
                assert agent0_glyph == glyph, f"Agent 0 should have glyph {glyph}, got {agent0_glyph}"

            if expected_agent1_glyph == 0:
                assert agent1_glyph is None, f"Agent 1 with glyph 0 should have no token, got {agent1_glyph}"
            else:
                assert agent1_glyph == expected_agent1_glyph, (
                    f"Agent 1 should have glyph {expected_agent1_glyph}, got {agent1_glyph}"
                )

            # Verify cross-visibility
            agent0_sees_agent1 = helper.find_token_value_at_location(obs[0], 2, 1, TokenTypes.GLYPH)
            agent1_sees_agent0 = helper.find_token_value_at_location(obs[1], 0, 1, TokenTypes.GLYPH)

            if expected_agent1_glyph == 0:
                assert agent0_sees_agent1 is None, (
                    f"Agent 0 should see Agent 1 with no glyph token (glyph 0), got {agent0_sees_agent1}"
                )
            else:
                assert agent0_sees_agent1 == expected_agent1_glyph, (
                    f"Agent 0 should see Agent 1 with glyph {expected_agent1_glyph}, got {agent0_sees_agent1}"
                )

            if glyph == 0:
                assert agent1_sees_agent0 is None, (
                    f"Agent 1 should see Agent 0 with no glyph token (glyph 0), got {agent1_sees_agent0}"
                )
            else:
                assert agent1_sees_agent0 == glyph, (
                    f"Agent 1 should see Agent 0 with glyph {glyph}, got {agent1_sees_agent0}"
                )

        print("✓ All glyph values (0-7) work correctly with glyph 0 suppressed")

        # Test 4: Invalid glyph values (should be no-op)
        print("\n=== Testing Invalid Glyph Values ===")

        # First set agents to known glyph values
        actions = np.array(
            [
                [change_glyph_idx, 3],  # Agent 0 to glyph 3
                [change_glyph_idx, 5],  # Agent 1 to glyph 5
            ],
            dtype=dtype_actions,
        )
        obs, _, _, _, _ = env.step(actions)

        # Verify initial glyphs
        assert helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH) == 3
        assert helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH) == 5

        test_cases = [
            (8, "8 should be no-op"),
            (10, "10 should be no-op"),
            (255, "255 should be no-op"),
            (100, "100 should be no-op"),
        ]

        for invalid_value, description in test_cases:
            actions = np.array(
                [
                    [change_glyph_idx, invalid_value],  # Agent 0 tries invalid glyph
                    [noop_idx, 0],  # Agent 1 does nothing
                ],
                dtype=dtype_actions,
            )

            obs, _, _, _, _ = env.step(actions)

            agent0_glyph = helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH)
            agent1_glyph = helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH)

            # Glyphs should remain unchanged
            assert agent0_glyph == 3, f"{description}: Agent 0 glyph should stay 3, got {agent0_glyph}"
            assert agent1_glyph == 5, f"Agent 1 glyph should stay 5, got {agent1_glyph}"

        print("✓ Invalid glyph values correctly ignored (no-op)")

        # Test 5: Glyph persistence (glyphs should stay until changed)
        print("\n=== Testing Glyph Persistence ===")

        # Set distinct glyphs
        actions = np.array(
            [
                [change_glyph_idx, 2],
                [change_glyph_idx, 6],
            ],
            dtype=dtype_actions,
        )
        obs, _, _, _, _ = env.step(actions)

        # Take several noop actions - glyphs should persist
        for i in range(3):
            actions = np.array(
                [
                    [noop_idx, 0],
                    [noop_idx, 0],
                ],
                dtype=dtype_actions,
            )
            obs, _, _, _, _ = env.step(actions)

            agent0_glyph = helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH)
            agent1_glyph = helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH)

            assert agent0_glyph == 2, f"Agent 0 glyph should persist as 2 after {i + 1} steps, got {agent0_glyph}"
            assert agent1_glyph == 6, f"Agent 1 glyph should persist as 6 after {i + 1} steps, got {agent1_glyph}"

        print("✓ Glyphs persist correctly across multiple steps")

        # Test 6: Changing back to glyph 0 removes the token
        print("\n=== Testing Changing Back to Glyph 0 ===")

        # First set to non-zero glyphs
        actions = np.array(
            [
                [change_glyph_idx, 4],
                [change_glyph_idx, 5],
            ],
            dtype=dtype_actions,
        )
        obs, _, _, _, _ = env.step(actions)

        # Verify they have glyph tokens
        assert helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH) == 4
        assert helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH) == 5

        # Change back to glyph 0
        actions = np.array(
            [
                [change_glyph_idx, 0],
                [change_glyph_idx, 0],
            ],
            dtype=dtype_actions,
        )
        obs, _, _, _, _ = env.step(actions)

        # Verify glyph tokens are gone
        agent0_glyph = helper.find_token_value_at_location(obs[0], 1, 1, TokenTypes.GLYPH)
        agent1_glyph = helper.find_token_value_at_location(obs[1], 1, 1, TokenTypes.GLYPH)

        assert agent0_glyph is None, f"Agent 0 changed to glyph 0 should have no token, got {agent0_glyph}"
        assert agent1_glyph is None, f"Agent 1 changed to glyph 0 should have no token, got {agent1_glyph}"

        print("✓ Changing back to glyph 0 correctly removes glyph tokens")

        print("\n=== All Glyph Tests Passed! ===")


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

        # Place an altar at (7, 5) - we'll watch it move through our view
        game_map[5, 7] = "_"

        # Create environment with 7x7 observation window
        cfg = EnvConfig(
            game=GameConfig(
                num_agents=1,
                max_steps=50,  # Enough steps to walk around
                obs_width=7,
                obs_height=7,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move_8way=ActionConfig(),
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
                        cooldown=3,
                        initial_resource_count=0,
                        color=42,  # Distinctive color
                    ),
                },
                groups={"agent": GroupConfig(id=0)},  # "@" maps to "agent.agent"
                inventory_item_names=["laser", "resource1", "resource2"],  # laser required for attack action
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist()),
            )
        )
        env = MettaGridCore(cfg)

        obs, _ = env.reset()

        # Get action indices
        move_idx = env.action_names.index("move_8way")

        # Verify initial position - agent should be at center of observation
        agent_tokens = helper.find_tokens_at_location(obs[0], 3, 3)
        assert len(agent_tokens) > 0, "Agent should see itself at center (3,3)"

        # The altar at grid (7,5) should not be visible initially
        # Agent at (2,2) with 7x7 window sees from (-1,-1) to (5,5)
        # So altar at (7,5) is outside the view
        altar_visible = helper.count_features_by_type(obs[0], TokenTypes.ALTAR_TYPE_ID) > 0
        assert not altar_visible, "Altar should not be visible initially"

        print("\nInitial state: Agent at (2,2), altar at (7,5) - not visible")

        # Move right 3 steps using move_8way (direction 2 = East)
        for step in range(3):
            actions = np.array([[move_idx, 2]], dtype=dtype_actions)  # 2 = East
            obs, _, _, _, _ = env.step(actions)

            # After step 0: agent at (3,2), window covers (0,0) to (6,5) - altar still not visible
            # After step 1: agent at (4,2), window covers (1,0) to (7,5) - altar just enters view!
            # After step 2: agent at (5,2), window covers (2,0) to (8,5) - altar clearly visible

            if step >= 1:  # Altar should be visible after first step
                # Find altar in observation
                altar_tokens = helper.find_features_by_type(obs[0], TokenTypes.ALTAR_TYPE_ID)
                assert len(altar_tokens) > 0, f"Altar should be visible after step {step}"

                altar_location = altar_tokens[0, 0]
                altar_coords = PackedCoordinate.unpack(altar_location)
                assert altar_coords is not None, "Should be able to unpack altar coordinates"

                obs_row, obs_col = altar_coords
                # Calculate expected position
                # Agent is at grid (3+step, 2) after step steps
                # Altar at grid (7,5)
                # Relative position: altar - agent = (7-(3+step), 5-2) = (4-step, 3)
                # In observation coords: relative + center = (4-step+3, 3+3) = (7-step, 6)
                expected_col = 7 - step
                expected_row = 6
                print(f"\nStep {step}: Agent at ({3 + step},2), altar visible at obs ({obs_col},{obs_row})")
                assert obs_col == expected_col and obs_row == expected_row, (
                    f"Altar should be at ({expected_col},{expected_row}), found at ({obs_col},{obs_row})"
                )

        # Continue moving right until altar leaves view
        for step in range(3, 6):
            actions = np.array([[move_idx, 2]], dtype=dtype_actions)  # 2 = East
            obs, _, _, _, _ = env.step(actions)

            # After step 3: agent at (6,2), altar at relative (1,3) - still visible
            # After step 4: agent at (7,2), altar at relative (0,3) - at center column
            # After step 5: agent at (8,2), altar at relative (-1,3) - at left edge

            altar_found = helper.count_features_by_type(obs[0], TokenTypes.ALTAR_TYPE_ID) > 0
            if altar_found:
                altar_tokens = helper.find_features_by_type(obs[0], TokenTypes.ALTAR_TYPE_ID)
                altar_location = altar_tokens[0, 0]
                altar_coords = PackedCoordinate.unpack(altar_location)
                if altar_coords:
                    obs_row, obs_col = altar_coords
                    expected_col = 7 - step
                    print(f"\nStep {step}: Agent at ({3 + step},2), altar at obs ({obs_col},{obs_row})")

            if step <= 5:
                assert altar_found, f"Altar should still be visible at step {step}"

        # Continue moving right until altar leaves view
        for step in range(6, 9):
            actions = np.array([[move_idx, 2]], dtype=dtype_actions)  # 2 = East
            obs, _, _, _, _ = env.step(actions)

            # After step 6: agent at (9,2), altar at relative (-2,3) - obs position (1,6)
            # After step 7: agent at (10,2), altar at relative (-3,3) - obs position (0,6) - at very edge
            # After step 8: agent at (11,2), altar at relative (-4,3) - outside 7x7 window

            altar_found = helper.count_features_by_type(obs[0], TokenTypes.ALTAR_TYPE_ID) > 0
            if altar_found:
                altar_tokens = helper.find_features_by_type(obs[0], TokenTypes.ALTAR_TYPE_ID)
                altar_location = altar_tokens[0, 0]
                altar_coords = PackedCoordinate.unpack(altar_location)
                if altar_coords:
                    obs_row, obs_col = altar_coords
                    print(f"\nStep {step}: Agent at ({3 + step},2), altar at obs ({obs_col},{obs_row})")

            if step <= 7:
                assert altar_found, f"Altar should still be visible at step {step}"
            else:
                assert not altar_found, "Altar should have left the view"
                print(f"\nStep {step}: Agent at ({3 + step},2), altar no longer visible")

        # Now walk to bottom-right corner
        # Continue right to x=13
        for _ in range(5):
            actions = np.array([[move_idx, 2]], dtype=dtype_actions)  # 2 = East
            obs, _, _, _, _ = env.step(actions)

        # Move down to y=8 using move_8way (direction 4 = South)
        for _ in range(6):
            actions = np.array([[move_idx, 4]], dtype=dtype_actions)  # 4 = South
            obs, _, _, _, _ = env.step(actions)

        # Verify agent is still at center of observation
        agent_tokens = helper.find_tokens_at_location(obs[0], 3, 3)
        assert len(agent_tokens) > 0, "Agent should still see itself at center (3,3)"

        # Check walls at edges
        # Right wall at x=14 -> obs x=4
        for obs_y in range(7):
            grid_y = 8 + obs_y - 3
            if 0 <= grid_y <= 9:
                assert helper.has_wall_at(obs[0], 4, obs_y), f"Should see right wall at obs ({4}, {obs_y})"

        # Bottom wall at y=9 -> obs y=4
        for obs_x in range(7):
            grid_x = 13 + obs_x - 3
            if 0 <= grid_x <= 14:
                assert helper.has_wall_at(obs[0], obs_x, 4), f"Should see bottom wall at obs ({obs_x}, {4})"

        # Verify padding areas have no tokens
        for x in range(5, 7):
            for y in range(7):
                tokens = helper.find_tokens_at_location(obs[0], x, y)
                for i, token in enumerate(tokens):
                    if i >= 4:
                        assert np.array_equal(token, TokenTypes.EMPTY_TOKEN), f"Expected empty token at obs ({x},{y})"

        print("\nSUCCESS: Watched altar move through field of view and verified edge behavior")
