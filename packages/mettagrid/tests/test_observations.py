import numpy as np
import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    ChangeGlyphActionConfig,
    GameConfig,
    GlobalObsConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.mettagrid_c import PackedCoordinate
from mettagrid.simulator import Simulation
from mettagrid.test_support import ObservationHelper, TokenTypes

NUM_OBS_TOKENS = 50


@pytest.fixture
def basic_env() -> Simulation:
    """Create a basic test environment."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            obs=ObsConfig(width=3, height=3, num_tokens=NUM_OBS_TOKENS),
            max_steps=1000,
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
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

    return Simulation(cfg)


@pytest.fixture
def adjacent_agents_env() -> Simulation:
    """Create an environment with adjacent agents."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            obs=ObsConfig(width=3, height=3, num_tokens=NUM_OBS_TOKENS),
            max_steps=1000,
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
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

    return Simulation(cfg)


class TestObservations:
    """Test observation functionality."""

    def test_observation_structure(self, basic_env):
        """Test basic observation structure."""
        obs = basic_env._c_sim.observations()

        # global token is always at the center of the observation window
        global_token_location = PackedCoordinate.pack(
            basic_env.config.game.obs.height // 2, basic_env.config.game.obs.width // 2
        )

        # Test global tokens (first 4 tokens)
        for agent_idx in range(basic_env.num_agents):
            for token_idx in range(4):
                assert obs[agent_idx, token_idx, 0] == global_token_location

        # Test empty terminator
        assert (obs[0, -1, :] == TokenTypes.EMPTY_TOKEN).all()
        assert (obs[1, -1, :] == TokenTypes.EMPTY_TOKEN).all()

    def test_detailed_wall_observations(self, basic_env):
        """Test detailed wall observations for both agents."""
        obs = basic_env._c_sim.observations()
        type_id_feature_id = basic_env._c_sim.feature_spec()["type_id"]["id"]
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
        obs = adjacent_agents_env._c_sim.observations()
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
        obs = basic_env._c_sim.observations()

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
        obs = basic_env._c_sim.observations()
        episode_completion_pct_feature_id = basic_env._c_sim.feature_spec()["episode_completion_pct"]["id"]
        last_action_feature_id = basic_env._c_sim.feature_spec()["last_action"]["id"]
        last_reward_feature_id = basic_env._c_sim.feature_spec()["last_reward"]["id"]
        helper = ObservationHelper()

        # Global tokens are at the center of the observation window
        global_x = basic_env.config.game.obs.width // 2
        global_y = basic_env.config.game.obs.height // 2

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
                obs=ObsConfig(width=3, height=3, num_tokens=NUM_OBS_TOKENS),
                max_steps=10,  # Important: 10 steps total so 1 step = 10%
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
                objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
                global_obs=GlobalObsConfig(episode_completion_pct=True, last_action=True, last_reward=True),
                resource_names=["laser", "armor", "heart"],
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist(), char_to_name_map=DEFAULT_CHAR_TO_NAME),
            )
        )
        env = Simulation(cfg)
        episode_completion_pct_feature_id = env._c_sim.feature_spec()["episode_completion_pct"]["id"]
        last_action_feature_id = env._c_sim.feature_spec()["last_action"]["id"]
        obs = env._c_sim.observations()
        helper = ObservationHelper()

        # Global tokens are at the center of the observation window
        global_x = env.config.game.obs.width // 2
        global_y = env.config.game.obs.height // 2

        # Take a noop action
        for agent_id in range(env.num_agents):
            env.agent(agent_id).set_action("noop")
        env.step()
        obs = env._c_sim.observations()

        # Check episode completion updated (1/10 = 10%)
        expected_completion = int(round(0.1 * 255))
        completion_values = helper.find_token_values(
            obs[0], location=(global_x, global_y), feature_id=episode_completion_pct_feature_id
        )
        assert completion_values == [expected_completion], (
            f"Expected completion {expected_completion}, got {completion_values}"
        )

        # Check last action - verify it's the noop action
        last_action = helper.find_token_values(obs[0], location=(global_x, global_y), feature_id=last_action_feature_id)
        assert last_action == env.action_names.index("noop"), f"Expected noop action, got {last_action}"

        # Take a move action
        for agent_id in range(env.num_agents):
            env.agent(agent_id).set_action("move_south")
        env.step()
        obs = env._c_sim.observations()

        # Check updates
        expected_completion = int(round(0.2 * 255))
        completion_value = helper.find_token_values(
            obs[0], location=(global_x, global_y), feature_id=episode_completion_pct_feature_id
        )
        assert completion_value == expected_completion

        last_action = helper.find_token_values(obs[0], location=(global_x, global_y), feature_id=last_action_feature_id)
        assert last_action == env.action_names.index("move_south"), f"Expected move_south action, got {last_action}"

    @pytest.mark.skip(reason="Requires direct C++ buffer access for detailed observation validation")
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
                obs=ObsConfig(width=3, height=3, num_tokens=NUM_OBS_TOKENS),
                max_steps=10,
                # Use 3x3 observation window
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(),
                    change_glyph=ChangeGlyphActionConfig(enabled=True, number_of_glyphs=8),
                ),
                objects={"wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID)},
                resource_names=["laser", "armor"],
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist(), char_to_name_map=DEFAULT_CHAR_TO_NAME),
            )
        )
        sim = Simulation(cfg)
        glyph_feature_id = sim._c_sim.feature_spec()["agent:glyph"]["id"]

        obs = sim._c_sim.observations()

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
        # Get available change_glyph actions
        glyph_actions = [name for name in sim.action_names if name.startswith("change_glyph_")]
        assert len(glyph_actions) >= 6, f"Should have at least 6 glyph actions, got {glyph_actions}"

        # Pick specific glyph actions (use actual action names from list)
        glyph_action_3 = glyph_actions[3] if len(glyph_actions) > 3 else glyph_actions[0]
        glyph_action_5 = glyph_actions[5] if len(glyph_actions) > 5 else glyph_actions[1]

        # Test 1: Agent 0 changes to glyph 3, Agent 1 changes to glyph 5
        sim.agent(0).set_action(glyph_action_3)
        sim.agent(1).set_action(glyph_action_5)
        sim.step()
        obs = sim._c_sim.observations()

        agent0_self_glyph = helper.find_token_values(obs[0], location=(1, 1), feature_id=glyph_feature_id)
        # Agent 0 should now have a non-zero glyph (glyph was changed)
        assert len(agent0_self_glyph) > 0, "Agent 0 should have a glyph token after changing glyph"
        assert agent0_self_glyph != 0, f"Agent 0 glyph should not be 0 (default), got {agent0_self_glyph}"

        agent1_sees_agent0_glyph = helper.find_token_values(obs[1], location=(0, 1), feature_id=glyph_feature_id)
        assert len(agent1_sees_agent0_glyph) > 0, "Agent 1 should see Agent 0's glyph"
        assert agent1_sees_agent0_glyph == agent0_self_glyph, "Agent 1 should see the same glyph as Agent 0 has"

        agent1_self_glyph = helper.find_token_values(obs[1], location=(1, 1), feature_id=glyph_feature_id)
        assert len(agent1_self_glyph) > 0, "Agent 1 should have a glyph token after changing glyph"
        assert agent1_self_glyph != agent0_self_glyph, "Agent 1 should have different glyph than Agent 0"

        # Test 2: Invalid glyph values (should be no-op)
        assert "change_glyph_invalid" not in sim.action_names, "Invalid glyph action should not exist"

        # Test 3: Changing back to glyph 0 removes the token

        # Change back to glyph 0 (first glyph action)
        glyph_action_0 = glyph_actions[0]
        sim.agent(0).set_action(glyph_action_0)
        sim.agent(1).set_action(glyph_action_0)
        sim.step()
        obs = sim._c_sim.observations()

        # Verify glyph tokens are gone
        agent0_glyph = helper.find_token_values(obs[0], location=(1, 1), feature_id=glyph_feature_id)
        agent1_glyph = helper.find_token_values(obs[1], location=(1, 1), feature_id=glyph_feature_id)

        assert len(agent0_glyph) == 0, f"Agent 0 changed to glyph 0 should have no token, got {agent0_glyph}"
        assert len(agent1_glyph) == 0, f"Agent 1 changed to glyph 0 should have no token, got {agent1_glyph}"


class TestEdgeObservations:
    """Test observation behavior near world edges."""

    @pytest.mark.skip(reason="Requires direct C++ buffer access and detailed observation position validation")
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
                obs=ObsConfig(width=7, height=7, num_tokens=NUM_OBS_TOKENS),
                max_steps=50,  # Enough steps to walk around
<<<<<<< HEAD
                obs_width=7,
                obs_height=7,
                num_observation_tokens=NUM_OBS_TOKENS,
                actions=ActionsConfig(
                    noop=ActionConfig(),
                    move=ActionConfig(),
                ),
=======
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
>>>>>>> 4b8de25bb4 (feat: implement mettagrid API changes)
                objects={
                    "wall": WallConfig(type_id=TokenTypes.WALL_TYPE_ID),
                },
                resource_names=["laser", "resource1", "resource2"],  # laser required for attack action
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist(), char_to_name_map=DEFAULT_CHAR_TO_NAME),
            )
        )
        sim = Simulation(cfg)
        type_id_feature_id = sim._c_sim.feature_spec()["type_id"]["id"]

        obs = sim._c_sim.observations()

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
            sim.agent(0).set_action("move_east")
            sim.step()
            obs = sim._c_sim.observations()

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
            sim.agent(0).set_action("move_east")
            sim.step()
            obs = sim._c_sim.observations()

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
            sim.agent(0).set_action("move_east")
            sim.step()
            obs = sim._c_sim.observations()

        # Move down to y=8 using move (direction 4 = South)
        for _ in range(6):
            sim.agent(0).set_action("move_south")
            sim.step()
            obs = sim._c_sim.observations()

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
