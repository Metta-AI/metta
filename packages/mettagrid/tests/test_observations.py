import pytest

from mettagrid.config.mettagrid_config import (
    ActionsConfig,
    ChangeVibeActionConfig,
    GameConfig,
    GlobalObsConfig,
    MettaGridConfig,
    MoveActionConfig,
    NoopActionConfig,
    ObsConfig,
    WallConfig,
)
from mettagrid.config.vibes import VIBES
from mettagrid.map_builder.ascii import AsciiMapBuilder
from mettagrid.map_builder.utils import create_grid
from mettagrid.mapgen.utils.ascii_grid import DEFAULT_CHAR_TO_NAME
from mettagrid.mettagrid_c import PackedCoordinate
from mettagrid.simulator import Simulation
from mettagrid.test_support import ObservationHelper, TokenTypes

NUM_OBS_TOKENS = 50


@pytest.fixture
def basic_sim() -> Simulation:
    """Create a basic test environment."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            obs=ObsConfig(width=3, height=3, num_tokens=NUM_OBS_TOKENS),
            max_steps=1000,
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            objects={"wall": WallConfig(tags=["wall"])},
            resource_names=["laser", "armor", "heart"],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#", "#", "#", "#"],
                    ["#", "@", ".", ".", ".", ".", ".", "#"],
                    ["#", ".", ".", ".", "@", ".", ".", "#"],
                    ["#", "#", "#", "#", "#", "#", "#", "#"],
                ],
                char_to_map_name=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )

    cfg.game.global_obs.compass = True

    return Simulation(cfg)


@pytest.fixture
def adjacent_agents_sim() -> Simulation:
    """Create an environment with adjacent agents."""
    cfg = MettaGridConfig(
        game=GameConfig(
            num_agents=2,
            obs=ObsConfig(width=3, height=3, num_tokens=NUM_OBS_TOKENS),
            max_steps=1000,
            actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
            objects={"wall": WallConfig(tags=["wall"])},
            resource_names=["laser", "armor", "heart"],
            map_builder=AsciiMapBuilder.Config(
                map_data=[
                    ["#", "#", "#", "#", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "@", "@", ".", "#"],
                    ["#", ".", ".", ".", "#"],
                    ["#", "#", "#", "#", "#"],
                ],
                char_to_map_name=DEFAULT_CHAR_TO_NAME,
            ),
        )
    )

    return Simulation(cfg)


class TestObservations:
    """Test observation functionality."""

    def test_observation_structure(self, basic_sim):
        """Test basic observation structure."""
        obs = basic_sim._c_sim.observations()

        # global token is always at the center of the observation window
        obs_half_height = basic_sim.config.game.obs.height // 2
        obs_half_width = basic_sim.config.game.obs.width // 2
        global_token_location = PackedCoordinate.pack(obs_half_height, obs_half_width)
        compass_feature_id = basic_sim.config.game.id_map().feature_id("agent:compass")
        helper = ObservationHelper()

        # Map agent_id -> (row, col)
        agent_positions: dict[int, tuple[int, int]] = {}
        for obj in basic_sim.grid_objects().values():
            type_name = obj["type_name"]
            if type_name in {"agent", "agent.agent"}:
                agent_positions[int(obj["agent_id"])] = (int(obj["r"]), int(obj["c"]))

        map_center_row = basic_sim.map_height // 2
        map_center_col = basic_sim.map_width // 2
        center_obs_position = (obs_half_width, obs_half_height)

        # Test center-aligned global tokens
        for agent_idx in range(basic_sim.num_agents):
            for token_idx in range(3):
                assert obs[agent_idx, token_idx, 0] == global_token_location

            compass_tokens = helper.find_tokens(obs[agent_idx], feature_id=compass_feature_id)

            agent_row, agent_col = agent_positions[agent_idx]
            delta_row = map_center_row - agent_row
            delta_col = map_center_col - agent_col
            step_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
            step_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)

            if step_row == 0 and step_col == 0:
                assert compass_tokens.shape[0] == 0, "Compass should be absent when agent is at the center"
            else:
                assert compass_tokens.shape[0] == 1, "Expected exactly one compass token"
                compass_position = helper.get_positions_from_tokens(compass_tokens)[0]
                expected_position = (center_obs_position[0] + step_col, center_obs_position[1] + step_row)
                assert compass_position == expected_position, (
                    f"Compass should point from {center_obs_position} toward {expected_position}"
                )

        # Test empty terminator
        assert (obs[0, -1, :] == TokenTypes.EMPTY_TOKEN).all()
        assert (obs[1, -1, :] == TokenTypes.EMPTY_TOKEN).all()

    def test_detailed_wall_observations(self, basic_sim):
        """Test detailed wall observations for both agents."""
        obs = basic_sim._c_sim.observations()
        tag_feature_id = basic_sim.config.game.id_map().feature_id("tag")
        # Find tag id for 'wall'
        tag_names_list = basic_sim.config.game.id_map().tag_names()  # list of tag names, sorted alphabetically
        wall_tag_id = next(i for i, name in enumerate(tag_names_list) if name == "wall")
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

        agent0_wall_tokens = helper.find_tokens(agent0_obs, feature_id=tag_feature_id, value=wall_tag_id)
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

        agent1_wall_tokens = helper.find_tokens(agent1_obs, feature_id=tag_feature_id, value=wall_tag_id)
        agent1_wall_positions = helper.get_positions_from_tokens(agent1_wall_tokens)
        assert set(agent1_wall_positions) == set(wall_positions_agent1), (
            f"Agent 1: Expected walls at {wall_positions_agent1}, got {agent1_wall_positions}"
        )

        # Verify wall count
        assert len(agent1_wall_tokens) == 3, "Agent 1 should see exactly 3 walls"

    def test_agents_see_each_other(self, adjacent_agents_sim):
        """Test that adjacent agents can see each other."""
        obs = adjacent_agents_sim._c_sim.observations()
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

    def test_observation_token_order(self, basic_sim):
        """Test that observation tokens are ordered by distance."""
        obs = basic_sim._c_sim.observations()

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

    def test_initial_global_tokens(self, basic_sim):
        """Test initial global token values."""
        obs = basic_sim._c_sim.observations()
        episode_completion_pct_feature_id = basic_sim.config.game.id_map().feature_id("episode_completion_pct")
        last_action_feature_id = basic_sim.config.game.id_map().feature_id("last_action")
        last_reward_feature_id = basic_sim.config.game.id_map().feature_id("last_reward")
        helper = ObservationHelper()

        # Global tokens are at the center of the observation window
        global_x = basic_sim.config.game.obs.width // 2
        global_y = basic_sim.config.game.obs.height // 2

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
                objects={"wall": WallConfig(tags=["wall"])},
                global_obs=GlobalObsConfig(episode_completion_pct=True, last_action=True, last_reward=True),
                resource_names=["laser", "armor", "heart"],
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist(), char_to_map_name=DEFAULT_CHAR_TO_NAME),
            )
        )
        env = Simulation(cfg)
        episode_completion_pct_feature_id = env.config.game.id_map().feature_id("episode_completion_pct")
        last_action_feature_id = env.config.game.id_map().feature_id("last_action")
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
        expected_completion = int(0.1 * 256)
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

        # take a bunch more steps and check episode completion.
        for _ in range(15):
            env.step()
        expected_completion = 255
        completion_values = helper.find_token_values(
            obs[0], location=(global_x, global_y), feature_id=episode_completion_pct_feature_id
        )
        assert completion_values == [expected_completion], (
            f"Expected completion {expected_completion}, got {completion_values}"
        )

    def test_vibe_signaling(self):
        """Test that agents can signal using vibes and observe each other's vibes."""
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

        # Create environment with change_vibe enabled and 8 vibes
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=2,
                obs=ObsConfig(width=3, height=3, num_tokens=NUM_OBS_TOKENS),
                max_steps=10,
                # Use 3x3 observation window
                actions=ActionsConfig(
                    noop=NoopActionConfig(),
                    move=MoveActionConfig(),
                    change_vibe=ChangeVibeActionConfig(enabled=True, vibes=VIBES[:8]),
                ),
                objects={"wall": WallConfig(tags=["wall"])},
                resource_names=["laser", "armor"],
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist(), char_to_map_name=DEFAULT_CHAR_TO_NAME),
            )
        )
        sim = Simulation(cfg)
        vibe_feature_id = sim.config.game.id_map().feature_id("vibe")

        obs = sim._c_sim.observations()

        # Check if we're seeing uninitialized memory issues
        agent0_self_vibe = helper.find_token_values(obs[0], location=(1, 1), feature_id=vibe_feature_id)
        agent0_sees_agent1_vibe = helper.find_token_values(obs[0], location=(2, 1), feature_id=vibe_feature_id)
        agent1_self_vibe = helper.find_token_values(obs[1], location=(1, 1), feature_id=vibe_feature_id)
        agent1_sees_agent0_vibe = helper.find_token_values(obs[1], location=(0, 1), feature_id=vibe_feature_id)

        # Initially, both agents should have vibe 0 (default)
        # Since vibe 0 is suppressed, we should NOT find any vibe tokens
        assert len(agent0_self_vibe) == 0, f"Agent 0 with vibe 0 should have no vibe token, got {agent0_self_vibe}"
        assert len(agent0_sees_agent1_vibe) == 0, (
            f"Agent 0 should see Agent 1 with no vibe token (vibe 0), got {agent0_sees_agent1_vibe}"
        )
        assert len(agent1_self_vibe) == 0, f"Agent 1 with vibe 0 should have no vibe token, got {agent1_self_vibe}"
        assert len(agent1_sees_agent0_vibe) == 0, (
            f"Agent 1 should see Agent 0 with no vibe token (vibe 0), got {agent1_sees_agent0_vibe}"
        )

        # Test changing vibes
        # Get available change_vibe actions (they use change_vibe_ prefix internally)
        vibe_actions = [name for name in sim.action_names if name.startswith("change_vibe_")]
        assert len(vibe_actions) >= 6, f"Should have at least 6 vibe actions, got {vibe_actions}"

        # Pick specific vibe actions (use actual action names from list)
        vibe_action_3 = vibe_actions[3] if len(vibe_actions) > 3 else vibe_actions[0]
        vibe_action_5 = vibe_actions[5] if len(vibe_actions) > 5 else vibe_actions[1]

        # Test 1: Agent 0 changes to vibe 3, Agent 1 changes to vibe 5
        sim.agent(0).set_action(vibe_action_3)
        sim.agent(1).set_action(vibe_action_5)
        sim.step()
        obs = sim._c_sim.observations()

        agent0_self_vibe = helper.find_token_values(obs[0], location=(1, 1), feature_id=vibe_feature_id)
        # Agent 0 should now have a non-zero vibe (vibe was changed)
        assert len(agent0_self_vibe) > 0, "Agent 0 should have a vibe token after changing vibe"
        assert agent0_self_vibe != 0, f"Agent 0 vibe should not be 0 (default), got {agent0_self_vibe}"

        agent1_sees_agent0_vibe = helper.find_token_values(obs[1], location=(0, 1), feature_id=vibe_feature_id)
        assert len(agent1_sees_agent0_vibe) > 0, "Agent 1 should see Agent 0's vibe"
        assert agent1_sees_agent0_vibe == agent0_self_vibe, "Agent 1 should see the same vibe as Agent 0 has"

        agent1_self_vibe = helper.find_token_values(obs[1], location=(1, 1), feature_id=vibe_feature_id)
        assert len(agent1_self_vibe) > 0, "Agent 1 should have a vibe token after changing vibe"
        assert agent1_self_vibe != agent0_self_vibe, "Agent 1 should have different vibe than Agent 0"

        # Test 2: Invalid vibe values (should be no-op)
        assert "change_vibe_invalid" not in sim.action_names, "Invalid vibe action should not exist"

        # Test 3: Changing back to vibe 0 removes the token

        # Change back to vibe 0 (first vibe action)
        vibe_action_0 = vibe_actions[0]
        sim.agent(0).set_action(vibe_action_0)
        sim.agent(1).set_action(vibe_action_0)
        sim.step()
        obs = sim._c_sim.observations()

        # Verify vibe tokens are gone
        agent0_vibe = helper.find_token_values(obs[0], location=(1, 1), feature_id=vibe_feature_id)
        agent1_vibe = helper.find_token_values(obs[1], location=(1, 1), feature_id=vibe_feature_id)

        assert len(agent0_vibe) == 0, f"Agent 0 changed to vibe 0 should have no token, got {agent0_vibe}"
        assert len(agent1_vibe) == 0, f"Agent 1 changed to vibe 0 should have no token, got {agent1_vibe}"


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

        # We'll track walls at the edges instead of using a special object
        # No need to place any other objects

        # Create environment with 7x7 observation window
        cfg = MettaGridConfig(
            game=GameConfig(
                num_agents=1,
                obs=ObsConfig(width=7, height=7, num_tokens=NUM_OBS_TOKENS),
                max_steps=50,  # Enough steps to walk around
                actions=ActionsConfig(noop=NoopActionConfig(), move=MoveActionConfig()),
                objects={
                    "wall": WallConfig(tags=["wall"]),
                },
                resource_names=["laser", "resource1", "resource2"],  # laser required for attack action
                map_builder=AsciiMapBuilder.Config(map_data=game_map.tolist(), char_to_map_name=DEFAULT_CHAR_TO_NAME),
            )
        )
        sim = Simulation(cfg)
        tag_feature_id = sim.config.game.id_map().feature_id("tag")
        tag_names_list = sim.config.game.id_map().tag_names()
        wall_tag_id = tag_names_list.index("wall")

        obs = sim._c_sim.observations()

        # Verify initial position - agent should be at center of observation
        agent_tokens = helper.find_tokens(obs[0], location=(3, 3))
        assert len(agent_tokens) > 0, "Agent should see itself at center (3,3)"

        # Check walls are visible around the edges
        # Agent at (row=2, col=2) with 7x7 window should see walls at top and left
        wall_tokens = helper.find_tokens(obs[0], feature_id=tag_feature_id, value=wall_tag_id)
        assert len(wall_tokens) > 0, "Should see walls around edges"

        print("\nInitial state: Agent at (2,2), walls visible")

        # Move right (East) several steps to test observation window tracking
        for step in range(5):
            sim.agent(0).set_action("move_east")
            sim.step()
            obs = sim._c_sim.observations()

            # Verify agent is still at center of its observation
            agent_tokens = helper.find_tokens(obs[0], location=(3, 3))
            assert len(agent_tokens) > 0, f"Agent should still see itself at center (3,3) after step {step + 1}"

            # Verify walls are still visible at edges
            wall_tokens = helper.find_tokens(obs[0], feature_id=tag_feature_id, value=wall_tag_id)
            assert len(wall_tokens) > 0, f"Should still see walls after step {step + 1}"

        print("\nAfter moving east 5 steps: Agent observation window correctly tracks position")

        # Now walk to bottom-right corner
        # Move right to x=13
        for _ in range(5):
            sim.agent(0).set_action("move_east")
            sim.step()
            obs = sim._c_sim.observations()

        # Move down to y=8 using move_south
        for _ in range(6):
            sim.agent(0).set_action("move_south")
            sim.step()
            obs = sim._c_sim.observations()

        # Verify agent is still at center of observation
        agent_tokens = helper.find_tokens(obs[0], location=(3, 3))
        assert len(agent_tokens) > 0, "Agent should still see itself at center (3,3)"

        # Verify walls are still visible at the edges
        wall_tokens = helper.find_tokens(obs[0], feature_id=tag_feature_id, value=wall_tag_id)
        assert len(wall_tokens) > 0, "Should still see walls at edges even at bottom-right corner"

        print("\nSUCCESS: Observation window correctly tracks agent movement to corner")
