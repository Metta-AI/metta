from typing import List, Tuple

import numpy as np

from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid import dtype_actions

from .test_mettagrid import EnvConfig, TestEnvironmentBuilder, TokenTypes


class ObservationHelper:
    """Helper class for observation-related operations."""

    @staticmethod
    def find_tokens_at_location(obs: np.ndarray, x: int, y: int) -> np.ndarray:
        """Find all tokens at a specific location."""
        location = PackedCoordinate.pack(y, x)
        return obs[obs[:, 0] == location]

    @staticmethod
    def find_tokens_by_type(obs: np.ndarray, type_id: int) -> np.ndarray:
        """Find all tokens of a specific type."""
        return obs[obs[:, 1] == type_id]

    @staticmethod
    def count_walls(obs: np.ndarray) -> int:
        """Count the number of wall tokens in an observation."""
        return np.sum(obs[:, 2] == TokenTypes.WALL_TYPE_ID)

    @staticmethod
    def get_wall_positions(obs: np.ndarray) -> List[Tuple[int, int]]:
        """Get all wall positions from an observation."""
        positions = []
        wall_tokens = obs[obs[:, 2] == TokenTypes.WALL_TYPE_ID]
        for token in wall_tokens:
            coords = PackedCoordinate.unpack(token[0])
            if coords:
                row, col = coords
                positions.append((col, row))  # Return as (x, y)
        return positions


class TestObservations:
    """Test observation functionality."""

    def test_observation_structure(self, basic_env):
        """Test basic observation structure."""
        obs, _ = basic_env.reset()

        # global token is always at the center of the observation window
        global_token_location = PackedCoordinate.pack(EnvConfig.OBS_HEIGHT // 2, EnvConfig.OBS_WIDTH // 2)

        # Test global tokens (first 4 tokens)
        for agent_idx in range(EnvConfig.NUM_AGENTS):
            for token_idx in range(4):
                assert obs[agent_idx, token_idx, 0] == global_token_location

        # Test empty terminator
        assert (obs[0, -1, :] == EnvConfig.EMPTY_TOKEN).all()
        assert (obs[1, -1, :] == EnvConfig.EMPTY_TOKEN).all()

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
            self._check_token_exists(agent0_obs, x, y, TokenTypes.TYPE_ID_FEATURE, TokenTypes.WALL_TYPE_ID, "Agent 0")

        # Check no walls at empty positions
        for x, y in no_wall_positions_agent0:
            location = PackedCoordinate.pack(y, x)
            wall_tokens = (agent0_obs[:, 0] == location) & (agent0_obs[:, 2] == TokenTypes.WALL_TYPE_ID)
            assert not wall_tokens.any(), f"Agent 0: Expected no wall at ({x}, {y})"

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
            self._check_token_exists(agent1_obs, x, y, TokenTypes.TYPE_ID_FEATURE, TokenTypes.WALL_TYPE_ID, "Agent 1")

        # Check no walls at empty positions
        for x, y in no_wall_positions_agent1:
            location = PackedCoordinate.pack(y, x)
            wall_tokens = (agent1_obs[:, 0] == location) & (agent1_obs[:, 2] == TokenTypes.WALL_TYPE_ID)
            assert not wall_tokens.any(), f"Agent 1: Expected no wall at ({x}, {y})"

        # Verify wall count
        assert helper.count_walls(agent1_obs) == 3, "Agent 1 should see exactly 3 walls"

    def test_agent_surrounded_by_altars(self):
        """Test agent observation when surrounded by colored altars."""
        # Create a 5x5 environment with agent in center surrounded by altars
        builder = TestEnvironmentBuilder()
        game_map = builder.create_basic_grid(5, 5)

        # Place agent in center at grid position (2,2)
        game_map[2, 2] = "agent.red"

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

        # Create game config with altar objects
        game_config = {
            "max_steps": 10,
            "num_agents": 1,  # Just one agent for this test
            "obs_width": EnvConfig.OBS_WIDTH,
            "obs_height": EnvConfig.OBS_HEIGHT,
            "num_observation_tokens": EnvConfig.NUM_OBS_TOKENS,
            "inventory_item_names": ["resource1", "resource2"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": False},
                "put_items": {"enabled": False},
                "get_items": {"enabled": False},
                "swap": {"enabled": False},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            },
            "groups": {"red": {"id": 0, "props": {}}},
            "objects": {
                "wall": {"type_id": 1},
            },
            "agent": {},
        }

        # Add altar configurations with different colors
        for i, (x, y, color) in enumerate(altar_positions):
            altar_name = f"altar_{i + 1}"
            game_map[y, x] = altar_name
            game_config["objects"][altar_name] = {
                "type_id": i + 2,  # type_ids 2-9 for altars
                "input_resources": {"resource1": 1},
                "output_resources": {"resource2": 1},
                "max_output": 10,
                "conversion_ticks": 5,
                "cooldown": 3,
                "initial_resource_count": 0,
                "color": color,
            }

        env = MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)
        obs, _ = env.reset()

        agent_obs = obs[0]

        # The agent at (2,2) should see all 8 altars in its 3x3 observation window
        # Expected altar positions in observation coordinates:
        #   A B C     (0,0) (1,0) (2,0)
        #   D & E  -> (0,1) (1,1) (2,1)
        #   F G H     (0,2) (1,2) (2,2)

        expected_altars = [
            (0, 0, 2, 1),  # A: top-left, type_id=2, color=1
            (1, 0, 3, 2),  # B: top-center, type_id=3, color=2
            (2, 0, 4, 3),  # C: top-right, type_id=4, color=3
            (0, 1, 5, 4),  # D: middle-left, type_id=5, color=4
            (2, 1, 6, 5),  # E: middle-right, type_id=6, color=5
            (0, 2, 7, 6),  # F: bottom-left, type_id=7, color=6
            (1, 2, 8, 7),  # G: bottom-center, type_id=8, color=7
            (2, 2, 9, 8),  # H: bottom-right, type_id=9, color=8
        ]

        # Check that we see each altar with correct type_id
        for x, y, expected_type_id, expected_color in expected_altars:
            location = PackedCoordinate.pack(y, x)

            # Find tokens at this location
            location_tokens = agent_obs[agent_obs[:, 0] == location]

            # Should have tokens for this altar
            assert len(location_tokens) > 0, f"Should have tokens at ({x}, {y}) for altar"

            # Check type_id token
            type_id_tokens = location_tokens[location_tokens[:, 1] == TokenTypes.TYPE_ID_FEATURE]
            assert len(type_id_tokens) > 0, f"Should have type_id token at ({x}, {y})"
            assert type_id_tokens[0, 2] == expected_type_id, (
                f"Altar at ({x}, {y}) should have type_id {expected_type_id}, got {type_id_tokens[0, 2]}"
            )

            # Check color token (ObservationFeature::Color = 5)
            color_tokens = location_tokens[location_tokens[:, 1] == 5]
            assert len(color_tokens) > 0, f"Should have color token at ({x}, {y})"
            assert color_tokens[0, 2] == expected_color, (
                f"Altar at ({x}, {y}) should have color {expected_color}, got {color_tokens[0, 2]}"
            )

            # Check converter status token (ObservationFeature::ConvertingOrCoolingDown = 6)
            converter_tokens = location_tokens[location_tokens[:, 1] == 6]
            assert len(converter_tokens) > 0, f"Should have converter status token at ({x}, {y})"

        # Verify the agent sees itself at center (1,1)
        self_location = PackedCoordinate.pack(1, 1)
        self_tokens = agent_obs[agent_obs[:, 0] == self_location]
        assert len(self_tokens) > 0, "Agent should see itself at center position"

        # Count unique type_ids (excluding walls and empty)
        type_id_tokens = agent_obs[agent_obs[:, 1] == TokenTypes.TYPE_ID_FEATURE]
        unique_type_ids = set(type_id_tokens[:, 2])
        unique_type_ids.discard(0)  # Remove empty/agent type
        unique_type_ids.discard(1)  # Remove wall type

        assert len(unique_type_ids) == 8, f"Should see 8 different altar types, got {len(unique_type_ids)}"

    def _check_token_exists(self, obs, x, y, type_id, feature_id, agent_name):
        """Helper to check if a specific token exists at a location."""
        location = PackedCoordinate.pack(y, x)
        token_matches = obs[:, :] == [location, type_id, feature_id]
        assert token_matches.all(axis=1).any(), (
            f"{agent_name}: Expected token [{location}, {type_id}, {feature_id}] at ({x}, {y})"
        )

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

        # Check token types
        assert obs[0, 0, 1] == TokenTypes.EPISODE_COMPLETION_PCT
        assert obs[0, 1, 1] == TokenTypes.LAST_ACTION
        assert obs[0, 2, 1] == TokenTypes.LAST_ACTION_ARG
        assert obs[0, 3, 1] == TokenTypes.LAST_REWARD

        # Check initial values
        assert obs[0, 0, 2] == 0  # 0% completion
        assert obs[0, 1, 2] == 0  # No last action
        assert obs[0, 2, 2] == 0  # No last action arg
        assert obs[0, 3, 2] == 0  # No last reward

    def test_global_tokens_update(self, basic_env):
        """Test that global tokens update correctly."""
        basic_env.reset()

        # Take a noop action
        noop_idx = basic_env.action_names().index("noop")
        actions = np.full((EnvConfig.NUM_AGENTS, 2), [noop_idx, 0], dtype=dtype_actions)
        obs, _, _, _, _ = basic_env.step(actions)

        # Check episode completion updated (1/10 = 10%)
        expected_completion = int(round(0.1 * 255))
        assert obs[0, 0, 2] == expected_completion
        assert obs[0, 1, 2] == noop_idx
        assert obs[0, 2, 2] == 0

        # Take a move action
        move_idx = basic_env.action_names().index("move")
        actions = np.full((EnvConfig.NUM_AGENTS, 2), [move_idx, 1], dtype=dtype_actions)
        obs, _, _, _, _ = basic_env.step(actions)

        # Check updates
        expected_completion = int(round(0.2 * 255))
        assert obs[0, 0, 2] == expected_completion
        assert obs[0, 1, 2] == move_idx
        assert obs[0, 2, 2] == 1

    def test_glyph_signaling(self):
        """Test that agents can signal using glyphs and observe each other's glyphs."""
        # Create a 5x5 environment with two adjacent agents
        builder = TestEnvironmentBuilder()
        game_map = builder.create_basic_grid(5, 5)

        # Place two agents next to each other
        # Agent 0 at (1,2), Agent 1 at (2,2)
        game_map[2, 1] = "agent.red"
        game_map[2, 2] = "agent.blue"

        # Create environment with change_glyph enabled and 8 glyphs
        game_config = {
            "max_steps": 10,
            "num_agents": 2,
            "obs_width": EnvConfig.OBS_WIDTH,
            "obs_height": EnvConfig.OBS_HEIGHT,
            "num_observation_tokens": EnvConfig.NUM_OBS_TOKENS,
            "inventory_item_names": ["laser", "armor"],
            "actions": {
                "noop": {"enabled": True},
                "move": {"enabled": True},
                "rotate": {"enabled": True},
                "attack": {"enabled": False},
                "put_items": {"enabled": False},
                "get_items": {"enabled": False},
                "swap": {"enabled": False},
                "change_color": {"enabled": False},
                "change_glyph": {"enabled": True, "number_of_glyphs": 8},
            },
            "groups": {"red": {"id": 0, "props": {}}, "blue": {"id": 1, "props": {}}},
            "objects": {"wall": {"type_id": 1}},
            "agent": {},
        }

        env = MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)
        obs, _ = env.reset()

        # Define glyph feature type
        GLYPH_FEATURE = TokenTypes.GLYPH

        # Helper function to find glyph tokens
        def find_glyph_at_location(observation, x, y):
            location = PackedCoordinate.pack(y, x)
            location_tokens = observation[observation[:, 0] == location]
            glyph_tokens = location_tokens[location_tokens[:, 1] == GLYPH_FEATURE]
            return glyph_tokens[0, 2] if len(glyph_tokens) > 0 else None

        print("\n=== Testing Initial Glyph Values ===")

        # Initially, both agents should have glyph 0 (default)
        # Since glyph 0 is suppressed, we should NOT find any glyph tokens
        agent0_self_glyph = find_glyph_at_location(obs[0], 1, 1)
        assert agent0_self_glyph is None, f"Agent 0 with glyph 0 should have no glyph token, got {agent0_self_glyph}"

        agent0_sees_agent1_glyph = find_glyph_at_location(obs[0], 2, 1)
        assert agent0_sees_agent1_glyph is None, (
            f"Agent 0 should see Agent 1 with no glyph token (glyph 0), got {agent0_sees_agent1_glyph}"
        )

        agent1_self_glyph = find_glyph_at_location(obs[1], 1, 1)
        assert agent1_self_glyph is None, f"Agent 1 with glyph 0 should have no glyph token, got {agent1_self_glyph}"

        agent1_sees_agent0_glyph = find_glyph_at_location(obs[1], 0, 1)
        assert agent1_sees_agent0_glyph is None, (
            f"Agent 1 should see Agent 0 with no glyph token (glyph 0), got {agent1_sees_agent0_glyph}"
        )

        print("✓ Both agents start with glyph 0 (no glyph token)")

        # Test changing glyphs
        print("\n=== Testing Glyph Changes ===")

        change_glyph_idx = env.action_names().index("change_glyph")
        noop_idx = env.action_names().index("noop")

        # Test 1: Agent 0 changes to glyph 3, Agent 1 stays at 0
        actions = np.array(
            [
                [change_glyph_idx, 3],  # Agent 0 changes to glyph 3
                [noop_idx, 0],  # Agent 1 does nothing
            ],
            dtype=dtype_actions,
        )

        obs, _, _, _, _ = env.step(actions)

        agent0_self_glyph = find_glyph_at_location(obs[0], 1, 1)
        assert agent0_self_glyph == 3, f"Agent 0 should have glyph 3, got {agent0_self_glyph}"

        agent1_sees_agent0_glyph = find_glyph_at_location(obs[1], 0, 1)
        assert agent1_sees_agent0_glyph == 3, f"Agent 1 should see Agent 0 with glyph 3, got {agent1_sees_agent0_glyph}"

        agent1_self_glyph = find_glyph_at_location(obs[1], 1, 1)
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

        agent0_self_glyph = find_glyph_at_location(obs[0], 1, 1)
        agent1_self_glyph = find_glyph_at_location(obs[1], 1, 1)

        assert agent0_self_glyph == 5, f"Agent 0 should have glyph 5, got {agent0_self_glyph}"
        assert agent1_self_glyph == 7, f"Agent 1 should have glyph 7, got {agent1_self_glyph}"

        # Verify they see each other's new glyphs
        agent0_sees_agent1 = find_glyph_at_location(obs[0], 2, 1)
        agent1_sees_agent0 = find_glyph_at_location(obs[1], 0, 1)

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

            agent0_glyph = find_glyph_at_location(obs[0], 1, 1)
            agent1_glyph = find_glyph_at_location(obs[1], 1, 1)
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
            agent0_sees_agent1 = find_glyph_at_location(obs[0], 2, 1)
            agent1_sees_agent0 = find_glyph_at_location(obs[1], 0, 1)

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
        assert find_glyph_at_location(obs[0], 1, 1) == 3
        assert find_glyph_at_location(obs[1], 1, 1) == 5

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

            agent0_glyph = find_glyph_at_location(obs[0], 1, 1)
            agent1_glyph = find_glyph_at_location(obs[1], 1, 1)

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

            agent0_glyph = find_glyph_at_location(obs[0], 1, 1)
            agent1_glyph = find_glyph_at_location(obs[1], 1, 1)

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
        assert find_glyph_at_location(obs[0], 1, 1) == 4
        assert find_glyph_at_location(obs[1], 1, 1) == 5

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
        agent0_glyph = find_glyph_at_location(obs[0], 1, 1)
        agent1_glyph = find_glyph_at_location(obs[1], 1, 1)

        assert agent0_glyph is None, f"Agent 0 changed to glyph 0 should have no token, got {agent0_glyph}"
        assert agent1_glyph is None, f"Agent 1 changed to glyph 0 should have no token, got {agent1_glyph}"

        print("✓ Changing back to glyph 0 correctly removes glyph tokens")

        print("\n=== All Glyph Tests Passed! ===")


class TestEdgeObservations:
    """Test observation behavior near world edges."""

    def test_observation_off_edge_with_large_window(self):
        """Test observation window behavior when agent walks to corner of large map."""
        # Create a 15x10 grid (width=15, height=10) with 7x7 observation window
        _builder = TestEnvironmentBuilder()
        game_map = np.full((10, 15), "empty", dtype="<U50")

        # Add walls around perimeter
        game_map[0, :] = "wall"
        game_map[-1, :] = "wall"
        game_map[:, 0] = "wall"
        game_map[:, -1] = "wall"

        # Place agent near top-left at (2, 2)
        game_map[2, 2] = "agent.red"

        # Place an altar at (7, 5) - we'll watch it move through our view
        game_map[5, 7] = "altar"

        # Create environment with 7x7 observation window
        game_config = {
            "max_steps": 50,  # Enough steps to walk around
            "num_agents": 1,
            "obs_width": 7,
            "obs_height": 7,
            "num_observation_tokens": 200,
            "inventory_item_names": ["resource1", "resource2"],
            "actions": {
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
                "altar": {
                    "type_id": 10,
                    "input_resources": {"resource1": 1},
                    "output_resources": {"resource2": 1},
                    "max_output": 10,
                    "conversion_ticks": 5,
                    "cooldown": 3,
                    "initial_resource_count": 0,
                    "color": 42,  # Distinctive color
                },
            },
            "agent": {},
        }

        env = MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)
        obs, _ = env.reset()
        helper = ObservationHelper()

        # Get action indices
        move_idx = env.action_names().index("move")
        rotate_idx = env.action_names().index("rotate")

        # Verify initial position - agent should be at center of observation
        agent_tokens = helper.find_tokens_at_location(obs[0], 3, 3)
        assert len(agent_tokens) > 0, "Agent should see itself at center (3,3)"

        # The altar at grid (7,5) should not be visible initially
        # Agent at (2,2) with 7x7 window sees from (-1,-1) to (5,5)
        # So altar at (7,5) is outside the view
        altar_visible = False
        for i in range(len(obs[0])):
            if obs[0, i, 1] == TokenTypes.TYPE_ID_FEATURE and obs[0, i, 2] == 10:  # type_id 10 = altar
                altar_visible = True
                break
        assert not altar_visible, "Altar should not be visible initially"

        print("\nInitial state: Agent at (2,2), altar at (7,5) - not visible")

        # Face right and move right 3 steps
        actions = np.array([[rotate_idx, 3]], dtype=dtype_actions)
        obs, _, _, _, _ = env.step(actions)

        for step in range(3):
            actions = np.array([[move_idx, 0]], dtype=dtype_actions)
            obs, _, _, _, _ = env.step(actions)

            # After step 0: agent at (3,2), window covers (0,0) to (6,5) - altar still not visible
            # After step 1: agent at (4,2), window covers (1,0) to (7,5) - altar just enters view!
            # After step 2: agent at (5,2), window covers (2,0) to (8,5) - altar clearly visible

            if step >= 1:  # Altar should be visible after first step
                # Find altar in observation
                altar_found = False
                for i in range(len(obs[0])):
                    if obs[0, i, 1] == TokenTypes.TYPE_ID_FEATURE and obs[0, i, 2] == 10:
                        altar_location = obs[0, i, 0]
                        altar_coords = PackedCoordinate.unpack(altar_location)
                        if altar_coords:
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
                            altar_found = True
                            break
                assert altar_found, f"Altar should be visible after step {step}"

        # Continue moving right until altar leaves view
        for step in range(3, 6):
            actions = np.array([[move_idx, 0]], dtype=dtype_actions)
            obs, _, _, _, _ = env.step(actions)

            # After step 3: agent at (6,2), altar at relative (1,3) - still visible
            # After step 4: agent at (7,2), altar at relative (0,3) - at center column
            # After step 5: agent at (8,2), altar at relative (-1,3) - at left edge

            altar_found = False
            for i in range(len(obs[0])):
                if obs[0, i, 1] == TokenTypes.TYPE_ID_FEATURE and obs[0, i, 2] == 10:
                    altar_location = obs[0, i, 0]
                    altar_coords = PackedCoordinate.unpack(altar_location)
                    if altar_coords:
                        obs_row, obs_col = altar_coords
                        expected_col = 7 - step
                        print(f"\nStep {step}: Agent at ({3 + step},2), altar at obs ({obs_col},{obs_row})")
                        altar_found = True
                        break

            if step <= 5:
                assert altar_found, f"Altar should still be visible at step {step}"

        # Continue moving right until altar leaves view
        for step in range(6, 9):
            actions = np.array([[move_idx, 0]], dtype=dtype_actions)
            obs, _, _, _, _ = env.step(actions)

            # After step 6: agent at (9,2), altar at relative (-2,3) - obs position (1,6)
            # After step 7: agent at (10,2), altar at relative (-3,3) - obs position (0,6) - at very edge
            # After step 8: agent at (11,2), altar at relative (-4,3) - outside 7x7 window

            altar_found = False
            for i in range(len(obs[0])):
                if obs[0, i, 1] == TokenTypes.TYPE_ID_FEATURE and obs[0, i, 2] == 10:
                    altar_location = obs[0, i, 0]
                    altar_coords = PackedCoordinate.unpack(altar_location)
                    if altar_coords:
                        obs_row, obs_col = altar_coords
                        print(f"\nStep {step}: Agent at ({3 + step},2), altar at obs ({obs_col},{obs_row})")
                    altar_found = True
                    break

            if step <= 7:
                assert altar_found, f"Altar should still be visible at step {step}"
            else:
                assert not altar_found, "Altar should have left the view"
                print(f"\nStep {step}: Agent at ({3 + step},2), altar no longer visible")

        # Now walk to bottom-right corner
        # Continue right to x=13
        for _ in range(5):
            actions = np.array([[move_idx, 0]], dtype=dtype_actions)
            obs, _, _, _, _ = env.step(actions)

        # Face down and move to y=8
        actions = np.array([[rotate_idx, 1]], dtype=dtype_actions)
        obs, _, _, _, _ = env.step(actions)

        for _ in range(6):
            actions = np.array([[move_idx, 0]], dtype=dtype_actions)
            obs, _, _, _, _ = env.step(actions)

        # Verify agent is still at center of observation
        agent_tokens = helper.find_tokens_at_location(obs[0], 3, 3)
        assert len(agent_tokens) > 0, "Agent should still see itself at center (3,3)"

        # Check walls at edges as before
        # Right wall at x=14 -> obs x=4
        for obs_y in range(7):
            grid_y = 8 + obs_y - 3
            if 0 <= grid_y <= 9:
                wall_tokens = helper.find_tokens_at_location(obs[0], 4, obs_y)
                wall_tokens = wall_tokens[wall_tokens[:, 2] == TokenTypes.WALL_TYPE_ID]
                assert len(wall_tokens) > 0, f"Should see right wall at obs ({4}, {obs_y})"

        # Bottom wall at y=9 -> obs y=4
        for obs_x in range(7):
            grid_x = 13 + obs_x - 3
            if 0 <= grid_x <= 14:
                wall_tokens = helper.find_tokens_at_location(obs[0], obs_x, 4)
                wall_tokens = wall_tokens[wall_tokens[:, 2] == TokenTypes.WALL_TYPE_ID]
                assert len(wall_tokens) > 0, f"Should see bottom wall at obs ({obs_x}, {4})"

        # Verify padding areas have no tokens
        for x in range(5, 7):
            for y in range(7):
                tokens = helper.find_tokens_at_location(obs[0], x, y)
                for i, token in enumerate(tokens):
                    if i >= 4:
                        assert np.array_equal(token, EnvConfig.EMPTY_TOKEN), f"Expected empty token at obs ({x},{y})"

        print("\nSUCCESS: Watched altar move through field of view and verified edge behavior")
