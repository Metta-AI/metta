from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config
from metta.mettagrid.mettagrid_env import dtype_actions


# Constants from C++ code
@dataclass
class TokenTypes:
    # Observation features
    TYPE_ID_FEATURE: int = 0
    GROUP: int = 1
    HP: int = 2
    FROZEN: int = 3
    ORIENTATION: int = 4
    COLOR: int = 5
    CONVERTING_OR_COOLING_DOWN: int = 6
    SWAPPABLE: int = 7
    EPISODE_COMPLETION_PCT: int = 8
    LAST_ACTION: int = 9
    LAST_ACTION_ARG: int = 10
    LAST_REWARD: int = 11
    GLYPH: int = 12

    # Object type IDs
    WALL_TYPE_ID: int = 1


@dataclass
class EnvConfig:  # Renamed from TestConfig to avoid pytest confusion
    NUM_AGENTS: int = 2
    OBS_HEIGHT: int = 3
    OBS_WIDTH: int = 3
    NUM_OBS_TOKENS: int = 100
    OBS_TOKEN_SIZE: int = 3
    EMPTY_TOKEN = [0xFF, 0xFF, 0xFF]


class TestEnvironmentBuilder:
    """Helper class to build test environments with different configurations."""

    @staticmethod
    def create_basic_grid(width: int = 8, height: int = 4) -> np.ndarray:
        """Create a basic grid with walls around perimeter."""
        game_map = np.full((height, width), "empty", dtype="<U50")
        game_map[0, :] = "wall"
        game_map[-1, :] = "wall"
        game_map[:, 0] = "wall"
        game_map[:, -1] = "wall"
        return game_map

    @staticmethod
    def place_agents(game_map: np.ndarray, positions: List[Tuple[int, int]]) -> np.ndarray:
        """Place agents at specified positions."""

        # Coordinate convention: grid position (x, y) = (col, row)
        # The grid is treated as a logical object with width and height.
        # Internally, the NumPy map uses shape (height, width) = (rows, cols),
        # so indexing is game_map[y, x].

        for _, (y, x) in enumerate(positions):
            game_map[y, x] = "agent.red"
        return game_map

    @staticmethod
    def create_environment(game_map: np.ndarray, max_steps: int = 10, num_agents: int | None = None) -> MettaGrid:
        """Create a MettaGrid environment from a game map."""
        if num_agents is None:
            num_agents = EnvConfig.NUM_AGENTS

        game_config = {
            "max_steps": max_steps,
            "num_agents": num_agents,
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
                "change_glyph": {"enabled": False, "number_of_glyphs": 4},
            },
            "groups": {"red": {"id": 0, "props": {}}},
            "objects": {"wall": {"type_id": 1}},
            "agent": {},
        }
        return MettaGrid(from_mettagrid_config(game_config), game_map.tolist(), 42)


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


@pytest.fixture
def basic_env():
    """Create a basic test environment."""
    builder = TestEnvironmentBuilder()
    game_map = builder.create_basic_grid()
    game_map = builder.place_agents(game_map, [(1, 1), (2, 4)])
    return builder.create_environment(game_map)


@pytest.fixture
def adjacent_agents_env():
    """Create an environment with adjacent agents."""
    builder = TestEnvironmentBuilder()
    game_map = builder.create_basic_grid(5, 5)
    game_map = builder.place_agents(game_map, [(2, 1), (2, 2)])
    return builder.create_environment(game_map)


class TestBasicFunctionality:
    """Test basic environment functionality."""

    def test_environment_initialization(self, basic_env):
        """Test basic environment properties."""
        assert basic_env.map_width == 8
        assert basic_env.map_height == 4
        assert len(basic_env.action_names()) > 0
        assert "noop" in basic_env.action_names()

        obs, info = basic_env.reset()
        assert obs.shape == (EnvConfig.NUM_AGENTS, EnvConfig.NUM_OBS_TOKENS, EnvConfig.OBS_TOKEN_SIZE)
        assert isinstance(info, dict)

    def test_grid_hash(self, basic_env):
        """Test grid hash consistency."""
        assert basic_env.initial_grid_hash == 9437127895318323250

    def test_action_interface(self, basic_env):
        """Test action interface and basic action execution."""
        basic_env.reset()

        action_names = basic_env.action_names()
        assert "noop" in action_names
        assert "move" in action_names
        assert "rotate" in action_names

        noop_idx = action_names.index("noop")
        actions = np.full((EnvConfig.NUM_AGENTS, 2), [noop_idx, 0], dtype=dtype_actions)

        obs, rewards, terminals, truncations, info = basic_env.step(actions)

        # Check shapes and types
        assert obs.shape == (EnvConfig.NUM_AGENTS, EnvConfig.NUM_OBS_TOKENS, EnvConfig.OBS_TOKEN_SIZE)
        assert rewards.shape == (EnvConfig.NUM_AGENTS,)
        assert terminals.shape == (EnvConfig.NUM_AGENTS,)
        assert truncations.shape == (EnvConfig.NUM_AGENTS,)
        assert isinstance(info, dict)

        # Action success should be boolean and per-agent
        action_success = basic_env.action_success()
        assert len(action_success) == EnvConfig.NUM_AGENTS
        assert all(isinstance(x, bool) for x in action_success)

    def test_environment_state_consistency(self, basic_env):
        """Test that environment state remains consistent across operations."""
        obs1, _ = basic_env.reset()
        initial_objects = basic_env.grid_objects()

        noop_idx = basic_env.action_names().index("noop")
        actions = np.full((EnvConfig.NUM_AGENTS, 2), [noop_idx, 0], dtype=dtype_actions)

        obs2, _, _, _, _ = basic_env.step(actions)
        post_step_objects = basic_env.grid_objects()

        # Object count should remain unchanged
        assert len(initial_objects) == len(post_step_objects)

        # Dimensions should remain unchanged
        assert basic_env.map_width == 8
        assert basic_env.map_height == 4

        # Action set should remain unchanged after stepping
        actions1 = basic_env.action_names()
        basic_env.step(actions)
        actions2 = basic_env.action_names()
        assert actions1 == actions2


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


class TestPackedCoordinate:
    """Test PackedCoordinate functionality."""

    def test_packed_coordinate(self):
        """Test the PackedCoordinate functionality."""
        # Test constants
        assert PackedCoordinate.MAX_PACKABLE_COORD == 14

        # Test all valid coordinates
        successfully_packed = 0
        for row in range(15):  # 0-14
            for col in range(15):  # 0-14
                packed = PackedCoordinate.pack(row, col)
                unpacked = PackedCoordinate.unpack(packed)
                assert unpacked == (row, col), f"Roundtrip failed for ({row}, {col})"
                successfully_packed += 1

        # Verify we can pack 225 positions (15x15 grid)
        assert successfully_packed == 225, f"Expected 225 packable positions, got {successfully_packed}"

        # Test empty/0xFF handling
        assert PackedCoordinate.is_empty(0xFF)
        assert PackedCoordinate.unpack(0xFF) is None
        assert not PackedCoordinate.is_empty(0x00)
        assert not PackedCoordinate.is_empty(0xE0)
        assert not PackedCoordinate.is_empty(0xEE)

        # Test invalid coordinates
        invalid_coords = [(15, 0), (0, 15), (15, 15), (16, 0), (0, 16), (255, 255)]
        for row, col in invalid_coords:
            try:
                PackedCoordinate.pack(row, col)
                raise AssertionError(f"Should have raised exception for ({row}, {col})")
            except ValueError:
                pass  # Expected
