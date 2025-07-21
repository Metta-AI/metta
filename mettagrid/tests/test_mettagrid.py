from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

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
