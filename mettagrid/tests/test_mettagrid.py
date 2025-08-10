from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from metta.mettagrid.mettagrid_c import PackedCoordinate, dtype_actions
from metta.mettagrid.test_support import EnvConfig


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
