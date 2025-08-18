import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid, PackedCoordinate, dtype_actions
from metta.mettagrid.test_support import TestEnvironmentBuilder, TokenTypes

NUM_OBS_TOKENS = 50


@pytest.fixture
def basic_env() -> MettaGrid:
    """Create a basic test environment."""
    builder = TestEnvironmentBuilder()
    game_map = builder.create_basic_grid()
    game_map = builder.place_agents(game_map, [(1, 1), (2, 4)])
    return builder.create_environment(game_map, obs_width=3, obs_height=3, num_observation_tokens=NUM_OBS_TOKENS)


class TestBasicFunctionality:
    """Test basic environment functionality."""

    def test_environment_initialization(self, basic_env: MettaGrid):
        """Test basic environment properties."""
        assert basic_env.map_width == 8
        assert basic_env.map_height == 4
        assert len(basic_env.action_names()) > 0
        assert "noop" in basic_env.action_names()

        obs, info = basic_env.reset()

        assert obs.shape == (basic_env.num_agents, NUM_OBS_TOKENS, TokenTypes.OBS_TOKEN_SIZE)
        assert isinstance(info, dict)

    def test_grid_hash(self, basic_env: MettaGrid):
        """Test grid hash consistency."""
        assert basic_env.initial_grid_hash == 9437127895318323250

    def test_action_interface(self, basic_env: MettaGrid):
        """Test action interface and basic action execution."""
        basic_env.reset()

        action_names = basic_env.action_names()
        assert "noop" in action_names
        assert "move" in action_names
        assert "rotate" in action_names

        noop_idx = action_names.index("noop")
        actions = np.full((basic_env.num_agents, 2), [noop_idx, 0], dtype=dtype_actions)

        obs, rewards, terminals, truncations, info = basic_env.step(actions)

        # Check shapes and types
        assert obs.shape == (basic_env.num_agents, NUM_OBS_TOKENS, TokenTypes.OBS_TOKEN_SIZE)
        assert rewards.shape == (basic_env.num_agents,)
        assert terminals.shape == (basic_env.num_agents,)
        assert truncations.shape == (basic_env.num_agents,)
        assert isinstance(info, dict)

        # Action success should be boolean and per-agent
        action_success = basic_env.action_success()
        assert len(action_success) == basic_env.num_agents
        assert all(isinstance(x, bool) for x in action_success)

    def test_environment_state_consistency(self, basic_env: MettaGrid):
        """Test that environment state remains consistent across operations."""
        obs1, _ = basic_env.reset()
        initial_objects = basic_env.grid_objects()

        noop_idx = basic_env.action_names().index("noop")
        actions = np.full((basic_env.num_agents, 2), [noop_idx, 0], dtype=dtype_actions)

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
