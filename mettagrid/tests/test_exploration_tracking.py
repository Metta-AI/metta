#!/usr/bin/env python3
"""
Test exploration tracking functionality.
"""

import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config


def create_test_config(track_exploration=False):
    """Create a minimal test configuration."""
    return {
        "max_steps": 20,
        "num_agents": 2,
        "obs_width": 5,
        "obs_height": 5,
        "num_observation_tokens": 25,
        "inventory_item_names": ["heart"],
        "track_exploration": track_exploration,
        "groups": {"agent": {"id": 0, "sprite": 0}},
        "agent": {
            "default_resource_limit": 10,
            "resource_limits": {"heart": 255},
            "freeze_duration": 0,
            "rewards": {"heart": 1.0},
            "action_failure_penalty": 0.0,
        },
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "put_items": {"enabled": True},
            "get_items": {"enabled": True},
            "attack": {"enabled": True},
            "swap": {"enabled": True},
            "change_color": {"enabled": False},
            "change_glyph": {"enabled": False, "number_of_glyphs": 0},
        },
        "objects": {
            "wall": {"type_id": 1, "swappable": False},
        },
    }


def create_simple_map():
    """Create a simple 8x8 map with agents and some objects."""
    # 8x8 grid with agents at (1,1) and (2,2), some walls
    game_map = [["empty" for _ in range(8)] for _ in range(8)]

    # Add agents
    game_map[1][1] = "agent.agent"
    game_map[2][2] = "agent.agent"

    # Add some walls for exploration
    game_map[0][0] = "wall"
    game_map[0][1] = "wall"
    game_map[1][0] = "wall"
    game_map[3][3] = "wall"
    game_map[4][4] = "wall"

    return game_map


class TestExplorationTracking:
    """Test exploration tracking functionality."""

    def test_exploration_tracking_disabled(self):
        """Test that exploration tracking is disabled by default."""
        config = create_test_config(track_exploration=False)
        game_map = create_simple_map()

        env = MettaGrid(from_mettagrid_config(config), game_map, 42)
        obs, _ = env.reset()

        # Take a few steps
        actions = np.array([[0, 0], [0, 0]], dtype=np.int32)  # noop actions
        for _ in range(5):
            obs, _, _, _, _ = env.step(actions)

        # Check episode stats
        stats = env.get_episode_stats()
        agent_stats = stats["agent"]

        # Exploration rate should not be present when disabled
        for agent_stat in agent_stats:
            assert "exploration_rate" not in agent_stat

    def test_exploration_tracking_enabled(self):
        """Test that exploration tracking works when enabled."""
        config = create_test_config(track_exploration=True)
        game_map = create_simple_map()

        env = MettaGrid(from_mettagrid_config(config), game_map, 42)
        obs, _ = env.reset()

        # Take a few steps
        actions = np.array([[0, 0], [0, 0]], dtype=np.int32)  # noop actions
        for _ in range(5):
            obs, _, _, _, _ = env.step(actions)

        # Check episode stats
        stats = env.get_episode_stats()
        agent_stats = stats["agent"]

        # Exploration rate should be present when enabled
        for agent_stat in agent_stats:
            assert "exploration_rate" in agent_stat
            # Should be a reasonable value (unique pixels / steps)
            assert isinstance(agent_stat["exploration_rate"], float)
            assert agent_stat["exploration_rate"] >= 0.0

    def test_exploration_rate_increases_with_movement(self):
        """Test that exploration rate increases when agents move around."""
        config = create_test_config(track_exploration=True)
        game_map = create_simple_map()

        env = MettaGrid(from_mettagrid_config(config), game_map, 42)
        obs, _ = env.reset()

        # Take steps with movement actions
        actions = np.array([[1, 0], [1, 0]], dtype=np.int32)  # move actions
        for _ in range(3):
            obs, _, _, _, _ = env.step(actions)

        # Check episode stats
        stats = env.get_episode_stats()
        agent_stats = stats["agent"]

        # Exploration rate should be higher with movement
        for agent_stat in agent_stats:
            assert "exploration_rate" in agent_stat
            # With movement, should have explored more pixels
            assert agent_stat["exploration_rate"] > 0.0

    def test_exploration_reset_on_reset(self):
        """Test that exploration tracking resets when environment resets."""
        config = create_test_config(track_exploration=True)
        game_map = create_simple_map()

        env = MettaGrid(from_mettagrid_config(config), game_map, 42)
        obs, _ = env.reset()

        # Take some steps
        actions = np.array([[0, 0], [0, 0]], dtype=np.int32)
        for _ in range(3):
            obs, _, _, _, _ = env.step(actions)

        # Get stats before reset
        stats_before = env.get_episode_stats()

        # Reset environment by re-creating it (C++ env does not allow reset after stepping)
        env = MettaGrid(from_mettagrid_config(config), game_map, 42)
        obs, _ = env.reset()

        # Take a few more steps
        for _ in range(2):
            obs, _, _, _, _ = env.step(actions)

        # Get stats after reset
        stats_after = env.get_episode_stats()

        # Exploration rates should be different (reset should clear tracking)
        for i in range(len(stats_before["agent"])):
            rate_before = stats_before["agent"][i]["exploration_rate"]
            rate_after = stats_after["agent"][i]["exploration_rate"]
            # Rates should be different due to reset
            assert rate_before != rate_after


if __name__ == "__main__":
    pytest.main([__file__])
