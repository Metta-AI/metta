"""Test stats-based rewards functionality."""

import numpy as np
import pytest

from mettagrid.config.mettagrid_c_config import from_mettagrid_config
from mettagrid.mettagrid_c import MettaGrid
from mettagrid.test_support.orientation import Orientation

# Test constants
NUM_AGENTS = 1
OBS_WIDTH = 11
OBS_HEIGHT = 11
NUM_OBS_TOKENS = 200


def create_stats_reward_test_env(max_steps=50, num_agents=NUM_AGENTS):
    """Helper function to create a MettaGrid environment with stats rewards for testing."""
    if num_agents == 1:
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", ".", ".", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]
    else:
        game_map = [
            ["wall", "wall", "wall", "wall", "wall"],
            ["wall", "agent.red", ".", "agent.blue", "wall"],
            ["wall", ".", ".", ".", "wall"],
            ["wall", "agent.red", ".", "agent.blue", "wall"],
            ["wall", "wall", "wall", "wall", "wall"],
        ]

    game_config = {
        "max_steps": max_steps,
        "num_agents": num_agents,
        "obs_width": OBS_WIDTH,
        "obs_height": OBS_HEIGHT,
        "num_observation_tokens": NUM_OBS_TOKENS,
        "resource_names": ["laser", "armor"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
        },
        "agent": {
            "default_resource_limit": 10,
            "rewards": {
                "inventory": {},  # No inventory rewards
                "stats": {
                    "action.move.success": 0.1,  # 0.1 reward per successful move
                    "action.attack.success": 1.0,  # 1.0 reward per successful attack
                    "action.attack.success_max": 5.0,  # Max 5.0 total reward from attacks
                },
            },
        },
        "objects": {
            "wall": {"type_id": 1},
        },
        "allow_diagonals": True,
    }

    return MettaGrid(from_mettagrid_config(game_config), game_map, 42)


class TestStatsRewards:
    def test_move_stats_reward(self):
        """Test that agents receive rewards for successful moves."""
        env = create_stats_reward_test_env(num_agents=1)
        obs, _ = env.reset()

        # Get action indices
        action_names = env.action_names()
        move_idx = action_names.index("move")

        # Agent should get 0.1 reward per successful move
        # Move down (South)
        actions = np.array([[move_idx, Orientation.SOUTH.value]], dtype=np.int32)

        # Execute move
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check that we got the movement reward
        assert rewards[0] == pytest.approx(0.1), f"Expected 0.1 reward for move, got {rewards[0]}"

    def test_combined_stats_rewards(self):
        """Test that multiple stat rewards can be earned together."""
        env = create_stats_reward_test_env(num_agents=1)
        obs, _ = env.reset()

        # Get action indices
        action_names = env.action_names()
        move_idx = action_names.index("move")

        # Do several moves to accumulate move rewards
        total_reward = 0.0
        successful_moves = 0

        # Move down (South)
        _, rewards, _, _, _ = env.step(np.array([[move_idx, Orientation.SOUTH.value]], dtype=np.int32))
        if rewards[0] > 0:
            successful_moves += 1
            total_reward += rewards[0]

        # Move right (East)
        _, rewards, _, _, _ = env.step(np.array([[move_idx, Orientation.EAST.value]], dtype=np.int32))
        if rewards[0] > 0:
            successful_moves += 1
            total_reward += rewards[0]

        # Move left (West)
        _, rewards, _, _, _ = env.step(np.array([[move_idx, Orientation.WEST.value]], dtype=np.int32))
        if rewards[0] > 0:
            successful_moves += 1
            total_reward += rewards[0]

        # We should have at least 2 successful moves out of 3
        assert successful_moves >= 2, f"Expected at least 2 successful moves, got {successful_moves}"
        # Each successful move gives 0.1 reward
        expected_reward = successful_moves * 0.1
        assert total_reward == pytest.approx(expected_reward), (
            f"Expected {expected_reward} total reward from {successful_moves} moves, got {total_reward}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
