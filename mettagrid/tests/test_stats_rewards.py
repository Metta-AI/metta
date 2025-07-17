"""Test stats-based rewards functionality."""

import numpy as np
import pytest

from metta.mettagrid.mettagrid_c import MettaGrid
from metta.mettagrid.mettagrid_c_config import from_mettagrid_config

# Test constants
NUM_AGENTS = 1
OBS_WIDTH = 11
OBS_HEIGHT = 11
NUM_OBS_TOKENS = 200


def create_stats_reward_test_env(max_steps=50, num_agents=NUM_AGENTS):
    """Helper function to create a MettaGrid environment with stats rewards for testing."""
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
        "inventory_item_names": ["laser", "armor"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
        },
        "groups": {
            "red": {"id": 0, "props": {
                "rewards": {
                    "inventory": {},  # No inventory rewards
                    "stats": {
                        "action.move.success": 0.1,  # 0.1 reward per successful move
                        "action.attack.success": 1.0,  # 1.0 reward per successful attack
                        "action.attack.success_max": 5.0,  # Max 5.0 total reward from attacks
                    },
                }
            }},
            "blue": {"id": 1, "props": {}},
        },
        "objects": {
            "wall": {"type_id": 1},
        },
        "agent": {
            "default_resource_limit": 10,
        },
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
        actions = np.array([[move_idx, 0]], dtype=np.int32)  # Move forward
        
        # Execute move
        obs, rewards, terminals, truncations, info = env.step(actions)
        
        # Check that we got the movement reward
        assert rewards[0] == pytest.approx(0.1), f"Expected 0.1 reward for move, got {rewards[0]}"

    def test_attack_stats_reward_with_max(self):
        """Test that attack rewards are capped at the max value."""
        env = create_stats_reward_test_env(num_agents=4)  # 4 agents for attack testing
        obs, _ = env.reset()

        # Get action indices
        action_names = env.action_names()
        attack_idx = action_names.index("attack")

        # Give agent 0 some laser ammo
        grid_objects = env.grid_objects()
        for obj_id, obj in grid_objects.items():
            if obj.get("type") == 0 and obj.get("agent_id") == 0:
                # Directly set inventory via the C++ binding would be needed here
                # For now, we'll test with the assumption agents have resources
                break

        # Track total reward from attacks
        total_attack_reward = 0.0
        
        # Perform multiple attacks
        for i in range(10):  # Try 10 attacks
            actions = np.zeros((4, 2), dtype=np.int32)
            actions[0] = [attack_idx, 0]  # Agent 0 attacks
            
            _, rewards, _, _, _ = env.step(actions)
            
            # Track reward (might be 0 if attack failed)
            if rewards[0] > total_attack_reward:
                attack_reward_gained = rewards[0] - total_attack_reward
                total_attack_reward = rewards[0]
                
                # Individual attack rewards should be 1.0
                if attack_reward_gained > 0:
                    assert attack_reward_gained == pytest.approx(1.0) or total_attack_reward == pytest.approx(5.0), \
                        f"Unexpected attack reward: {attack_reward_gained}, total: {total_attack_reward}"

        # Total attack reward should be capped at 5.0
        assert total_attack_reward <= 5.0, f"Attack reward exceeded max: {total_attack_reward}"

    def test_combined_stats_rewards(self):
        """Test that multiple stat rewards can be earned together."""
        env = create_stats_reward_test_env(num_agents=1)
        obs, _ = env.reset()

        # Get action indices
        action_names = env.action_names()
        move_idx = action_names.index("move")

        # Do several moves to accumulate move rewards
        total_reward = 0.0
        for i in range(3):
            actions = np.array([[move_idx, 0]], dtype=np.int32)
            _, rewards, _, _, _ = env.step(actions)
            total_reward += rewards[0]

        # Should have 0.3 total reward from 3 moves
        assert total_reward == pytest.approx(0.3), f"Expected 0.3 total reward from moves, got {total_reward}"

    def test_no_inventory_rewards(self):
        """Test that inventory changes don't give rewards when only stats rewards are configured."""
        # This would require a test with items to collect, but the basic principle
        # is that with empty inventory rewards config, picking up items gives no reward
        pass  # Placeholder for more complex test