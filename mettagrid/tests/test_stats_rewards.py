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
        "inventory_item_names": ["laser", "armor"],
        "actions": {
            "noop": {"enabled": True},
            "move": {"enabled": True},
            "rotate": {"enabled": True},
            "attack": {"enabled": True, "consumed_resources": {"laser": 1}, "defense_resources": {"armor": 1}},
        },
        "groups": {
            "red": {
                "id": 0,
                "props": {
                    "rewards": {
                        "inventory": {},  # No inventory rewards
                        "stats": {
                            "action.move.success": 0.1,  # 0.1 reward per successful move
                            "action.attack.success": 1.0,  # 1.0 reward per successful attack
                            "action.attack.success_max": 5.0,  # Max 5.0 total reward from attacks
                        },
                    }
                },
            },
            "blue": {"id": 1, "props": {}},
        },
        "objects": {
            "wall": {"type_id": 1},
        },
        "agent": {
            "default_resource_limit": 10,
            "initial_inventory": {"laser": 10, "armor": 5},  # Start with resources for combat
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
        rotate_idx = action_names.index("rotate") if "rotate" in action_names else None

        # Rotate to face down (south) first to ensure we can move
        if rotate_idx is not None:
            rotate_action = np.array([[rotate_idx, 1]], dtype=np.int32)  # Rotate to face down
            env.step(rotate_action)

        # Agent should get 0.1 reward per successful move
        actions = np.array([[move_idx, 0]], dtype=np.int32)  # Move forward

        # Execute move
        obs, rewards, terminals, truncations, info = env.step(actions)

        # Check that we got the movement reward
        assert rewards[0] == pytest.approx(0.1), f"Expected 0.1 reward for move, got {rewards[0]}"

    def test_attack_stats_reward(self):
        """Test that agents receive rewards for successful attacks."""
        # Create environment with 2 agents for combat
        env = create_stats_reward_test_env(num_agents=2)
        obs, _ = env.reset()

        # Get action indices
        action_names = env.action_names()
        attack_idx = action_names.index("attack")
        rotate_idx = action_names.index("rotate")
        noop_idx = action_names.index("noop")

        # Red agent is at (1,1) facing right, blue agent is at (1,3)
        # Red agent needs to rotate to face right to attack blue agent
        rotate_action = np.array([[rotate_idx, 2], [noop_idx, 0]], dtype=np.int32)  # Red face right, Blue noop
        env.step(rotate_action)

        # Attack with red agent (index 0)
        attack_action = np.array([[attack_idx, 0], [noop_idx, 0]], dtype=np.int32)
        obs, rewards, terminals, truncations, info = env.step(attack_action)

        # Check attack success
        action_success = env.action_success()
        assert action_success[0], "Attack should succeed"
        assert rewards[0] == pytest.approx(1.0), f"Expected 1.0 reward for successful attack, got {rewards[0]}"

    def test_attack_reward_maximum(self):
        """Test that attack rewards respect the maximum limit."""
        # Create environment with 2 agents for combat
        env = create_stats_reward_test_env(num_agents=2)
        obs, _ = env.reset()

        # Get action indices
        action_names = env.action_names()
        attack_idx = action_names.index("attack")
        rotate_idx = action_names.index("rotate")
        noop_idx = action_names.index("noop")

        # Rotate red agent to face right
        rotate_action = np.array([[rotate_idx, 2], [noop_idx, 0]], dtype=np.int32)
        env.step(rotate_action)

        total_attack_reward = 0.0
        successful_attacks = 0

        # Keep attacking until we hit the limit (should be 5.0 max)
        for i in range(10):
            attack_action = np.array([[attack_idx, 0], [noop_idx, 0]], dtype=np.int32)
            obs, rewards, terminals, truncations, info = env.step(attack_action)
            
            if rewards[0] > 0:
                total_attack_reward += rewards[0]
                successful_attacks += 1

        # Should not exceed the max of 5.0
        assert total_attack_reward <= 5.0, f"Total attack rewards should not exceed 5.0, got {total_attack_reward}"
        assert total_attack_reward >= 4.0, f"Total attack rewards should be close to 5.0, got {total_attack_reward}"
        assert successful_attacks >= 4, f"Should have at least 4 successful attacks before hitting limit"

    def test_combined_move_and_attack_rewards(self):
        """Test that move and attack stat rewards work together."""
        # Create environment with 2 agents
        env = create_stats_reward_test_env(num_agents=2)
        obs, _ = env.reset()

        # Get action indices
        action_names = env.action_names()
        move_idx = action_names.index("move")
        attack_idx = action_names.index("attack")
        rotate_idx = action_names.index("rotate")
        noop_idx = action_names.index("noop")

        total_reward = 0.0

        # Move down (0.1 reward)
        rotate_action = np.array([[rotate_idx, 1], [noop_idx, 0]], dtype=np.int32)  # Red face down
        env.step(rotate_action)
        
        move_action = np.array([[move_idx, 0], [noop_idx, 0]], dtype=np.int32)
        obs, rewards, _, _, _ = env.step(move_action)
        total_reward += rewards[0]
        assert rewards[0] == pytest.approx(0.1), f"Expected 0.1 reward for move, got {rewards[0]}"

        # Rotate to face right
        rotate_action = np.array([[rotate_idx, 2], [noop_idx, 0]], dtype=np.int32)  # Red face right
        env.step(rotate_action)

        # Attack (1.0 reward)
        attack_action = np.array([[attack_idx, 0], [noop_idx, 0]], dtype=np.int32)
        obs, rewards, _, _, _ = env.step(attack_action)
        total_reward += rewards[0]
        assert rewards[0] == pytest.approx(1.0), f"Expected 1.0 reward for attack, got {rewards[0]}"

        # Move right (0.1 reward)
        move_action = np.array([[move_idx, 0], [noop_idx, 0]], dtype=np.int32)
        obs, rewards, _, _, _ = env.step(move_action)
        total_reward += rewards[0]
        assert rewards[0] == pytest.approx(0.1), f"Expected 0.1 reward for second move, got {rewards[0]}"

        # Total should be 0.1 + 1.0 + 0.1 = 1.2
        expected_total = 1.2
        assert total_reward == pytest.approx(expected_total), (
            f"Expected {expected_total} total reward, got {total_reward}"
        )

    def test_combined_stats_rewards(self):
        """Test that multiple stat rewards can be earned together."""
        env = create_stats_reward_test_env(num_agents=1)
        obs, _ = env.reset()

        # Get action indices
        action_names = env.action_names()
        move_idx = action_names.index("move")
        rotate_idx = action_names.index("rotate")

        # Do several moves to accumulate move rewards
        total_reward = 0.0
        successful_moves = 0

        # Move down
        env.step(np.array([[rotate_idx, 1]], dtype=np.int32))  # Face down
        _, rewards, _, _, _ = env.step(np.array([[move_idx, 0]], dtype=np.int32))
        if rewards[0] > 0:
            successful_moves += 1
            total_reward += rewards[0]

        # Move right
        env.step(np.array([[rotate_idx, 2]], dtype=np.int32))  # Face right
        _, rewards, _, _, _ = env.step(np.array([[move_idx, 0]], dtype=np.int32))
        if rewards[0] > 0:
            successful_moves += 1
            total_reward += rewards[0]

        # Move left
        env.step(np.array([[rotate_idx, 3]], dtype=np.int32))  # Face left
        _, rewards, _, _, _ = env.step(np.array([[move_idx, 0]], dtype=np.int32))
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
