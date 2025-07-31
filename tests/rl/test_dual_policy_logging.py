"""Test dual-policy logging functionality."""

from metta.rl.rollout import process_dual_policy_stats


def test_process_dual_policy_stats():
    """Test that dual-policy stats processing works correctly."""

    # Mock info structure with episode rewards and agent stats
    raw_infos = [
        {
            "agent": {"heart": 5.0},  # Average hearts per agent
            "game": {"score": 100},
            "episode_rewards": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # 6 agents total
        }
    ]

    # Process with 50% training agents (3 training, 3 NPC)
    process_dual_policy_stats(
        raw_infos=raw_infos,
        training_agents_pct=0.5,
        num_agents_per_env=6,
        num_envs=1,
    )

    # Check that dual_policy section was added
    assert "dual_policy" in raw_infos[0]
    dual_policy = raw_infos[0]["dual_policy"]

    # Check reward statistics
    assert dual_policy["training_reward_sum"] == 6.0  # 1+2+3
    assert dual_policy["training_reward_mean"] == 2.0  # 6/3
    assert dual_policy["training_reward_count"] == 3
    assert dual_policy["npc_reward_sum"] == 15.0  # 4+5+6
    assert dual_policy["npc_reward_mean"] == 5.0  # 15/3
    assert dual_policy["npc_reward_count"] == 3
    assert dual_policy["combined_reward_sum"] == 21.0  # 6+15
    assert dual_policy["combined_reward_mean"] == 3.5  # 21/6
    assert dual_policy["total_agent_count"] == 6

    # Check hearts statistics (estimated based on percentage)
    assert dual_policy["training_hearts"] == 15.0  # 5*6*0.5
    assert dual_policy["npc_hearts"] == 15.0  # 5*6*0.5
    assert dual_policy["combined_hearts"] == 30.0  # 5*6
    assert dual_policy["training_hearts_per_agent"] == 5.0  # 15/3
    assert dual_policy["npc_hearts_per_agent"] == 5.0  # 15/3
    assert dual_policy["combined_hearts_per_agent"] == 5.0  # 30/6


def test_process_dual_policy_stats_no_npc():
    """Test dual-policy stats when there are no NPC agents."""

    raw_infos = [
        {
            "agent": {"heart": 3.0},
            "game": {"score": 50},
            "episode_rewards": [1.0, 2.0, 3.0],  # 3 agents total
        }
    ]

    # Process with 100% training agents (all agents are training)
    process_dual_policy_stats(
        raw_infos=raw_infos,
        training_agents_pct=1.0,
        num_agents_per_env=3,
        num_envs=1,
    )

    dual_policy = raw_infos[0]["dual_policy"]

    # Check reward statistics
    assert dual_policy["training_reward_sum"] == 6.0  # 1+2+3
    assert dual_policy["training_reward_mean"] == 2.0  # 6/3
    assert dual_policy["training_reward_count"] == 3
    assert dual_policy["npc_reward_sum"] == 0.0
    assert dual_policy["npc_reward_mean"] == 0.0
    assert dual_policy["npc_reward_count"] == 0
    assert dual_policy["combined_reward_sum"] == 6.0
    assert dual_policy["combined_reward_mean"] == 2.0
    assert dual_policy["total_agent_count"] == 3


def test_process_dual_policy_stats_multiple_envs():
    """Test dual-policy stats with multiple environments."""

    raw_infos = [
        {
            "agent": {"heart": 4.0},
            "game": {"score": 75},
            "episode_rewards": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # 2 envs, 4 agents each
        }
    ]

    # Process with 50% training agents (2 training, 2 NPC per env)
    process_dual_policy_stats(
        raw_infos=raw_infos,
        training_agents_pct=0.5,
        num_agents_per_env=4,
        num_envs=2,
    )

    dual_policy = raw_infos[0]["dual_policy"]

    # Check reward statistics
    # For 2 envs with 4 agents each, 50% training = 2 training agents per env
    # Training agents: [1,2] from env 0, [5,6] from env 1 = 1+2+5+6 = 14
    assert dual_policy["training_reward_sum"] == 14.0  # (1+2) + (5+6)
    assert dual_policy["training_reward_mean"] == 3.5  # 14/4
    assert dual_policy["training_reward_count"] == 4
    # NPC agents: [3,4] from env 0, [7,8] from env 1 = 3+4+7+8 = 22
    assert dual_policy["npc_reward_sum"] == 22.0  # (3+4) + (7+8)
    assert dual_policy["npc_reward_mean"] == 5.5  # 22/4
    assert dual_policy["npc_reward_count"] == 4
    assert dual_policy["combined_reward_sum"] == 36.0  # 14+22
    assert dual_policy["combined_reward_mean"] == 4.5  # 36/8
    assert dual_policy["total_agent_count"] == 8


def test_process_dual_policy_stats_no_episode_rewards():
    """Test dual-policy stats when episode_rewards are not available."""

    raw_infos = [
        {
            "agent": {"heart": 2.0},
            "game": {"score": 25},
            # No episode_rewards
        }
    ]

    # Process with 50% training agents
    process_dual_policy_stats(
        raw_infos=raw_infos,
        training_agents_pct=0.5,
        num_agents_per_env=4,
        num_envs=1,
    )

    # Should still have hearts but no rewards
    assert "dual_policy" in raw_infos[0]
    dual_policy = raw_infos[0]["dual_policy"]

    # Should have hearts but no reward stats
    assert "training_hearts" in dual_policy
    assert "npc_hearts" in dual_policy
    assert "combined_hearts" in dual_policy
    assert "training_reward_sum" not in dual_policy
