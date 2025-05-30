import numpy as np

from mettagrid.util.diversity import calculate_diversity_bonus


def test_calculate_diversity_bonus_simple_case():
    episode_rewards = np.array([10.0, 12.0, 20.0, 22.0])
    agent_groups = np.array([0, 0, 1, 1])
    similarity_coef = 0.1
    diversity_coef = 0.2

    actual_factors = calculate_diversity_bonus(episode_rewards, agent_groups, similarity_coef, diversity_coef)

    # Group 0: rewards [10, 12], mean = 11, std = 1
    # Group 1: rewards [20, 22], mean = 21, std = 1
    exp_m_1 = np.exp(-1)
    exp_m_9 = np.exp(-9)
    exp_m_11 = np.exp(-11)

    factor_0 = 1 + similarity_coef * exp_m_1 + diversity_coef * (1 - exp_m_11)
    factor_1 = 1 + similarity_coef * exp_m_1 + diversity_coef * (1 - exp_m_9)
    factor_2 = 1 + similarity_coef * exp_m_1 + diversity_coef * (1 - exp_m_9)
    factor_3 = 1 + similarity_coef * exp_m_1 + diversity_coef * (1 - exp_m_11)

    expected_factors = np.array([factor_0, factor_1, factor_2, factor_3])
    np.testing.assert_allclose(actual_factors, expected_factors, rtol=1e-6)


def test_calculate_diversity_bonus_single_group():
    episode_rewards = np.array([10.0, 12.0, 11.0])
    agent_groups = np.array([0, 0, 0])
    similarity_coef = 0.1
    diversity_coef = 0.2

    # Group 0: rewards [10, 12, 11], mean = 11, std = sqrt(2/3)
    std_g0 = np.sqrt(2 / 3)

    sim_score_agent0 = np.exp(-1 / std_g0)
    sim_score_agent1 = np.exp(-1 / std_g0)
    sim_score_agent2 = 1.0

    expected_factors = np.array(
        [
            1 + similarity_coef * sim_score_agent0,  # Agent 0, diversity score is 0
            1 + similarity_coef * sim_score_agent1,  # Agent 1, diversity score is 0
            1 + similarity_coef * sim_score_agent2,  # Agent 2, diversity score is 0
        ]
    )

    actual_factors = calculate_diversity_bonus(episode_rewards, agent_groups, similarity_coef, diversity_coef)
    np.testing.assert_allclose(actual_factors, expected_factors, rtol=1e-6)


def test_calculate_diversity_bonus_zero_coefficients():
    episode_rewards = np.array([10.0, 12.0, 20.0, 22.0])
    agent_groups = np.array([0, 0, 1, 1])
    similarity_coef = 0.0
    diversity_coef = 0.0

    # If both coeffs are 0, the factor should be 1 for all agents
    expected_factors = np.array([1.0, 1.0, 1.0, 1.0])

    actual_factors = calculate_diversity_bonus(episode_rewards, agent_groups, similarity_coef, diversity_coef)
    np.testing.assert_allclose(actual_factors, expected_factors, rtol=1e-6)


def test_calculate_diversity_bonus_identical_rewards_in_group():
    episode_rewards = np.array([10.0, 10.0, 20.0, 20.0])
    agent_groups = np.array([0, 0, 1, 1])
    similarity_coef = 0.1
    diversity_coef = 0.2

    # Group 0: rewards [10, 10], mean = 10, std = 1e-6 (due to epsilon)
    # Group 1: rewards [20, 20], mean = 20, std = 1e-6
    # Similarity score for all agents is 1 (exp(0)).
    # Diversity value for group 0 agents (to group 1) is 1 - exp(-10/1e-6) -> approx 1.
    # Diversity value for group 1 agents (to group 0) is 1 - exp(-10/1e-6) -> approx 1.
    sim_score = 1.0
    div_val = 1 - np.exp(-10 / 1e-6)  # Effectively 1.0

    expected_factor_val = 1 + similarity_coef * sim_score + diversity_coef * div_val
    expected_factors = np.full_like(episode_rewards, expected_factor_val)

    actual_factors = calculate_diversity_bonus(episode_rewards, agent_groups, similarity_coef, diversity_coef)
    np.testing.assert_allclose(actual_factors, expected_factors, rtol=1e-6)


def test_calculate_diversity_bonus_three_groups():
    episode_rewards = np.array(
        [
            10.0,
            15.0,  # Group 0
            20.0,
            25.0,  # Group 1
            30.0,
            35.0,
        ]
    )  # Group 2
    agent_groups = np.array([0, 0, 1, 1, 2, 2])
    similarity_coef = 0.1
    diversity_coef = 0.2

    # Group means and stds
    m_g0, s_g0 = np.mean(episode_rewards[agent_groups == 0]), np.std(episode_rewards[agent_groups == 0]) + 1e-6
    m_g1, s_g1 = np.mean(episode_rewards[agent_groups == 1]), np.std(episode_rewards[agent_groups == 1]) + 1e-6
    m_g2, s_g2 = np.mean(episode_rewards[agent_groups == 2]), np.std(episode_rewards[agent_groups == 2]) + 1e-6

    factors = []

    # Agent 0 (reward 10, group 0)
    r0 = episode_rewards[0]
    sim_0 = np.exp(-abs(r0 - m_g0) / s_g0)
    div_0_g1 = 1 - np.exp(-abs(r0 - m_g1) / s_g1)
    div_0_g2 = 1 - np.exp(-abs(r0 - m_g2) / s_g2)
    factors.append(1 + similarity_coef * sim_0 + diversity_coef * np.mean([div_0_g1, div_0_g2]))

    # Agent 1 (reward 15, group 0)
    r1 = episode_rewards[1]
    sim_1 = np.exp(-abs(r1 - m_g0) / s_g0)
    div_1_g1 = 1 - np.exp(-abs(r1 - m_g1) / s_g1)
    div_1_g2 = 1 - np.exp(-abs(r1 - m_g2) / s_g2)
    factors.append(1 + similarity_coef * sim_1 + diversity_coef * np.mean([div_1_g1, div_1_g2]))

    # Agent 2 (reward 20, group 1)
    r2 = episode_rewards[2]
    sim_2 = np.exp(-abs(r2 - m_g1) / s_g1)
    div_2_g0 = 1 - np.exp(-abs(r2 - m_g0) / s_g0)
    div_2_g2 = 1 - np.exp(-abs(r2 - m_g2) / s_g2)
    factors.append(1 + similarity_coef * sim_2 + diversity_coef * np.mean([div_2_g0, div_2_g2]))

    # Agent 3 (reward 25, group 1)
    r3 = episode_rewards[3]
    sim_3 = np.exp(-abs(r3 - m_g1) / s_g1)
    div_3_g0 = 1 - np.exp(-abs(r3 - m_g0) / s_g0)
    div_3_g2 = 1 - np.exp(-abs(r3 - m_g2) / s_g2)
    factors.append(1 + similarity_coef * sim_3 + diversity_coef * np.mean([div_3_g0, div_3_g2]))

    # Agent 4 (reward 30, group 2)
    r4 = episode_rewards[4]
    sim_4 = np.exp(-abs(r4 - m_g2) / s_g2)
    div_4_g0 = 1 - np.exp(-abs(r4 - m_g0) / s_g0)
    div_4_g1 = 1 - np.exp(-abs(r4 - m_g1) / s_g1)
    factors.append(1 + similarity_coef * sim_4 + diversity_coef * np.mean([div_4_g0, div_4_g1]))

    # Agent 5 (reward 35, group 2)
    r5 = episode_rewards[5]
    sim_5 = np.exp(-abs(r5 - m_g2) / s_g2)
    div_5_g0 = 1 - np.exp(-abs(r5 - m_g0) / s_g0)
    div_5_g1 = 1 - np.exp(-abs(r5 - m_g1) / s_g1)
    factors.append(1 + similarity_coef * sim_5 + diversity_coef * np.mean([div_5_g0, div_5_g1]))

    expected_factors = np.array(factors)
    actual_factors = calculate_diversity_bonus(episode_rewards, agent_groups, similarity_coef, diversity_coef)
    np.testing.assert_allclose(actual_factors, expected_factors, rtol=1e-6)


def test_zero_total_diversity_score_when_agent_reward_equals_all_other_group_means():
    """Agent's diversity score is 0 if its reward matches all other group means."""
    episode_rewards = np.array(
        [
            25.0,
            15.0,  # Group 0: Agent 0 (reward 25), Agent 1 (reward 15). Mean G0 = 20. Std G0 = 5.
            20.0,
            30.0,  # Group 1: Agents 2,3. Mean G1 = 25. Std G1 = 5.
            22.0,
            28.0,  # Group 2: Agents 4,5. Mean G2 = 25. Std G2 = 3.
        ]
    )
    agent_groups = np.array([0, 0, 1, 1, 2, 2])
    similarity_coef = 0.1
    diversity_coef = 0.2

    actual_factors = calculate_diversity_bonus(episode_rewards, agent_groups, similarity_coef, diversity_coef)

    agent0_reward = episode_rewards[0]
    group0_rewards = episode_rewards[agent_groups == 0]
    mean_g0 = np.mean(group0_rewards)
    std_g0 = np.std(group0_rewards) + 1e-6

    norm_own_distance_agent0 = abs(agent0_reward - mean_g0) / std_g0
    similarity_score_agent0 = np.exp(-norm_own_distance_agent0)

    # Diversity score for Agent 0 is 0 because its reward (25) equals means of G1 (25) and G2 (25).
    expected_factor_agent0 = 1 + similarity_coef * similarity_score_agent0  # + diversity_coef * 0
    np.testing.assert_allclose(actual_factors[0], expected_factor_agent0, rtol=1e-6)
