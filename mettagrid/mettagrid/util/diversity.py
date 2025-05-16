import math

import numpy as np


def calculate_diversity_bonus(
    episode_rewards: np.ndarray,
    agent_groups: np.ndarray,
    similarity_coef: float,
    diversity_coef: float,
) -> np.ndarray:
    """Calculate diversity bonus for each agent based on their rewards and group.

    The bonus encourages agents to be similar to their own group but different from other groups.
    For each agent:
    1. Calculate normalized distance to own group mean
    2. Calculate normalized distance to each other group's mean
    3. Convert distances to similarity and diversity scores
    4. Combine scores with coefficients to get final bonus

    Args:
        episode_rewards: Array of rewards for each agent
        agent_groups: Array of group IDs for each agent
        similarity_coef: Coefficient for within-group similarity
        diversity_coef: Coefficient for between-group diversity

    Returns:
        Array of bonus values to add to each agent's reward
    """
    # Get number of agents and their group IDs
    num_agents = len(agent_groups)
    group_ids = agent_groups
    unique_groups = np.unique(group_ids)

    # Calculate mean and standard deviation for each group
    group_means = {g: np.mean(episode_rewards[group_ids == g]) for g in unique_groups}
    group_stds = {g: np.std(episode_rewards[group_ids == g]) + 1e-6 for g in unique_groups}

    # Initialize bonus array (0 means no bonus)
    diversity_bonuses = np.zeros_like(episode_rewards)

    # Calculate bonus for each agent
    for agent_idx in range(num_agents):
        group_id = group_ids[agent_idx]
        agent_reward = episode_rewards[agent_idx]

        # Get statistics for agent's own group
        own_mean = group_means[group_id]
        own_std = group_stds[group_id]

        # Calculate normalized distance to own group
        norm_own_distance = abs(agent_reward - own_mean) / own_std
        similarity_score = math.exp(-norm_own_distance)

        # Calculate diversity scores for each other group
        diversity_scores = []
        for other_group in unique_groups:
            if other_group != group_id:
                other_mean = group_means[other_group]
                other_std = group_stds[other_group]
                norm_other_distance = abs(agent_reward - other_mean) / other_std
                diversity_scores.append(1 - math.exp(-norm_other_distance))

        # Average the diversity scores across other groups
        diversity_score = np.mean(diversity_scores) if diversity_scores else 0

        # Calculate final bonus (now additive instead of multiplicative)
        diversity_bonuses[agent_idx] = similarity_coef * similarity_score + diversity_coef * diversity_score

    return diversity_bonuses
