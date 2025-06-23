import math

import numpy as np


def calculate_diversity_bonus(
    episode_rewards: np.ndarray,
    agent_groups: np.ndarray,
    similarity_coef: float,
    diversity_coef: float,
) -> np.ndarray:
    """Calculate diversity scaling factor for each agent based on their rewards and group.

    The scaling factor encourages agents to be similar to their own group but different from other groups.
    For each agent:
    1. Calculate normalized distance to own group mean
    2. Calculate normalized distance to each other group's mean
    3. Convert distances to similarity and diversity scores
    4. Combine scores with coefficients to get final scaling factor

    Args:
        episode_rewards: Array of rewards for each agent
        agent_groups: Array of group IDs for each agent
        similarity_coef: Coefficient for within-group similarity
        diversity_coef: Coefficient for between-group diversity

    Returns:
        Array of scaling factors to multiply each agent's reward by
    """
    # Get number of agents and their group IDs
    num_agents = len(agent_groups)
    group_ids = agent_groups
    unique_groups = np.unique(group_ids)

    # Calculate mean and standard deviation for each group
    group_means = {g: np.mean(episode_rewards[group_ids == g]) for g in unique_groups}
    group_stds = {g: np.std(episode_rewards[group_ids == g]) + 1e-6 for g in unique_groups}
    # Add 1e-6 to prevent division by zero if group rewards are identical e.g. single agent in group

    # Initialize scaling factors array
    diversity_factors = np.ones_like(episode_rewards, dtype=float)

    # Calculate scaling factor for each agent
    for agent_idx in range(num_agents):
        agent_group_id = group_ids[agent_idx]
        agent_reward = episode_rewards[agent_idx]

        # Get statistics for agent's own group
        agent_group_mean = group_means[agent_group_id]
        agent_group_std = group_stds[agent_group_id]

        # Calculate normalized distance to own group
        norm_distance_to_own_group = abs(agent_reward - agent_group_mean) / agent_group_std
        similarity_score = math.exp(-norm_distance_to_own_group)

        # Calculate diversity scores for each other group
        diversity_scores = []
        for other_group in unique_groups:
            if other_group != agent_group_id:
                other_group_mean = group_means[other_group]
                other_group_std = group_stds[other_group]
                norm_distance_to_other_group = abs(agent_reward - other_group_mean) / other_group_std
                diversity_scores.append(1 - math.exp(-norm_distance_to_other_group))

        # Average the diversity scores across other groups
        diversity_score = np.mean(diversity_scores) if diversity_scores else 0

        # Calculate final scaling factor (multiplicative)
        diversity_factors[agent_idx] = 1 + similarity_coef * similarity_score + diversity_coef * diversity_score

    return diversity_factors
