"""Policy evaluation functions."""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite


@dataclass
class EvaluationResult:
    """Results from policy evaluation."""

    overall_score: float
    category_scores: Dict[str, float]
    individual_scores: Dict[str, float]
    metadata: Dict[str, Any]


def evaluate_policy(
    agent,
    sim_suite_config: SimulationSuiteConfig,
    device: str = "cuda",
    num_episodes: int = 10,
    render: bool = False,
) -> EvaluationResult:
    """Evaluate a policy on a simulation suite.

    Args:
        agent: The policy/agent to evaluate
        sim_suite_config: Configuration for evaluation environments
        device: Device to run evaluation on
        num_episodes: Number of episodes per environment
        render: Whether to render episodes

    Returns:
        EvaluationResult with scores and metadata
    """
    # Create simulation suite
    from metta.agent.policy_store import PolicyRecord

    # Create a dummy policy record for the agent
    policy_record = PolicyRecord(
        name="eval_policy",
        uri="eval://current",
        policy=lambda: agent,
        metadata={},
    )

    sim_suite = SimulationSuite(
        config=sim_suite_config,
        policy_pr=policy_record,
        policy_store=None,
        device=device,
        vectorization="serial",  # Use serial for eval
        stats_dir="/tmp/eval_stats",
    )

    # Run simulations
    results = sim_suite.simulate()
    stats_db = results.stats_db

    # Extract scores
    all_scores = {}
    category_scores = {}

    # Get scores for each simulation
    for sim_name in sim_suite_config.simulations.keys():
        score_df = stats_db.query(
            f"SELECT AVG(value) as avg_score FROM agent_metrics "
            f"WHERE metric = 'reward' AND simulation_name = '{sim_name}'"
        )
        if len(score_df) > 0:
            all_scores[sim_name] = float(score_df.iloc[0]["avg_score"])
        else:
            all_scores[sim_name] = 0.0

        # Extract category from simulation name
        category = sim_name.split("/")[0]
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(all_scores[sim_name])

    # Compute category averages
    category_avgs = {cat: np.mean(scores) for cat, scores in category_scores.items()}

    # Compute overall score
    overall_score = np.mean(list(all_scores.values()))

    return EvaluationResult(
        overall_score=overall_score,
        category_scores=category_avgs,
        individual_scores=all_scores,
        metadata={
            "num_episodes": num_episodes,
            "device": device,
            "num_simulations": len(all_scores),
        },
    )


def evaluate_on_env(
    agent,
    env,
    num_episodes: int = 10,
    render: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate a policy on a single environment.

    Args:
        agent: The policy/agent to evaluate
        env: Environment to evaluate on
        num_episodes: Number of episodes to run
        render: Whether to render episodes
        device: Device to run on

    Returns:
        Dictionary with evaluation statistics
    """
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        # Initialize LSTM state if needed
        from metta.agent.policy_state import PolicyState

        state = PolicyState()

        while not done:
            # Get action from agent
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(device)
                actions, _, _, _, _ = agent(obs_tensor, state)
                action = actions[0].cpu().numpy()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }
