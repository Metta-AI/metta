import logging
import uuid

import torch

from metta.agent.policy_record import PolicyRecord
from metta.agent.policy_store import PolicyStore
from metta.app_backend.stats_client import StatsClient
from metta.eval.eval_request_config import EvalResults, EvalRewardSummary
from metta.eval.eval_stats_db import EvalStatsDB
from metta.sim.simulation_config import SimulationSuiteConfig
from metta.sim.simulation_suite import SimulationSuite


def evaluate_policy(
    *,
    policy_record: PolicyRecord,
    simulation_suite: SimulationSuiteConfig,
    device: torch.device,
    vectorization: str,
    stats_dir: str = "/tmp/stats",
    replay_dir: str | None = None,
    export_stats_db_uri: str | None = None,
    stats_epoch_id: uuid.UUID | None = None,
    wandb_policy_name: str | None = None,
    eval_task_id: uuid.UUID | None = None,
    policy_store: PolicyStore,
    stats_client: StatsClient | None,
    logger: logging.Logger,
) -> EvalResults:
    """
    Evaluate **one** policy URI (may expand to several checkpoints).
    All simulations belonging to a single checkpoint are merged into one
    *StatsDB* which is optionally exported.

    Returns:
        Dictionary containing simulation results and metrics
    """
    pr = policy_record

    # For each checkpoint of the policy, simulate
    logger.info(f"Evaluating policy {pr.uri}")
    sim = SimulationSuite(
        config=simulation_suite,
        policy_pr=pr,
        policy_store=policy_store,
        replay_dir=replay_dir,  # TODO: check
        stats_dir=stats_dir,
        device=device,
        vectorization=vectorization,
        stats_client=stats_client,
        stats_epoch_id=stats_epoch_id,
        wandb_policy_name=wandb_policy_name,
        eval_task_id=eval_task_id,
    )
    result = sim.simulate()

    eval_stats_db = EvalStatsDB.from_sim_stats_db(result.stats_db)
    logger.info("Evaluation complete for policy %s", pr.uri)
    scores = extract_scores(policy_record, simulation_suite, eval_stats_db, logger)

    if export_stats_db_uri is not None:
        logger.info("Exporting merged stats DB → %s", export_stats_db_uri)
        result.stats_db.export(export_stats_db_uri)

    # Handle replay URLs
    replay_urls: dict[str, list[str]] = {}

    if replay_dir is not None:
        # Get all replay URLs from simulation results
        if result.replay_urls:
            replay_urls = result.replay_urls
            logger.info(f"Found {len(replay_urls)} replay URLs from simulations")

    results = EvalResults(
        scores=scores,
        replay_urls=replay_urls,
    )

    return results


def extract_scores(
    policy_record: PolicyRecord,
    simulation_suite: SimulationSuiteConfig,
    stats_db: EvalStatsDB,
    logger: logging.Logger,
) -> EvalRewardSummary:
    categories: set[str] = set()
    for sim_name in simulation_suite.simulations.keys():
        categories.add(sim_name.split("/")[0])

    category_scores: dict[str, float] = {}
    for category in categories:
        score = stats_db.get_average_metric_by_filter("reward", policy_record, f"sim_name LIKE '%{category}%'")
        logger.info(f"{category} score: {score}")
        if score is None:
            continue
        category_scores[category] = score

    per_sim_scores: dict[tuple[str, str], float] = {}
    all_scores = stats_db.simulation_scores(policy_record, "reward")
    for (_, sim_name, _), score in all_scores.items():
        category = sim_name.split("/")[0]
        sim_short_name = sim_name.split("/")[-1]
        per_sim_scores[(category, sim_short_name)] = score

    # Extract exploration rates per simulation
    per_sim_exploration_rates: dict[tuple[str, str], float] = {}
    try:
        all_exploration_rates = stats_db.simulation_scores(policy_record, "exploration_rate")

        # Log exploration rates prominently for each environment
        logger.info("=== Exploration Rates by Environment ===")
        for (_, sim_name, _), rate in all_exploration_rates.items():
            category = sim_name.split("/")[0]
            sim_short_name = sim_name.split("/")[-1]
            per_sim_exploration_rates[(category, sim_short_name)] = rate
            logger.info(f"  {category}/{sim_short_name}: {rate:.4f}")

        # Log summary statistics
        if per_sim_exploration_rates:
            rates = list(per_sim_exploration_rates.values())
            avg_rate = sum(rates) / len(rates)
            min_rate = min(rates)
            max_rate = max(rates)
            logger.info(f"Exploration Rate Summary: avg={avg_rate:.4f}, min={min_rate:.4f}, max={max_rate:.4f}")
        logger.info("=========================================")

    except Exception as e:
        logger.warning(f"Could not extract exploration rates: {e}")
        # Exploration rates may not be available if tracking is disabled

    return EvalRewardSummary(
        category_scores=category_scores,
        simulation_scores=per_sim_scores,
        exploration_rates=per_sim_exploration_rates,
    )
