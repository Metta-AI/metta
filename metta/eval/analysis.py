import fnmatch
import logging
from typing import Dict, List, Optional

from tabulate import tabulate

from metta.agent.metta_agent import MettaAgent
from metta.eval.analysis_config import AnalysisConfig
from metta.eval.eval_stats_db import EvalStatsDB
from mettagrid.util.file import local_copy

logger = logging.getLogger(__name__)


def analyze(agent: MettaAgent, config: AnalysisConfig) -> None:
    logger.info(f"Analyzing policy: {agent.uri}")
    logger.info(f"Using eval DB: {config.eval_db_uri}")

    with local_copy(config.eval_db_uri) as local_path:
        stats_db = EvalStatsDB(local_path)

        # Check if we have any data for this policy
        sample_count = stats_db.sample_count(agent, config.suite)
        if sample_count == 0:
            logger.warning(f"No samples found for policy: {agent.key}:v{agent.version}")
            return
        logger.info(f"Total sample count for specified policy/suite: {sample_count}")

        # Get available metrics and let user select which ones to analyze
        available_metrics = get_available_metrics(stats_db, agent)
        selected_metrics = config.metrics if config.metrics else available_metrics

        logger.info(f"Available metrics: {available_metrics}")
        logger.info(f"Selected metrics: {selected_metrics}")

        # Get metrics data and print table
        metrics_data = get_metrics_data(stats_db, agent, selected_metrics, config.suite)
        print_metrics_table(metrics_data, agent)


# --------------------------------------------------------------------------- #
#   helpers                                                                   #
# --------------------------------------------------------------------------- #
def get_available_metrics(stats_db: EvalStatsDB, agent: MettaAgent) -> List[str]:
    policy_key, policy_version = agent.key_and_version()
    try:
        result = stats_db.execute_query(
            """
            SELECT DISTINCT metric
            FROM episode_data
            WHERE policy_key = ? AND policy_version = ?
            ORDER BY metric
            """,
            (policy_key, policy_version),
        )
        return [row[0] for row in result]
    except Exception as e:
        logger.error(f"Error getting available metrics: {e}")
        return []


def filter_metrics(available_metrics: List[str], patterns: List[str]) -> List[str]:
    if not patterns or patterns == ["*"]:
        return available_metrics
    selected = []
    for pattern in patterns:
        selected.extend(m for m in available_metrics if fnmatch.fnmatch(m, pattern))
    return list(dict.fromkeys(selected))  # dedupe, preserve order


def get_metrics_data(
    stats_db: EvalStatsDB,
    agent: MettaAgent,
    metrics: List[str],
    suite_filter: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Return {metric: {"mean": μ, "std": σ,
                     "count": K_recorded,
                     "samples": N_potential}}
        • μ, σ are normalised (missing values = 0).
        • K_recorded  – rows in policy_simulation_agent_metrics.
        • N_potential – total agent-episode pairs for that filter.
    """
    policy_key, policy_version = agent.key_and_version()
    filter_condition = None
    if suite_filter:
        filter_condition = f"sim_name LIKE '%{suite_filter}%'"

    data: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        try:
            mean = stats_db.get_average_metric_by_filter(m, agent, filter_condition)
            if mean is None:
                continue
            std = stats_db.get_std_metric_by_filter(m, agent, filter_condition) or 0.0

            k_recorded = stats_db.count_metric_agents(policy_key, policy_version, m, filter_condition)
            n_potential = stats_db.potential_samples_for_metric(policy_key, policy_version, filter_condition)

            data[m] = {
                "mean": mean,
                "std": std,
                "count": k_recorded,
                "samples": n_potential,
            }
        except Exception as e:
            logger.error(f"Error getting data for metric {m}: {e}")
            continue
    return data


def print_metrics_table(metrics_data: Dict[str, Dict[str, float]], agent: MettaAgent) -> None:
    logger = logging.getLogger(__name__)
    if not metrics_data:
        logger.warning(f"No metrics data available for {agent.key}:v{agent.version}")
        return

    headers = ["Metric", "Average", "Std Dev", "Metric Samples", "Agent Samples"]
    rows = [
        [
            metric,
            f"{stats['mean']:.4f}",
            f"{stats['std']:.4f}",
            str(int(stats["count"])),
            str(int(stats["samples"])),
        ]
        for metric, stats in metrics_data.items()
    ]

    logger.info(f"\nMetrics for policy: {agent.uri}\n")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    logger.info("")
