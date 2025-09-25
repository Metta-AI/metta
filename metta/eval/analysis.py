import fnmatch
import logging
from typing import Dict, List

from tabulate import tabulate

from metta.eval.analysis_config import AnalysisConfig
from metta.eval.eval_stats_db import EvalStatsDB
from metta.rl.checkpoint_manager import CheckpointManager
from metta.utils.file import local_copy


def analyze(policy_uri: str, config: AnalysisConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing policy: {policy_uri}")
    logger.info(f"Using eval DB: {config.eval_db_uri}")

    with local_copy(config.eval_db_uri) as local_path:
        stats_db = EvalStatsDB(local_path)

        sample_count = stats_db.sample_count_uri(policy_uri, sim_name=config.sim_name)
        if sample_count == 0:
            logger.warning(f"No samples found for policy: {policy_uri}")
            return
        logger.info(f"Total sample count for specified policy/suite: {sample_count}")

        available_metrics = get_available_metrics(stats_db, policy_uri)
        logger.info(f"Available metrics: {available_metrics}")

        selected_metrics = filter_metrics(available_metrics, config.metrics)
        if not selected_metrics:
            logger.warning(f"No metrics found matching patterns: {config.metrics}")
            return
        logger.info(f"Selected metrics: {selected_metrics}")

        metrics_data = get_metrics_data(stats_db, policy_uri, selected_metrics, sim_name=config.sim_name)
        print_metrics_table(metrics_data, policy_uri)


# --------------------------------------------------------------------------- #
#   helpers                                                                   #
# --------------------------------------------------------------------------- #
def get_available_metrics(stats_db: EvalStatsDB, policy_uri: str) -> List[str]:
    metadata = CheckpointManager.get_policy_metadata(policy_uri)
    pk, pv = metadata["run_name"], metadata["epoch"]
    result = stats_db.query(
        f"""
        SELECT DISTINCT metric
          FROM policy_simulation_agent_metrics
         WHERE policy_key     = '{pk}'
           AND policy_version =  {pv}
         ORDER BY metric
        """
    )
    return [] if result.empty else result["metric"].tolist()


def filter_metrics(available_metrics: List[str], patterns: List[str]) -> List[str]:
    if not patterns or patterns == ["*"]:
        return available_metrics
    selected = []
    for pattern in patterns:
        selected.extend(m for m in available_metrics if fnmatch.fnmatch(m, pattern))
    return list(dict.fromkeys(selected))  # dedupe, preserve order


def get_metrics_data(
    stats_db: EvalStatsDB,
    policy_uri: str,
    metrics: List[str],
    sim_name: str | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    Return {metric: {"mean": μ, "std": σ,
                     "count": K_recorded,
                     "samples": N_potential}}
        • μ, σ are normalized (missing values = 0).
        • K_recorded  – rows in policy_simulation_agent_metrics.
        • N_potential – total agent-episode pairs for that filter.
    """
    metadata = CheckpointManager.get_policy_metadata(policy_uri)
    pk, pv = metadata["run_name"], metadata["epoch"]
    filter_condition = f"sim_name = '{sim_name}'" if sim_name else None

    data: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        mean = stats_db.get_average_metric(m, policy_uri, filter_condition)
        if mean is None:
            continue
        std = stats_db.get_std_metric(m, policy_uri, filter_condition) or 0.0

        k_recorded = stats_db.count_metric_agents(pk, pv, m, filter_condition)
        n_potential = stats_db.potential_samples_for_metric(pk, pv, filter_condition)

        data[m] = {
            "mean": mean,
            "std": std,
            "count": k_recorded,
            "samples": n_potential,
        }
    return data


def print_metrics_table(metrics_data: Dict[str, Dict[str, float]], policy_uri: str) -> None:
    logger = logging.getLogger(__name__)
    if not metrics_data:
        logger.warning(f"No metrics data available for policy: {policy_uri}")
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

    logger.info(f"\nMetrics for policy: {policy_uri}\n")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    logger.info("")
