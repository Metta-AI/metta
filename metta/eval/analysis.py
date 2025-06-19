import fnmatch
import logging
from typing import Dict, List, Optional

from tabulate import tabulate

from metta.agent.metta_agent import MettaAgent
from metta.eval.analysis_config import AnalysisConfig
from metta.eval.eval_stats_db import EvalStatsDB
from mettagrid.util.file import local_copy


def analyze(metta_agent: MettaAgent, config: AnalysisConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing policy: {metta_agent.uri}")
    logger.info(f"Using eval DB: {config.eval_db_uri}")

    with local_copy(config.eval_db_uri) as local_path:
        stats_db = EvalStatsDB(local_path)

        sample_count = stats_db.sample_count(metta_agent, config.suite)
        if sample_count == 0:
            logger.warning(f"No samples found for policy: {metta_agent.key}:v{metta_agent.version}")
            return
        logger.info(f"Total sample count for specified policy/suite: {sample_count}")

        available_metrics = get_available_metrics(stats_db, metta_agent)
        logger.info(f"Available metrics: {available_metrics}")

        selected_metrics = config.metrics or available_metrics
        if not selected_metrics:
            logger.warning("No metrics specified or found to analyze.")
            return
        logger.info(f"Selected metrics: {', '.join(selected_metrics)}")

        metrics_data = get_metrics_data(stats_db, metta_agent, selected_metrics, config.suite)
        print_metrics_table(metrics_data, metta_agent)


# --------------------------------------------------------------------------- #
#   helpers                                                                   #
# --------------------------------------------------------------------------- #
def get_available_metrics(stats_db: EvalStatsDB, metta_agent: MettaAgent) -> List[str]:
    policy_key, policy_version = metta_agent.key_and_version()
    result = stats_db.query(
        f"""
        SELECT DISTINCT metric
          FROM policy_simulation_agent_metrics
         WHERE policy_key = '{policy_key}'
           AND policy_version = {policy_version}
    """
    )
    return result["metric"].tolist()


def filter_metrics(available_metrics: List[str], patterns: List[str]) -> List[str]:
    if not patterns or patterns == ["*"]:
        return available_metrics
    selected = []
    for pattern in patterns:
        selected.extend(m for m in available_metrics if fnmatch.fnmatch(m, pattern))
    return list(dict.fromkeys(selected))  # dedupe, preserve order


def get_metrics_data(
    stats_db: EvalStatsDB,
    metta_agent: MettaAgent,
    metrics: List[str],
    suite: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Return {metric: {"mean": μ, "std": σ,
                     "count": K_recorded,
                     "samples": N_potential}}
        • μ, σ are normalised (missing values = 0).
        • K_recorded  – rows in policy_simulation_agent_metrics.
        • N_potential – total agent-episode pairs for that filter.
    """
    data = {}
    policy_key, policy_version = metta_agent.key_and_version()
    filter_condition = f"sim_suite = '{suite}'" if suite else None

    for m in metrics:
        data[m] = {}
        mean = stats_db.get_average_metric_by_filter(m, metta_agent, filter_condition)
        if mean is not None:
            data[m]["mean"] = mean
            std = stats_db.get_std_metric_by_filter(m, metta_agent, filter_condition) or 0.0
            data[m]["std"] = std

    return data


def print_metrics_table(metrics_data: Dict[str, Dict[str, float]], metta_agent: MettaAgent) -> None:
    logger = logging.getLogger(__name__)
    if not metrics_data:
        logger.warning(f"No metrics data available for {metta_agent.key}:v{metta_agent.version}")
        return

    # Filter out empty metrics
    metrics_data = {k: v for k, v in metrics_data.items() if v}

    headers = ["Metric", "Mean", "Std Dev"]
    table_data = [[k, v.get("mean", "N/A"), v.get("std", "N/A")] for k, v in metrics_data.items()]

    # Format numbers to 4 decimal places
    for row in table_data:
        if isinstance(row[1], float):
            row[1] = f"{row[1]:.4f}"
        if isinstance(row[2], float):
            row[2] = f"{row[2]:.4f}"

    logger.info(f"\nMetrics for policy: {metta_agent.uri}\n")
    logger.info(tabulate(table_data, headers=headers, tablefmt="grid"))
    logger.info("")
