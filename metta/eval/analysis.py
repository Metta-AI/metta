"""Analysis tool for MettaGrid evaluation results."""

import fnmatch
import logging
from typing import Dict, List, Optional

from tabulate import tabulate

from metta.agent.policy_store import PolicyRecord
from metta.eval.analysis_config import AnalysisConfig
from metta.eval.eval_stats_db import EvalStatsDB
from mettagrid.util.file import local_copy


def analyze(policy_record: PolicyRecord, config: AnalysisConfig) -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing policy: {policy_record.uri}")
    logger.info(f"Using eval DB: {config.eval_db_uri}")

    # Open the eval database
    with local_copy(config.eval_db_uri) as local_path:
        stats_db = EvalStatsDB(local_path)

        sample_count = stats_db.sample_count(policy_record, config.suite)
        if sample_count == 0:
            logger.warning(f"No samples found for policy: {policy_record.key}:v{policy_record.version}")
            return
        else:
            logger.info(f"Total sample count for specified policy/suite: {sample_count}")
        # Get all available metrics for this policy
        available_metrics = get_available_metrics(stats_db, policy_record)
        logger.info(f"Available metrics: {available_metrics}")

        # Filter metrics based on glob patterns
        selected_metrics = filter_metrics(available_metrics, config.metrics)
        if not selected_metrics:
            logger.warning(f"No metrics found matching patterns: {config.metrics}")
            return
        logger.info(f"Selected metrics: {selected_metrics}")

        metrics_data = get_metrics_data(stats_db, policy_record, selected_metrics, config.suite)
        print_metrics_table(metrics_data, policy_record)


def get_available_metrics(stats_db: EvalStatsDB, policy_record: PolicyRecord) -> List[str]:
    policy_key, policy_version = policy_record.key_and_version()

    # Query the database for all metrics for this policy
    query = f"""
    SELECT DISTINCT metric
    FROM policy_simulation_agent_metrics
    WHERE policy_key = '{policy_key}'
    AND policy_version = {policy_version}
    ORDER BY metric
    """

    result = stats_db.query(query)
    if result.empty:
        return []
    return result["metric"].tolist()


def filter_metrics(available_metrics: List[str], patterns: List[str]) -> List[str]:
    """Filter metrics based on glob patterns."""
    if not patterns or patterns == ["*"]:
        return available_metrics

    selected_metrics = []
    for pattern in patterns:
        matching_metrics = [m for m in available_metrics if fnmatch.fnmatch(m, pattern)]
        selected_metrics.extend(matching_metrics)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(selected_metrics))


def get_metrics_data(
    stats_db: EvalStatsDB, policy_record: PolicyRecord, metrics: List[str], suite: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    policy_key, policy_version = policy_record.key_and_version()

    # Build the SQL query
    sql = f"""
    SELECT metric, AVG(value) as mean, STDDEV_SAMP(value) as std, COUNT(*) as count
    FROM policy_simulation_agent_metrics
    WHERE policy_key = '{policy_key}'
    AND policy_version = {policy_version}
    AND metric IN ({", ".join(["?" for _ in metrics])})
    """
    params = metrics.copy()
    # Add suite filter if specified
    if suite:
        sql += " AND sim_suite = ?"
        params.append(suite)

    sql += " GROUP BY metric ORDER BY metric"

    # Execute the query
    result = stats_db.con.execute(sql, params).fetchdf()

    if result.empty:
        return {}

    # Convert to dictionary for easier access
    metrics_data = {}
    for _, row in result.iterrows():
        metrics_data[row["metric"]] = {"mean": row["mean"], "std": row["std"], "count": row["count"]}

    return metrics_data


def print_metrics_table(metrics_data: Dict[str, Dict[str, float]], policy_record: PolicyRecord) -> None:
    if not metrics_data:
        print(f"No metrics data available for {policy_record.key}:v{policy_record.version}")
        return

    # Format the table headers
    headers = ["Metric", "Average", "Std Dev", "Sample Count"]

    # Format the table rows
    rows = []
    for metric, stats in metrics_data.items():
        rows.append([metric, f"{stats['mean']:.4f}", f"{stats['std']:.4f}", f"{int(stats['count'])}"])

    # Print the policy information
    print(f"\nMetrics for policy: {policy_record.key}:v{policy_record.version}\n")

    # Print the table
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()
