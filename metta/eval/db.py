"""
Database interface for storing and querying policy evaluation metrics.
"""

import logging
from typing import Literal, Tuple

import pandas as pd

from metta.sim.stats_db import StatsDB


def parse_versioned_uri(uri: str) -> Tuple[str, str]:
    """
    Parse a versioned policy uri into uri and version.
    Examples:
    - "navigation_training_suite_onlyhearts:v10" -> ("navigation_training_suite_onlyhearts", "v10")
    - "wandb://run/navigation_training_suite_onlyhearts:v10" -> ("navigation_training_suite_onlyhearts", "v10")
    """
    # Remove any prefix
    if "://" in uri:
        uri = uri.split("/")[-1]

    # Extract version if present
    if ":v" in uri:
        parts = uri.split(":v")
        name = parts[0]
        version = "v" + parts[1]
    else:
        # If no version specified, treat the entire URI as the name and set version to "latest"
        name = uri
        version = "latest"

    return name, version


def get_matrix_data(
    stats_db: StatsDB,
    metric: str,
    view_type: Literal["latest", "policy_versions", "chronological", "all"] = "latest",
    policy_uri: str | None = None,
    num_output_policies: int | Literal["all"] = "all",
) -> pd.DataFrame:
    """
    Get matrix data for the specified metric from a StatsDB.

    Args:
        stats_db: StatsDB instance
        metric: The metric to get data for
        view_type:
            - "latest": Only the latest version of each policy, sorted by score
            - "all": All versions of all policies, sorted by score
            - "policy_versions": All versions of a specific policy, sorted by version
            - "chronological": All policies and versions, sorted by creation date
        policy_uri: Required for "policy_versions" view_type
        num_output_policies: Optional number of policies to output

    Returns:
        DataFrame with policies as rows and evaluations as columns
    """
    # Ensure the materialized view exists
    view_name = f"policy_simulations_{metric}"
    logger = logging.getLogger(__name__)
    # Try to materialize the view if it doesn't exist
    try:
        logger.info(f"Materializing view for metric {metric}")
        stats_db.materialize_policy_simulations_view(metric)
        logger.info(f"Materialized view for metric {metric}")
    except Exception as e:
        logger.error(f"Failed to materialize view for metric {metric}: {e}")
        return pd.DataFrame()

    # Parse the policy_uri if provided
    if policy_uri:
        if "://" in policy_uri:
            policy_uri = policy_uri.split("/")[-1]

        # Extract without version if it has one
        if ":v" in policy_uri:
            policy_uri = policy_uri.split(":v")[0]

    # Build the SQL query based on view_type
    if view_type == "latest":
        # Get the latest version for each policy
        sql = f"""
        WITH latest_versions AS (
            SELECT policy_key, MAX(policy_version) as policy_version
            FROM policies
            GROUP BY policy_key
        )
        SELECT 
            ps.policy_key || ':' || ps.policy_version as policy_uri,
            ps.eval_name,
            ps.{metric} as value
        FROM {view_name} ps
        JOIN latest_versions lv ON ps.policy_key = lv.policy_key AND ps.policy_version = lv.policy_version
        """
    elif view_type == "policy_versions" and policy_uri:
        # Get all versions of a specific policy
        sql = f"""
        SELECT 
            ps.policy_key || ':' || ps.policy_version as policy_uri,
            ps.eval_name,
            ps.{metric} as value
        FROM {view_name} ps
        WHERE ps.policy_key = '{policy_uri}'
        """
    elif view_type == "chronological":
        # All policies and versions, sorted chronologically
        # Note: We don't have creation date in materialized view,
        # but policy_version often has chronological meaning
        sql = f"""
        SELECT 
            ps.policy_key || ':' || ps.policy_version as policy_uri,
            ps.eval_name,
            ps.{metric} as value
        FROM {view_name} ps
        ORDER BY ps.policy_key, ps.policy_version
        """
    else:  # "all" or default
        # Get all versions of all policies
        sql = f"""
        SELECT 
            ps.policy_key || ':' || ps.policy_version as policy_uri,
            ps.eval_name,
            ps.{metric} as value
        FROM {view_name} ps
        """

    # Execute the query
    try:
        df = stats_db.query(sql)
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        return pd.DataFrame()

    if len(df) == 0:
        logger.warning(f"No data found for metric {metric}")
        return pd.DataFrame()

    # Process data into matrix format
    policies = df["policy_uri"].unique()
    eval_names = df["eval_name"].unique()

    # Generate an overall score for each policy
    overall_scores = {}
    for policy in policies:
        policy_data = df[df["policy_uri"] == policy]
        overall_scores[policy] = policy_data["value"].mean()

    # Create mapping for easy lookup
    data_map = {}
    for _, row in df.iterrows():
        data_map[(row["policy_uri"], row["eval_name"])] = row["value"]

    # Create matrix data using full eval names
    matrix_data = []
    for policy in policies:
        row_data = {"policy_uri": policy}
        row_data["Overall"] = overall_scores[policy]
        for eval_name in eval_names:
            key = (policy, eval_name)
            if key in data_map:
                row_data[eval_name] = data_map[key]
        matrix_data.append(row_data)

    # Convert to DataFrame
    matrix = pd.DataFrame(matrix_data)
    matrix = matrix.set_index("policy_uri")

    # Sort appropriately based on view_type
    if view_type in ["latest", "all"]:
        # Sort by overall score (lowest first)
        sorted_policies = sorted(policies, key=lambda p: overall_scores[p])
        matrix = matrix.reindex(sorted_policies)
    elif view_type == "policy_versions" and policy_uri:
        # For policy versions view, sort by version number for clear progression
        def version_key(uri):
            parts = uri.split(":v")
            if len(parts) > 1 and parts[1].isdigit():
                return int(parts[1])
            return uri  # Fallback to string comparison

        sorted_policies = sorted(policies, key=version_key)
        matrix = matrix.reindex(sorted_policies)
    # For chronological view, we don't need to re-sort as it's already sorted in the SQL query

    # Limit the number of policies
    if num_output_policies != "all" and isinstance(num_output_policies, int):
        matrix = matrix.tail(num_output_policies)

    return matrix
