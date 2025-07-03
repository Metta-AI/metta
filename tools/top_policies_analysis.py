#!/usr/bin/env -S uv run
"""
Tool to extract top performing policies from the observatory database.

This script:
1. Connects to the observatory PostgreSQL database
2. Queries for policies ranked by success rates across all evaluations
3. Extracts policy URIs and metadata for the top N performers
4. Outputs results in formats suitable for downstream analysis

Usage:
    ./tools/top_policies_analysis.py ++observatory_db_uri=postgresql://user:pass@host:port/db ++num_policies=100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from metta.common.util.logging import setup_mettagrid_logger


class ObservatoryPolicyAnalyzer:
    """Analyzer for extracting top performing policies from observatory database."""

    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self.logger = logging.getLogger(__name__)

    def connect(self) -> psycopg.Connection:
        """Create database connection."""
        return psycopg.connect(self.db_uri, row_factory=dict_row)

    def get_top_policies_by_success_rate(self, num_policies: int = 100) -> List[Dict]:
        """
        Query for top N policies ranked by average success rate across all evaluations.

        Args:
            num_policies: Number of top policies to return

        Returns:
            List of policy dictionaries with metadata and performance data
        """
        query = """
        WITH policy_success_rates AS (
            SELECT
                p.id as policy_id,
                p.name as policy_name,
                p.url as policy_url,
                p.created_at as policy_created_at,
                tr.name as run_name,
                tr.user_id,
                tr.created_at as run_created_at,
                tr.finished_at as run_finished_at,
                tr.status as run_status,
                tr.description as run_description,
                tr.tags as run_tags,
                e.eval_category,
                e.env_name,
                AVG(eam.value) as avg_success_rate,
                COUNT(DISTINCT e.id) as num_episodes,
                COUNT(DISTINCT e.eval_category || '/' || e.env_name) as num_eval_environments
            FROM policies p
            JOIN epochs ep ON p.epoch_id = ep.id
            JOIN training_runs tr ON ep.run_id = tr.id
            JOIN episodes e ON e.primary_policy_id = p.id
            JOIN episode_agent_metrics eam ON e.id = eam.episode_id
            WHERE eam.metric = 'success'
                AND e.eval_category IS NOT NULL
                AND e.env_name IS NOT NULL
            GROUP BY p.id, p.name, p.url, p.created_at, tr.name, tr.user_id,
                     tr.created_at, tr.finished_at, tr.status, tr.description, tr.tags,
                     e.eval_category, e.env_name
        ),
        policy_overall_performance AS (
            SELECT
                policy_id,
                policy_name,
                policy_url,
                policy_created_at,
                run_name,
                user_id,
                run_created_at,
                run_finished_at,
                run_status,
                run_description,
                run_tags,
                AVG(avg_success_rate) as overall_success_rate,
                COUNT(DISTINCT eval_category || '/' || env_name) as total_eval_environments,
                COUNT(DISTINCT eval_category) as num_eval_categories,
                STDDEV(avg_success_rate) as success_rate_std,
                MIN(avg_success_rate) as min_success_rate,
                MAX(avg_success_rate) as max_success_rate
            FROM policy_success_rates
            GROUP BY policy_id, policy_name, policy_url, policy_created_at, run_name,
                     user_id, run_created_at, run_finished_at, run_status,
                     run_description, run_tags
        )
        SELECT *
        FROM policy_overall_performance
        WHERE overall_success_rate IS NOT NULL
        ORDER BY overall_success_rate DESC
        LIMIT %s
        """

        with self.connect() as conn:
            self.logger.info(f"Querying for top {num_policies} policies...")
            df = pd.read_sql_query(query, conn, params=(num_policies,))

        self.logger.info(f"Found {len(df)} policies with success rate data")
        return df.to_dict("records")

    def get_policy_evaluation_details(self, policy_ids: List[str]) -> pd.DataFrame:
        """
        Get detailed evaluation performance for specific policies.

        Args:
            policy_ids: List of policy IDs to get details for

        Returns:
            DataFrame with detailed evaluation performance
        """
        query = """
        SELECT
            p.id as policy_id,
            p.name as policy_name,
            e.eval_category,
            e.env_name,
            e.eval_category || '/' || e.env_name as eval_name,
            AVG(eam.value) as avg_success_rate,
            COUNT(DISTINCT e.id) as num_episodes,
            COUNT(DISTINCT eam.agent_id) as num_agents
        FROM policies p
        JOIN episodes e ON e.primary_policy_id = p.id
        JOIN episode_agent_metrics eam ON e.id = eam.episode_id
        WHERE p.id = ANY(%s)
            AND eam.metric = 'success'
            AND e.eval_category IS NOT NULL
            AND e.env_name IS NOT NULL
        GROUP BY p.id, p.name, e.eval_category, e.env_name
        ORDER BY p.name, e.eval_category, e.env_name
        """

        with self.connect() as conn:
            self.logger.info(f"Getting detailed evaluation data for {len(policy_ids)} policies...")
            df = pd.read_sql_query(query, conn, params=(policy_ids,))

        return df

    def extract_policy_uris(self, policies: List[Dict]) -> List[str]:
        """
        Extract policy URIs from policy data.

        Args:
            policies: List of policy dictionaries from get_top_policies_by_success_rate

        Returns:
            List of policy URIs in wandb:// format
        """
        uris = []
        for policy in policies:
            # Convert policy name to wandb URI format
            # Assuming policy names are in format like "user.run_name"
            if policy["policy_name"] and "." in policy["policy_name"]:
                uri = f"wandb://run/{policy['policy_name']}"
                uris.append(uri)
            else:
                self.logger.warning(f"Could not convert policy name to URI: {policy['policy_name']}")

        return uris

    def save_results(self, policies: List[Dict], eval_details: pd.DataFrame, policy_uris: List[str], output_dir: Path):
        """
        Save analysis results to files.

        Args:
            policies: Top policies data
            eval_details: Detailed evaluation performance
            policy_uris: List of policy URIs
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save top policies summary
        policies_file = output_dir / "top_policies_summary.json"
        with open(policies_file, "w") as f:
            json.dump(policies, f, indent=2, default=str)

        # Save evaluation details
        eval_file = output_dir / "policy_evaluation_details.csv"
        eval_details.to_csv(eval_file, index=False)

        # Save policy URIs
        uris_file = output_dir / "policy_uris.txt"
        with open(uris_file, "w") as f:
            for uri in policy_uris:
                f.write(f"{uri}\n")

        # Save policy URIs as JSON for easier processing
        uris_json_file = output_dir / "policy_uris.json"
        with open(uris_json_file, "w") as f:
            json.dump(policy_uris, f, indent=2)

        # Create performance matrix for factor analysis
        performance_matrix = eval_details.pivot_table(
            index="policy_name", columns="eval_name", values="avg_success_rate", fill_value=0.0
        )

        matrix_file = output_dir / "performance_matrix.csv"
        performance_matrix.to_csv(matrix_file)

        self.logger.info(f"Results saved to {output_dir}")
        self.logger.info(f"  - Top policies summary: {policies_file}")
        self.logger.info(f"  - Evaluation details: {eval_file}")
        self.logger.info(f"  - Policy URIs: {uris_file}")
        self.logger.info(f"  - Performance matrix: {matrix_file}")


def main():
    parser = argparse.ArgumentParser(description="Extract top performing policies from observatory database")
    parser.add_argument(
        "--observatory-db-uri", required=True, help="PostgreSQL connection string for observatory database"
    )
    parser.add_argument("--num-policies", type=int, default=100, help="Number of top policies to extract")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"), help="Directory to save results")

    args = parser.parse_args()

    # Setup logging
    logger = setup_mettagrid_logger("top_policies_analysis")
    logger.info(f"Starting analysis for top {args.num_policies} policies")

    # Create analyzer and run analysis
    analyzer = ObservatoryPolicyAnalyzer(args.observatory_db_uri)

    try:
        # Get top policies
        top_policies = analyzer.get_top_policies_by_success_rate(args.num_policies)

        if not top_policies:
            logger.error("No policies found with success rate data")
            return 1

        # Extract policy IDs for detailed analysis
        policy_ids = [p["policy_id"] for p in top_policies]

        # Get detailed evaluation data
        eval_details = analyzer.get_policy_evaluation_details(policy_ids)

        # Extract policy URIs
        policy_uris = analyzer.extract_policy_uris(top_policies)

        # Save results
        analyzer.save_results(top_policies, eval_details, policy_uris, args.output_dir)

        logger.info(f"Analysis complete. Found {len(top_policies)} top policies")
        logger.info(f"Policy URIs extracted: {len(policy_uris)}")

        # Print summary
        print("\nTop 10 Policies by Success Rate:")
        print("-" * 80)
        for i, policy in enumerate(top_policies[:10]):
            print(
                f"{i + 1:2d}. {policy['policy_name']:40s} "
                f"Success Rate: {policy['overall_success_rate']:.3f} "
                f"Environments: {policy['total_eval_environments']}"
            )

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
