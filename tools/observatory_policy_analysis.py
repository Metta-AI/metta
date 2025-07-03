#!/usr/bin/env -S uv run
"""
Tool to extract top performing policies from the observatory database via API.

This script:
1. Uses observatory CLI authentication to get API access
2. Queries the observatory API for policies ranked by success rates across all evaluations
3. Extracts policy URIs and metadata for the top N performers
4. Outputs results in formats suitable for downstream analysis

Usage:
    ./tools/observatory_policy_analysis.py ++num_policies=100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from metta.common.util.logging import setup_mettagrid_logger


class ObservatoryAPIClient:
    """Client for interacting with the observatory API."""

    def __init__(self, base_url: str = "https://api.observatory.softmax-research.net"):
        self.base_url = base_url
        self.token = self._get_auth_token()
        self.logger = logging.getLogger(__name__)

    def _get_auth_token(self) -> Optional[str]:
        """Get authentication token from observatory CLI."""
        token_file = Path.home() / ".metta" / "observatory_token"
        if not token_file.exists():
            raise Exception("No observatory authentication token found. Please run: python devops/observatory_login.py")

        with open(token_file, "r") as f:
            return f.read().strip()

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make authenticated request to observatory API."""
        headers = {
            "X-Auth-Token": self.token,
            "Content-Type": "application/json",
        }

        # Add any additional headers from kwargs
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, headers=headers, **kwargs)

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

        return response.json()

    def execute_query(self, query: str) -> Dict:
        """Execute a SQL query via the observatory API."""
        return self._make_request("POST", "/sql/query", json={"query": query})

    def list_tables(self) -> List[Dict]:
        """List available tables in the database."""
        return self._make_request("GET", "/sql/tables")

    def get_table_schema(self, table_name: str) -> Dict:
        """Get schema for a specific table."""
        return self._make_request("GET", f"/sql/tables/{table_name}/schema")


class ObservatoryPolicyAnalyzer:
    """Analyzer for extracting top performing policies from observatory database via API."""

    def __init__(self, api_client: ObservatoryAPIClient):
        self.api_client = api_client
        self.logger = logging.getLogger(__name__)

    def get_top_policies(self, num_policies: int = 100) -> pd.DataFrame:
        """
        Query for top performing policies ranked by average reward.

        Args:
            num_policies: Number of top policies to extract

        Returns:
            DataFrame with policy information and performance metrics
        """
        # Query to get top policies by average reward across all episodes
        query = f"""
        WITH policy_performance AS (
            SELECT
                p.id as policy_id,
                p.name as policy_name,
                p.description as policy_description,
                p.url as policy_url,
                p.created_at,
                p.epoch_id,
                AVG(eam.value) as avg_reward,
                COUNT(DISTINCT eam.episode_id) as episode_count,
                STDDEV(eam.value) as reward_std,
                MIN(eam.value) as min_reward,
                MAX(eam.value) as max_reward
            FROM policies p
            LEFT JOIN episode_agent_policies eap ON p.id = eap.policy_id
            LEFT JOIN episode_agent_metrics eam ON eap.episode_id = eam.episode_id
                AND eap.agent_id = eam.agent_id
            WHERE eam.metric = 'reward' AND eam.value IS NOT NULL
            GROUP BY p.id, p.name, p.description, p.url, p.created_at, p.epoch_id
            HAVING COUNT(DISTINCT eam.episode_id) >= 5  -- Only include policies with at least 5 episodes
        )
        SELECT
            policy_id,
            policy_name,
            policy_description,
            policy_url,
            created_at,
            epoch_id,
            avg_reward,
            episode_count,
            reward_std,
            min_reward,
            max_reward
        FROM policy_performance
        ORDER BY avg_reward DESC
        LIMIT {num_policies}
        """

        self.logger.info(f"Executing query for top {num_policies} policies...")
        result = self.api_client.execute_query(query)

        # Convert to DataFrame
        df = pd.DataFrame(result["rows"], columns=result["columns"])

        # Parse metadata JSON if present
        if "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(lambda x: json.loads(x) if x and isinstance(x, str) else x)

        self.logger.info(f"Retrieved {len(df)} top policies")
        return df

    def get_policy_evaluations(self, policy_ids: List[str]) -> pd.DataFrame:
        """
        Get detailed evaluation data for specific policies.

        Args:
            policy_ids: List of policy IDs to get evaluations for

        Returns:
            DataFrame with evaluation details
        """
        if not policy_ids:
            return pd.DataFrame()

        # Convert list to SQL IN clause
        policy_ids_str = "', '".join(policy_ids)

        query = f"""
        SELECT
            eam.episode_id,
            eap.policy_id,
            eam.agent_id,
            eam.metric,
            eam.value,
            ep.created_at
        FROM episode_agent_metrics eam
        LEFT JOIN episode_agent_policies eap ON eam.episode_id = eap.episode_id
            AND eam.agent_id = eap.agent_id
        LEFT JOIN episodes ep ON eam.episode_id = ep.id
        WHERE eap.policy_id IN ('{policy_ids_str}')
        ORDER BY eap.policy_id, eam.episode_id, eam.metric
        """

        self.logger.info(f"Getting evaluations for {len(policy_ids)} policies...")
        result = self.api_client.execute_query(query)

        df = pd.DataFrame(result["rows"], columns=result["columns"])

        # Parse metadata JSON if present
        if "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(lambda x: json.loads(x) if x and isinstance(x, str) else x)

        self.logger.info(f"Retrieved {len(df)} evaluations")
        return df

    def get_environment_info(self) -> pd.DataFrame:
        """Get information about all available episodes."""
        query = """
        SELECT
            id,
            created_at,
            primary_policy_id,
            stats_epoch,
            replay_url,
            eval_name,
            simulation_suite,
            attributes,
            eval_category,
            env_name
        FROM episodes
        ORDER BY created_at DESC
        """

        self.logger.info("Getting environment information...")
        result = self.api_client.execute_query(query)

        df = pd.DataFrame(result["rows"], columns=result["columns"])

        # Parse metadata JSON if present
        if "metadata" in df.columns:
            df["metadata"] = df["metadata"].apply(lambda x: json.loads(x) if x and isinstance(x, str) else x)

        self.logger.info(f"Retrieved {len(df)} environments")
        return df

    def save_results(
        self, policies_df: pd.DataFrame, evaluations_df: pd.DataFrame, environments_df: pd.DataFrame, output_dir: Path
    ) -> None:
        """Save analysis results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as CSV
        policies_df.to_csv(output_dir / "top_policies.csv", index=False)
        evaluations_df.to_csv(output_dir / "policy_evaluations.csv", index=False)
        environments_df.to_csv(output_dir / "environments.csv", index=False)

        # Save as JSON for downstream processing
        policies_df.to_json(output_dir / "top_policies.json", orient="records", indent=2)
        evaluations_df.to_json(output_dir / "policy_evaluations.json", orient="records", indent=2)
        environments_df.to_json(output_dir / "environments.json", orient="records", indent=2)

        # Save policy URIs as a simple list for evaluation stage
        policy_uris = policies_df["policy_url"].tolist()
        with open(output_dir / "policy_uris.json", "w") as f:
            json.dump(policy_uris, f, indent=2)

        # Create summary statistics
        summary = {
            "total_policies": len(policies_df),
            "total_evaluations": len(evaluations_df),
            "total_episodes": len(environments_df),
            "policy_reward_stats": {
                "mean": float(policies_df["avg_reward"].mean()),
                "std": float(policies_df["avg_reward"].std()),
                "min": float(policies_df["avg_reward"].min()),
                "max": float(policies_df["avg_reward"].max()),
                "median": float(policies_df["avg_reward"].median()),
            },
            "episode_status_counts": environments_df["status"].value_counts().to_dict()
            if "status" in environments_df.columns
            else {},
        }

        with open(output_dir / "summary_stats.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract top performing policies from observatory database via API")
    parser.add_argument("--num-policies", type=int, default=100, help="Number of top policies to extract")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_results"), help="Directory to save results")
    parser.add_argument(
        "--api-base-url", default="https://api.observatory.softmax-research.net", help="Observatory API base URL"
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_mettagrid_logger("observatory_policy_analysis")
    logger.info(f"Starting analysis for top {args.num_policies} policies")

    try:
        # Create API client
        api_client = ObservatoryAPIClient(args.api_base_url)
        logger.info("Successfully authenticated with observatory API")

        # Create analyzer
        analyzer = ObservatoryPolicyAnalyzer(api_client)

        # Get top policies
        policies_df = analyzer.get_top_policies(args.num_policies)

        # Get evaluations for these policies
        policy_ids = policies_df["policy_id"].tolist()
        evaluations_df = analyzer.get_policy_evaluations(policy_ids)

        # Get environment information
        environments_df = analyzer.get_environment_info()

        # Save results
        analyzer.save_results(policies_df, evaluations_df, environments_df, args.output_dir)

        logger.info("Analysis completed successfully!")

        # Print summary
        print("\n📊 Analysis Summary:")
        print(f"   Top policies extracted: {len(policies_df)}")
        print(f"   Total evaluations: {len(evaluations_df)}")
        print(f"   Total episodes: {len(environments_df)}")
        print(f"   Average reward: {policies_df['avg_reward'].mean():.3f}")
        print(f"   Results saved to: {args.output_dir}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
