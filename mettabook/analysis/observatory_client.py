"""
Observatory API client for policy analysis.
Based on the existing ObservatoryAPIClient from tools/observatory_policy_analysis.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import requests


class ObservatoryClient:
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

        # Set a longer timeout for all requests (10x the default)
        if "timeout" not in kwargs:
            kwargs["timeout"] = 200  # 200 seconds instead of ~20 seconds

        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, headers=headers, **kwargs)

        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

        return response.json()

    def execute_query(self, query: str) -> Dict:
        """Execute a SQL query via the observatory API."""
        return self._make_request("POST", "/sql/query", json={"query": query})

    def get_environments(self) -> List[str]:
        """Get list of available environments."""
        query = """
        SELECT DISTINCT env_name
        FROM episodes
        WHERE env_name IS NOT NULL
        ORDER BY env_name
        """
        result = self.execute_query(query)
        return [row[0] for row in result["rows"]]

    def get_top_policies(self, n: int = 20, environments: Optional[List[str]] = None) -> List[Dict]:
        """Get top N policies by average reward."""
        env_filter = ""
        if environments:
            env_list = "', '".join(environments)
            env_filter = f"AND ep.env_name IN ('{env_list}')"

        # First, let's try a simpler query to see what's available
        diagnostic_query = """
        SELECT
            COUNT(*) as total_policies,
            COUNT(DISTINCT p.id) as unique_policies,
            COUNT(DISTINCT eap.policy_id) as policies_with_episodes,
            COUNT(DISTINCT eam.agent_id) as agents_with_metrics,
            COUNT(DISTINCT ep.env_name) as unique_environments
        FROM policies p
        LEFT JOIN episode_agent_policies eap ON p.id = eap.policy_id
        LEFT JOIN episode_agent_metrics eam ON eap.agent_id = eam.agent_id
        LEFT JOIN episodes ep ON eap.episode_id = ep.id
        """

        print("🔍 Running diagnostic query...")
        diagnostic_result = self.execute_query(diagnostic_query)
        if diagnostic_result["rows"]:
            row = diagnostic_result["rows"][0]
            print("📊 Diagnostic results:")
            print(f"   Total policies: {row[0]}")
            print(f"   Unique policies: {row[1]}")
            print(f"   Policies with episodes: {row[2]}")
            print(f"   Agents with metrics: {row[3]}")
            print(f"   Unique environments: {row[4]}")

        # Let's also check what environments are available
        env_query = """
        SELECT DISTINCT env_name
        FROM episodes
        WHERE env_name IS NOT NULL
        ORDER BY env_name
        LIMIT 10
        """

        print("🌍 Checking available environments...")
        env_result = self.execute_query(env_query)
        if env_result["rows"]:
            print("   Available environments:")
            for row in env_result["rows"]:
                print(f"   - {row[0]}")

        # Let's also check what metrics are available
        metric_query = """
        SELECT DISTINCT metric
        FROM episode_agent_metrics
        WHERE metric IS NOT NULL
        ORDER BY metric
        LIMIT 10
        """

        print("📈 Checking available metrics...")
        metric_result = self.execute_query(metric_query)
        if metric_result["rows"]:
            print("   Available metrics:")
            for row in metric_result["rows"]:
                print(f"   - {row[0]}")

                # Use a much simpler and more efficient query
        query = f"""
        SELECT
            p.id as policy_id,
            p.name as policy_name,
            p.description as policy_description,
            p.url as policy_url,
            p.created_at,
            p.epoch_id,
            AVG(eam.value) as avg_reward,
            COUNT(*) as episode_count,
            STDDEV(eam.value) as reward_std,
            MIN(eam.value) as min_reward,
            MAX(eam.value) as max_reward
        FROM policies p
        INNER JOIN episode_agent_policies eap ON p.id = eap.policy_id
        INNER JOIN episode_agent_metrics eam ON eap.agent_id = eam.agent_id
        INNER JOIN episodes ep ON eap.episode_id = ep.id
        WHERE eam.metric = 'reward'
        AND eam.value IS NOT NULL
        {env_filter}
        GROUP BY p.id, p.name, p.description, p.url, p.created_at, p.epoch_id
        HAVING COUNT(*) >= 1
        ORDER BY avg_reward DESC
        LIMIT {n}
        """

        result = self.execute_query(query)

        policies = []
        for row in result["rows"]:
            policy = {
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "url": row[3],
                "created_at": row[4],
                "epoch_id": row[5],
                "avg_reward": row[6],
                "episode_count": row[7],
                "reward_std": row[8],
                "min_reward": row[9],
                "max_reward": row[10],
            }
            policies.append(policy)

        return policies

    def get_policy_evaluations(self, policy_ids: List[str], environments: Optional[List[str]] = None) -> List[Dict]:
        """Get evaluation data for specific policies."""
        if not policy_ids:
            return []

        policy_ids_str = "', '".join(policy_ids)
        env_filter = ""
        if environments:
            env_list = "', '".join(environments)
            env_filter = f"AND ep.env_name IN ('{env_list}')"

        query = f"""
        SELECT
            eap.episode_id,
            eap.policy_id,
            eam.agent_id,
            eam.metric,
            eam.value,
            ep.env_name as environment,
            ep.created_at
        FROM episode_agent_metrics eam
        LEFT JOIN episode_agent_policies eap ON eam.agent_id = eap.agent_id
        LEFT JOIN episodes ep ON eap.episode_id = ep.id
        WHERE eap.policy_id IN ('{policy_ids_str}') {env_filter}
        ORDER BY eap.policy_id, eap.episode_id, eam.metric
        """

        result = self.execute_query(query)

        evaluations = []
        for row in result["rows"]:
            evaluation = {
                "episode_id": row[0],
                "policy_id": row[1],
                "agent_id": row[2],
                "metric": row[3],
                "value": row[4],
                "environment": row[5],
                "created_at": row[6],
            }
            evaluations.append(evaluation)

        return evaluations

    def debug_database_structure(self) -> Dict:
        """Debug method to understand the database structure and data availability."""
        print("🔍 Starting comprehensive database debugging...")

        debug_results = {}

        # 1. Check basic table counts
        print("\n📊 Checking table counts...")
        table_counts = {
            "policies": "SELECT COUNT(*) FROM policies",
            "episode_agent_policies": "SELECT COUNT(*) FROM episode_agent_policies",
            "episode_agent_metrics": "SELECT COUNT(*) FROM episode_agent_metrics",
            "episodes": "SELECT COUNT(*) FROM episodes",
        }

        for table, query in table_counts.items():
            try:
                result = self.execute_query(query)
                count = result["rows"][0][0] if result["rows"] else 0
                debug_results[f"{table}_count"] = count
                print(f"   {table}: {count:,} records")
            except Exception as e:
                print(f"   ❌ Error querying {table}: {e}")
                debug_results[f"{table}_count"] = "ERROR"

        # 2. Check for policies with episodes
        print("\n🔗 Checking policy-episode relationships...")
        try:
            policy_episode_query = """
            SELECT COUNT(DISTINCT p.id) as policies_with_episodes
            FROM policies p
            INNER JOIN episode_agent_policies eap ON p.id = eap.policy_id
            """
            result = self.execute_query(policy_episode_query)
            count = result["rows"][0][0] if result["rows"] else 0
            debug_results["policies_with_episodes"] = count
            print(f"   Policies with episodes: {count:,}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            debug_results["policies_with_episodes"] = "ERROR"

        # 3. Check for policies with metrics
        print("\n📈 Checking policy-metric relationships...")
        try:
            policy_metric_query = """
            SELECT COUNT(DISTINCT eap.policy_id) as policies_with_metrics
            FROM episode_agent_policies eap
            INNER JOIN episode_agent_metrics eam ON eap.agent_id = eam.agent_id
            """
            result = self.execute_query(policy_metric_query)
            count = result["rows"][0][0] if result["rows"] else 0
            debug_results["policies_with_metrics"] = count
            print(f"   Policies with metrics: {count:,}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            debug_results["policies_with_metrics"] = "ERROR"

        # 4. Check available metrics
        print("\n📊 Checking available metrics...")
        try:
            metrics_query = """
            SELECT metric, COUNT(*) as count
            FROM episode_agent_metrics
            WHERE metric IS NOT NULL
            GROUP BY metric
            ORDER BY count DESC
            LIMIT 10
            """
            result = self.execute_query(metrics_query)
            debug_results["available_metrics"] = []
            for row in result["rows"]:
                metric, count = row[0], row[1]
                debug_results["available_metrics"].append({"metric": metric, "count": count})
                print(f"   {metric}: {count:,} records")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            debug_results["available_metrics"] = "ERROR"

        # 5. Check available environments
        print("\n🌍 Checking available environments...")
        try:
            env_query = """
            SELECT env_name, COUNT(*) as count
            FROM episodes
            WHERE env_name IS NOT NULL
            GROUP BY env_name
            ORDER BY count DESC
            LIMIT 10
            """
            result = self.execute_query(env_query)
            debug_results["available_environments"] = []
            for row in result["rows"]:
                env, count = row[0], row[1]
                debug_results["available_environments"].append({"env": env, "count": count})
                print(f"   {env}: {count:,} episodes")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            debug_results["available_environments"] = "ERROR"

        # 6. Test a simple policy query
        print("\n🧪 Testing simple policy query...")
        try:
            simple_query = """
            SELECT p.id, p.name, COUNT(eap.episode_id) as episode_count
            FROM policies p
            LEFT JOIN episode_agent_policies eap ON p.id = eap.policy_id
            GROUP BY p.id, p.name
            HAVING COUNT(eap.episode_id) > 0
            ORDER BY episode_count DESC
            LIMIT 5
            """
            result = self.execute_query(simple_query)
            debug_results["sample_policies"] = []
            for row in result["rows"]:
                policy_id, name, episode_count = row[0], row[1], row[2]
                debug_results["sample_policies"].append({"id": policy_id, "name": name, "episode_count": episode_count})
                print(f"   Policy {policy_id} ({name}): {episode_count} episodes")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            debug_results["sample_policies"] = "ERROR"

        print("\n✅ Database debugging complete!")
        return debug_results

    def test_simple_policy_query(self, n: int = 5) -> List[Dict]:
        """Test a very simple policy query to see if basic functionality works."""
        print(f"🧪 Testing simple policy query (limit {n})...")

        query = f"""
        SELECT
            p.id,
            p.name,
            p.description,
            p.url,
            p.created_at,
            p.epoch_id,
            COUNT(eap.episode_id) as episode_count
        FROM policies p
        LEFT JOIN episode_agent_policies eap ON p.id = eap.policy_id
        GROUP BY p.id, p.name, p.description, p.url, p.created_at, p.epoch_id
        HAVING COUNT(eap.episode_id) > 0
        ORDER BY episode_count DESC
        LIMIT {n}
        """

        try:
            result = self.execute_query(query)
            policies = []
            for row in result["rows"]:
                policy = {
                    "id": row[0],
                    "name": row[1],
                    "description": row[2],
                    "url": row[3],
                    "created_at": row[4],
                    "epoch_id": row[5],
                    "episode_count": row[6],
                }
                policies.append(policy)

            print(f"✅ Found {len(policies)} policies with episodes")
            return policies

        except Exception as e:
            print(f"❌ Simple query failed: {e}")
            return []

    def test_metric_query(self, metric_name: str = "reward") -> Dict:
        """Test if a specific metric exists and has data."""
        print(f"📊 Testing metric '{metric_name}'...")

        query = f"""
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT eam.agent_id) as unique_agents,
            AVG(eam.value) as avg_value,
            MIN(eam.value) as min_value,
            MAX(eam.value) as max_value
        FROM episode_agent_metrics eam
        WHERE eam.metric = '{metric_name}'
        AND eam.value IS NOT NULL
        """

        try:
            result = self.execute_query(query)
            if result["rows"]:
                row = result["rows"][0]
                metrics = {
                    "total_records": row[0],
                    "unique_agents": row[1],
                    "avg_value": row[2],
                    "min_value": row[3],
                    "max_value": row[4],
                }
                print(f"✅ Metric '{metric_name}' found:")
                print(f"   Total records: {metrics['total_records']:,}")
                print(f"   Unique agents: {metrics['unique_agents']:,}")
                print(f"   Value range: {metrics['min_value']:.3f} to {metrics['max_value']:.3f}")
                print(f"   Average: {metrics['avg_value']:.3f}")
                return metrics
            else:
                print(f"❌ No data found for metric '{metric_name}'")
                return {}

        except Exception as e:
            print(f"❌ Error testing metric '{metric_name}': {e}")
            return {}
