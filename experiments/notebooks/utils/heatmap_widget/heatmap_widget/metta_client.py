# 🔧 Fixed API Client Implementation
from typing import List, Optional

import httpx

from .HeatmapWidget import HeatmapWidget, create_heatmap_widget


class MettaAPIClient:
    """Fixed client that properly handles authentication and response parsing."""

    def __init__(self, base_url: str, auth_token: Optional[str] = None):
        print(base_url)
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if auth_token:
            # Use X-Auth-Token header format like extract_training_rewards.py
            self.headers["X-Auth-Token"] = auth_token

    async def _make_request(self, method: str, endpoint: str, **kwargs):
        """Make an HTTP request to the API."""
        url = f"{self.base_url}{endpoint}"
        print(f"🔍 Making {method} request to: {url}")
        print(f"🔑 Headers: {self.headers}")
        if "json" in kwargs:
            print(f"📦 Payload: {kwargs['json']}")

        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, headers=self.headers, timeout=30.0, **kwargs)
            print(f"📨 Response status: {response.status_code}")
            if response.status_code >= 400:
                print(f"❌ Response body: {response.text}")
            response.raise_for_status()
            return response.json()

    async def get_policies(self, search_text: Optional[str] = None, page_size: int = 50):
        """Get available policies and training runs."""
        url = "/heatmap/policies"
        payload = {
            "search_text": search_text,  # Use None instead of empty string
            "pagination": {"page": 1, "page_size": page_size},
        }
        return await self._make_request("POST", url, json=payload)

    async def get_eval_names(self, training_run_ids: List[str], run_free_policy_ids):
        """Get evaluation names for selected policies."""
        url = "/heatmap/evals"
        payload = {"training_run_ids": training_run_ids, "run_free_policy_ids": run_free_policy_ids}
        return await self._make_request("POST", url, json=payload)

    async def get_available_metrics(
        self, training_run_ids: List[str], run_free_policy_ids: List[str], eval_names: List[str]
    ):
        """Get available metrics for selected policies and evaluations."""
        url = "/heatmap/metrics"
        payload = {
            "training_run_ids": training_run_ids,
            "run_free_policy_ids": run_free_policy_ids,
            "eval_names": eval_names,
        }
        return await self._make_request("POST", url, json=payload)

    async def generate_heatmap(
        self,
        training_run_ids: List[str],
        run_free_policy_ids: List[str],
        eval_names: List[str],
        metric: str,
        policy_selector: str = "best",
    ):
        """Generate heatmap data."""
        url = "/heatmap/heatmap"
        payload = {
            "training_run_ids": training_run_ids,
            "run_free_policy_ids": run_free_policy_ids,
            "eval_names": eval_names,
            "metric": metric,
            "training_run_policy_selector": policy_selector,
        }
        return await self._make_request("POST", url, json=payload)

    async def get_all_training_runs(self, search_text: Optional[str] = None, page_size: int = 100):
        """Get all training run names."""
        url = "/heatmap/policies"
        payload = {
            "search_text": search_text,  # Use None instead of empty string
            "pagination": {"page": 1, "page_size": page_size},
        }
        return await self._make_request("POST", url, json=payload)


async def test_api_connection_fixed(api_base_url: str, auth_token: str):
    """Test API connection with the fixed client."""
    print("🧪 Testing fixed API client...")
    client = MettaAPIClient(api_base_url, auth_token)

    try:
        # Test getting policies
        print("\n📋 Testing /heatmap/policies endpoint...")
        policies_response = await client.get_policies(page_size=5)
        print(f"✅ Success! Got response: {type(policies_response)}")
        if isinstance(policies_response, dict) and "policies" in policies_response:
            print(f"📊 Found {len(policies_response['policies'])} policies")
            if policies_response["policies"]:
                print(f"🔍 Sample policy: {policies_response['policies'][0]}")
        else:
            print(f"⚠️  Unexpected response structure: {policies_response}")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


# print("🚀 Testing the fixed API client...")
# await test_api_connection_fixed(api_base_url=api_base_url, auth_token=auth_token)


async def fetch_real_heatmap_data(
    training_run_names: List[str],
    metrics: List[str],
    policy_selector: str = "best",
    api_base_url: str = "http://localhost:8000",
    auth_token: Optional[str] = None,
    max_policies: int = 20,
) -> HeatmapWidget:
    """
    Fetch real evaluation data using the metta HTTP API (same as repo.ts).

    Args:
        training_run_names: List of training run names (e.g., ["daveey.arena.rnd.16x4.2"])
        metrics: List of metrics to include (e.g., ["reward", "heart.get"])
        policy_selector: "best" or "latest" policy selection strategy
        api_base_url: Base URL for the metta API from ~/.metta/observatory_tokens.yaml
        auth_token: Auth token from ~/.metta/observatory_tokens.yaml
        max_policies: Maximum number of policies to display

    Returns:
        HeatmapWidget with real data
    """
    try:
        client = MettaAPIClient(api_base_url, auth_token)

        # Step 1: Get available policies to find training run IDs
        policies_data = await client.get_policies(page_size=100)

        # Find training run IDs that match our training run names
        training_run_ids = []
        for policy in policies_data["policies"]:
            if policy["type"] == "training_run" and any(run_name in policy["name"] for run_name in training_run_names):
                training_run_ids.append(policy["id"])

        if not training_run_ids:
            print(f"❌ No training runs found matching: {training_run_names}")
            return create_heatmap_widget()

        # Step 2: Get available evaluations for these training runs
        eval_names = await client.get_eval_names(training_run_ids, [])
        if not eval_names:
            print("❌ No evaluations found for selected training runs")
            return create_heatmap_widget()

        # Step 3: Get available metrics
        available_metrics = await get_available_metrics(training_run_ids, eval_names, api_base_url)
        if not available_metrics:
            print("❌ No metrics found")
            return create_heatmap_widget()

        # Filter to requested metrics that actually exist
        valid_metrics = [m for m in metrics if m in available_metrics]
        if not valid_metrics:
            print(f"❌ None of the requested metrics {metrics} are available")
            return create_heatmap_widget()

        # Step 4: Generate heatmap for the first metric
        primary_metric = valid_metrics[0]
        keys = eval_names
        heatmap_data = await client.generate_heatmap(training_run_ids, [], keys, primary_metric, policy_selector)

        if not heatmap_data["policyNames"]:
            print("❌ No heatmap data generated")
            return create_heatmap_widget()

        # Limit policies if requested
        policy_names = heatmap_data["policyNames"]
        if len(policy_names) > max_policies:
            # Sort by average score and take top N
            avg_scores = heatmap_data["policyAverageScores"]
            top_policies = sorted(avg_scores.keys(), key=lambda p: avg_scores[p], reverse=True)[:max_policies]

            # Filter the data
            filtered_cells = {p: heatmap_data["cells"][p] for p in top_policies if p in heatmap_data["cells"]}
            heatmap_data["policyNames"] = top_policies
            heatmap_data["cells"] = filtered_cells
            heatmap_data["policyAverageScores"] = {p: avg_scores[p] for p in top_policies if p in avg_scores}

        # Step 5: Convert to widget format
        cells = {}
        for policy_name in heatmap_data["policyNames"]:
            cells[policy_name] = {}
            for eval_name in heatmap_data["evalNames"]:
                cell = heatmap_data["cells"].get(policy_name, {}).get(eval_name, {})
                cells[policy_name][eval_name] = {
                    "metrics": {primary_metric: cell.get("value", 0.0)},
                    "replayUrl": cell.get("replayUrl"),
                    "evalName": eval_name,
                }

        # Create widget
        widget = create_heatmap_widget()
        widget.set_multi_metric_data(
            cells=cells,
            eval_names=heatmap_data["evalNames"],
            policy_names=heatmap_data["policyNames"],
            metrics=[primary_metric],
            selected_metric=primary_metric,
        )

        return widget

    except httpx.ConnectError:
        print("❌ Could not connect to metta API server")
        print("💡 Check ~/.metta/observatory_tokens.yaml for the correct API base URL and auth token")
        print("💡 Check if app_backend server is running on http://localhost:8000")
        print("💡 You can start it with: cd app_backend && uv run python server.py")
        return create_heatmap_widget()
    except Exception as e:
        print(f"❌ Error fetching real data: {e}")
        return create_heatmap_widget()


async def get_available_policies(
    api_base_url: str = "http://localhost:8000", limit: int = 50, auth_token: Optional[str] = None
):
    """Get available policies and training runs."""
    try:
        client = MettaAPIClient(api_base_url, auth_token)
        return await client.get_policies(page_size=limit)
    except Exception as e:
        print(f"❌ Error fetching policies: {e}")
        return {"policies": []}


async def get_available_eval_names(
    training_run_ids: List[str], api_base_url: str = "http://localhost:8000", auth_token: Optional[str] = None
):
    """Get available evaluation names."""
    try:
        client = MettaAPIClient(api_base_url, auth_token)
        return await client.get_eval_names(training_run_ids, [])
    except Exception as e:
        print(f"❌ Error fetching eval names: {e}")
        return []


async def get_available_metrics(
    training_run_ids: List[str],
    eval_names: List[str],
    api_base_url: str = "http://localhost:8000",
    auth_token: Optional[str] = None,
):
    """Get available metrics."""
    try:
        client = MettaAPIClient(api_base_url, auth_token)

        # Debug: Let's try with a simplified version first
        print(f"🔍 Attempting to get metrics for {len(eval_names)} eval names...")
        print(f"📋 Eval names: {eval_names}")

        return await client.get_available_metrics(training_run_ids, [], eval_names)

    except Exception as e:
        print(f"❌ Error fetching metrics: {e}")
        print(f"📋 Eval names that caused the error: {eval_names}")
        print("🔧 This might be a server-side issue with processing eval_names list")

        # Let's try with a single eval name to see if it's a list processing issue
        if eval_names:
            try:
                print(f"🔄 Trying with just the first eval name: {eval_names[0]}")
                result = await client.get_available_metrics(training_run_ids, [], [eval_names[0]])
                print(f"✅ Single eval name worked! Got {len(result)} metrics")
                return result
            except Exception as e2:
                print(f"❌ Single eval name also failed: {e2}")

        print("💡 Falling back to common metrics...")
        # Return some common metrics as fallback
        return ["reward", "heart.get", "ore_red.get", "action.move.success"]
