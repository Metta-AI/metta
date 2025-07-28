# 🔧 Fixed API Client Implementation
from typing import List, Optional

from metta.app_backend.eval_task_client import EvalTaskClient
from metta.app_backend.routes.heatmap_routes import HeatmapData, PoliciesResponse


class MettaAPIClient(EvalTaskClient):
    """Fixed client that properly handles authentication and response parsing."""

    def __init__(self, base_url: str):
        super().__init__(base_url)

    async def get_policies(self, search_text: Optional[str] = None, page_size: int = 50):
        """Get available policies and training runs."""
        url = "/heatmap/policies"
        payload = {
            "search_text": search_text,  # Use None instead of empty string
            "pagination": {"page": 1, "page_size": page_size},
        }
        return await self._make_request(PoliciesResponse, "POST", url, json=payload)

    async def get_eval_names(self, training_run_ids: List[str], run_free_policy_ids):
        """Get evaluation names for selected policies."""
        url = "/heatmap/evals"
        payload = {"training_run_ids": training_run_ids, "run_free_policy_ids": run_free_policy_ids}
        headers = {"X-Auth-Token": self._machine_token} if self._machine_token else {}
        response = await self._http_client.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

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
        headers = {"X-Auth-Token": self._machine_token} if self._machine_token else {}
        response = await self._http_client.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

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
        return await self._make_request(HeatmapData, "POST", url, json=payload)

    async def get_all_training_runs(self, search_text: Optional[str] = None, page_size: int = 100):
        """Get all training run names."""
        url = "/heatmap/policies"
        payload = {
            "search_text": search_text,  # Use None instead of empty string
            "pagination": {"page": 1, "page_size": page_size},
        }
        return await self._make_request(PoliciesResponse, "POST", url, json=payload)


async def test_api_connection_fixed(api_base_url: str):
    """Test API connection with the fixed client."""
    print("🧪 Testing fixed API client...")
    client = MettaAPIClient(api_base_url)

    try:
        # Test getting policies
        print("\n📋 Testing /heatmap/policies endpoint...")
        policies_response = await client.get_policies(page_size=5)
        print(f"✅ Success! Got response: {type(policies_response)}")
        if hasattr(policies_response, "policies"):
            print(f"📊 Found {len(policies_response.policies)} policies")
            if policies_response.policies:
                print(f"🔍 Sample policy: {policies_response.policies[0]}")
        else:
            print(f"⚠️  Unexpected response structure: {policies_response}")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


# print("🚀 Testing the fixed API client...")
# await test_api_connection_fixed(api_base_url=api_base_url, auth_token=auth_token)
