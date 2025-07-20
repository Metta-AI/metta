"""Client for sweep coordination API."""

from typing import Optional

import httpx

from metta.app_backend.routes.sweep_routes import (
    RunIdResponse,
    SweepCreateResponse,
    SweepInfo,
)


class SweepClient:
    """Client for interacting with the sweep coordination API."""

    def __init__(self, base_url: str = "http://localhost:8000", auth_token: Optional[str] = None):
        """
        Initialize the sweep client with automatic authentication if no token provided.

        Args:
            base_url: Base URL of the API server
            auth_token: Authentication token.
        """
        # Get machine token if no auth_token provided
        self.base_url = base_url.rstrip("/")
        self.headers = {}
        if auth_token:
            self.headers["X-Auth-Token"] = auth_token

    def create_sweep(self, sweep_name: str, project: str, entity: str, wandb_sweep_id: str) -> SweepCreateResponse:
        """Initialize a sweep (idempotent)."""
        response = httpx.post(
            f"{self.base_url}/sweeps/{sweep_name}/create_sweep",
            json={
                "project": project,
                "entity": entity,
                "wandb_sweep_id": wandb_sweep_id,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return SweepCreateResponse(**response.json())

    def get_sweep(self, sweep_name: str) -> SweepInfo:
        """Get sweep information."""
        response = httpx.get(
            f"{self.base_url}/sweeps/{sweep_name}",
            headers=self.headers,
        )
        response.raise_for_status()
        return SweepInfo(**response.json())

    def get_next_run_id(self, sweep_name: str) -> str:
        """Get the next run ID for a sweep (atomic operation)."""
        response = httpx.post(
            f"{self.base_url}/sweeps/{sweep_name}/runs/next",
            headers=self.headers,
        )
        response.raise_for_status()
        data = RunIdResponse(**response.json())
        return data.run_id


if __name__ == "__main__":
    # Example usage
    client = SweepClient()

    # Initialize a sweep
    result = client.create_sweep(
        sweep_name="protein_opt", project="my_project", entity="my_entity", wandb_sweep_id="wandb_12345"
    )
    print(f"Sweep initialized: created={result.created}, sweep_id={result.sweep_id}")

    # Get next run ID
    run_id = client.get_next_run_id("protein_opt")
    print(f"Next run ID: {run_id}")

    # Get sweep info
    info = client.get_sweep("protein_opt")
    print(f"Sweep info: exists={info.exists}")
