from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import requests

logger = logging.getLogger(__name__)


class GitHubClient:
    """Light wrapper over GitHub's REST API v3."""

    def __init__(self, token: str | None = None, *, base_url: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("GITHUB_API_URL", "https://api.github.com")
        self.session = requests.Session()
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": os.environ.get("GITHUB_USER_AGENT", "metta-infra-health"),
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
            logger.debug("GitHub client initialized with authentication token")
        else:
            logger.warning("GitHub client initialized without authentication token - rate limits will be lower")
        self.session.headers.update(headers)

    def _get(self, path: str, *, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params, timeout=30)
        try:
            response.raise_for_status()
        except requests.HTTPError:
            logger.error("GitHub API error (%s): %s", response.status_code, response.text)
            raise
        return response.json()

    def list_workflow_runs(
        self,
        repo: str,
        *,
        per_page: int = 100,
        branch: str | None = None,
        event: str | None = None,
        status: str | None = "completed",
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"per_page": per_page}
        if branch:
            params["branch"] = branch
        if event:
            params["event"] = event
        if status:
            params["status"] = status

        data = self._get(f"/repos/{repo}/actions/runs", params=params)
        return data.get("workflow_runs", [])

    def list_workflows(self, repo: str) -> List[Dict[str, Any]]:
        """List all workflows in a repository."""
        data = self._get(f"/repos/{repo}/actions/workflows")
        return data.get("workflows", [])

    def get_latest_workflow_run(
        self,
        repo: str,
        *,
        workflow_id: int | str | None = None,
        workflow_name: str | None = None,
        branch: str | None = None,
    ) -> Dict[str, Any] | None:
        """Get the latest run for a specific workflow."""
        if workflow_id is None and workflow_name is None:
            raise ValueError("Either workflow_id or workflow_name must be provided")

        # If workflow_name provided, find workflow_id first
        if workflow_id is None:
            workflows = self.list_workflows(repo)
            for workflow in workflows:
                wf_name = workflow.get("name") or ""
                wf_path = workflow.get("path", "")
                # Try exact match on name (case-insensitive)
                if wf_name.lower() == workflow_name.lower():
                    workflow_id = workflow.get("id")
                    logger.debug("Matched workflow '%s' to '%s' (exact name)", workflow_name, wf_name)
                    break
                # Try match on path (filename)
                if wf_path:
                    path_filename = wf_path.split("/")[-1]
                    if path_filename.lower() == workflow_name.lower():
                        workflow_id = workflow.get("id")
                        logger.debug("Matched workflow '%s' to path '%s'", workflow_name, wf_path)
                        break
            if workflow_id is None:
                # List available workflows for debugging
                available = [f"{w.get('name', 'N/A')} (ID: {w.get('id')})" for w in workflows[:10]]
                logger.warning(
                    "Workflow not found: '%s'. Available workflows (first 10): %s",
                    workflow_name,
                    available,
                )
                return None

        params: Dict[str, Any] = {"per_page": 1}
        if branch:
            params["branch"] = branch

        try:
            data = self._get(f"/repos/{repo}/actions/workflows/{workflow_id}/runs", params=params)
            runs = data.get("workflow_runs", [])
            return runs[0] if runs else None
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning("Workflow runs not found for workflow_id=%s", workflow_id)
                return None
            raise

    def search_issues(self, query: str) -> Dict[str, Any]:
        params = {"q": query}
        return self._get("/search/issues", params=params)

    @staticmethod
    def isoformat(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
