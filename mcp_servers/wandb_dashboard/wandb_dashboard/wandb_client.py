"""
WandB API Client Wrapper

Wrapper around the WandB API for dashboard and metrics management.
Handles authentication, error handling, and provides convenient methods for MCP tools.
"""

import logging
from typing import Any, Dict, List, Optional

import wandb
from wandb import Api

logger = logging.getLogger(__name__)


class WandBClient:
    """Client wrapper for WandB API operations."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the WandB client.

        Args:
            api_key: Optional API key. If not provided, will use environment variable or login.
        """
        self.api = None
        self.api_key = api_key
        self._initialize_api()

    def _initialize_api(self) -> None:
        """Initialize the WandB API client."""
        try:
            if self.api_key:
                wandb.login(key=self.api_key)
            else:
                # This will use the API key from environment or prompt for login
                wandb.login()

            self.api = Api()
            logger.info("WandB API client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize WandB API: {e}")
            raise

    async def list_workspaces(
        self, entity: str, project: Optional[str] = None, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List available workspaces/dashboards for an entity/project.

        Args:
            entity: WandB entity (user or team)
            project: Optional project name. If None, lists across all projects.
            filters: Optional filters to apply

        Returns:
            List of workspace information dictionaries
        """
        try:
            # Note: The wandb_workspaces API might not have a direct list method
            # This is a placeholder that would need to be implemented based on
            # the actual API capabilities

            # Note: The wandb_workspaces API might not have a direct list method
            # This is a placeholder that would need to be implemented based on
            # the actual API capabilities when available

            if project:
                # Get workspaces for specific project
                try:
                    # This would need the actual workspace listing API
                    # project_obj = self.api.project(f"{entity}/{project}")
                    # workspaces = project_obj.list_workspaces()  # Hypothetical
                    pass
                except Exception as e:
                    logger.warning(f"Could not list workspaces for {entity}/{project}: {e}")
            else:
                # List workspaces across all accessible projects
                try:
                    projects = self.api.projects(entity=entity)
                    for proj in projects:
                        # This would need the actual workspace listing API
                        # proj_workspaces = proj.list_workspaces()  # Hypothetical
                        # workspaces.extend(proj_workspaces)
                        pass
                except Exception as e:
                    logger.warning(f"Could not list projects for entity {entity}: {e}")

            # For now, return placeholder data
            # This would be replaced with actual workspace data
            placeholder_workspaces = [
                {
                    "name": "Example Workspace",
                    "url": f"https://wandb.ai/{entity}/{project or 'example'}/workspace",
                    "entity": entity,
                    "project": project or "example",
                    "created_at": "2025-01-25T00:00:00Z",
                    "sections_count": 2,
                }
            ]

            return placeholder_workspaces

        except Exception as e:
            logger.error(f"Failed to list workspaces: {e}")
            raise

    async def get_project_metrics(
        self, entity: str, project: str, run_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get available metrics for a project.

        Args:
            entity: WandB entity
            project: WandB project name
            run_filters: Optional filters for runs

        Returns:
            List of metric information
        """
        try:
            logger.info(f"Getting metrics for {entity}/{project}")

            # Get the project
            project_obj = self.api.project(f"{entity}/{project}")

            # Get runs from the project
            runs = project_obj.runs()

            # Collect unique metrics across all runs
            all_metrics = set()
            metric_info = {}

            for run in runs:
                # Apply run filters if provided
                if run_filters:
                    # Simple filter implementation - can be expanded
                    if "state" in run_filters and run.state != run_filters["state"]:
                        continue
                    if "tags" in run_filters:
                        run_tags = set(run.tags)
                        filter_tags = set(run_filters["tags"])
                        if not filter_tags.intersection(run_tags):
                            continue

                # Get metrics from run history
                if hasattr(run, "history"):
                    try:
                        # Get a sample of the history to identify metrics
                        history_sample = run.history(samples=1)
                        if not history_sample.empty:
                            for column in history_sample.columns:
                                if column not in ["_step", "_runtime", "_timestamp"]:
                                    all_metrics.add(column)

                                    # Store metric info
                                    if column not in metric_info:
                                        metric_info[column] = {
                                            "name": column,
                                            "type": "numeric",  # Could be inferred from data
                                            "runs_count": 0,
                                            "sample_values": [],
                                        }

                                    metric_info[column]["runs_count"] += 1

                                    # Store sample value
                                    sample_val = history_sample[column].iloc[0]
                                    if len(metric_info[column]["sample_values"]) < 5:
                                        metric_info[column]["sample_values"].append(sample_val)

                    except Exception as e:
                        logger.warning(f"Could not get history for run {run.name}: {e}")
                        continue

                # Get metrics from run summary
                if hasattr(run, "summary"):
                    for key, value in run.summary.items():
                        if not key.startswith("_"):
                            all_metrics.add(key)

                            if key not in metric_info:
                                metric_info[key] = {
                                    "name": key,
                                    "type": "summary",
                                    "runs_count": 0,
                                    "sample_values": [],
                                }

                            metric_info[key]["runs_count"] += 1
                            if len(metric_info[key]["sample_values"]) < 5:
                                metric_info[key]["sample_values"].append(value)

            # Convert to list format
            metrics_list = []
            for metric_name in sorted(all_metrics):
                if metric_name in metric_info:
                    metrics_list.append(metric_info[metric_name])

            logger.info(f"Found {len(metrics_list)} metrics for {entity}/{project}")
            return metrics_list

        except Exception as e:
            logger.error(f"Failed to get project metrics: {e}")
            raise

    async def get_run_details(self, entity: str, project: str, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific run.

        Args:
            entity: WandB entity
            project: WandB project name
            run_id: Run ID

        Returns:
            Run details dictionary
        """
        try:
            run = self.api.run(f"{entity}/{project}/{run_id}")

            return {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "config": dict(run.config),
                "summary": dict(run.summary),
                "tags": run.tags,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "updated_at": run.updated_at.isoformat() if run.updated_at else None,
                "url": run.url,
            }

        except Exception as e:
            logger.error(f"Failed to get run details: {e}")
            raise

    def is_authenticated(self) -> bool:
        """Check if the client is properly authenticated."""
        try:
            # Try to access the API
            if self.api:
                # Simple check - try to get current user
                self.api.viewer
                return True
            return False
        except Exception as e:
            logger.warning(f"Authentication check failed: {e}")
            return False
