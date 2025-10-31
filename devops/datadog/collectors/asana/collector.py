"""Asana metrics collector for Datadog monitoring."""

import datetime
from typing import Any

import asana

from devops.datadog.utils.base import BaseCollector


class AsanaCollector(BaseCollector):
    """Collector for Asana task and project metrics.

    Collects comprehensive metrics about tasks, projects, team velocity, and
    specific project tracking (e.g., Bugs project with Triage/Active/Backlog).

    Note: Uses Asana Python SDK v5+ API
    """

    def __init__(self, access_token: str, workspace_gid: str, bugs_project_gid: str | None = None):
        """Initialize Asana collector.

        Args:
            access_token: Asana API personal access token
            workspace_gid: Workspace ID to collect metrics from
            bugs_project_gid: Optional Bugs project ID for specific tracking
        """
        super().__init__(name="asana")
        # Configure Asana API client
        configuration = asana.Configuration()
        configuration.access_token = access_token
        self.api_client = asana.ApiClient(configuration)

        # Initialize API endpoints
        self.tasks_api = asana.TasksApi(self.api_client)
        self.projects_api = asana.ProjectsApi(self.api_client)
        self.workspaces_api = asana.WorkspacesApi(self.api_client)

        self.workspace_gid = workspace_gid
        self.bugs_project_gid = bugs_project_gid

    def collect_metrics(self) -> dict[str, Any]:
        """Collect all Asana metrics."""
        metrics = {}

        # Collect Bugs project metrics if configured
        if self.bugs_project_gid:
            metrics.update(self._collect_bugs_project_metrics())
        else:
            self.logger.warning("No bugs_project_gid configured, skipping Bugs project metrics")

        # Collect workspace project stats
        metrics.update(self._collect_workspace_projects())

        return metrics

    def _collect_workspace_projects(self) -> dict[str, Any]:
        """Collect workspace-level project metrics."""
        metrics = {
            "asana.projects.active": 0,
            "asana.projects.on_track": 0,
            "asana.projects.at_risk": 0,
            "asana.projects.off_track": 0,
        }

        try:
            # Use direct API call - the SDK's get_projects() has issues returning data
            # See: https://github.com/Asana/python-asana/issues
            url = (
                f"/projects?workspace={self.workspace_gid}&archived=false&opt_fields=current_status_update.status_type"
            )
            response_data, status_code, headers = self.api_client.call_api(
                url, "GET", response_type="object", auth_settings=["token"]
            )

            projects = response_data.get("data", []) if isinstance(response_data, dict) else []

            for project in projects:
                metrics["asana.projects.active"] += 1

                # Get status from project object
                if "current_status_update" in project and project["current_status_update"]:
                    status_update = project["current_status_update"]
                    status_type = status_update.get("status_type")

                    if status_type == "on_track":
                        metrics["asana.projects.on_track"] += 1
                    elif status_type == "at_risk":
                        metrics["asana.projects.at_risk"] += 1
                    elif status_type == "off_track":
                        metrics["asana.projects.off_track"] += 1

        except Exception as e:
            self.logger.error(f"Failed to collect workspace project metrics: {e}", exc_info=True)
            for key in metrics:
                metrics[key] = None

        return metrics

    def _collect_bugs_project_metrics(self) -> dict[str, Any]:
        """Collect metrics for the Bugs project, tracking sections and velocity."""
        metrics = {
            # Section counts
            "asana.projects.bugs.triage_count": 0,
            "asana.projects.bugs.active_count": 0,
            "asana.projects.bugs.backlog_count": 0,
            "asana.projects.bugs.other_count": 0,
            "asana.projects.bugs.total_open": 0,
            # Velocity
            "asana.projects.bugs.completed_7d": 0,
            "asana.projects.bugs.completed_30d": 0,
            "asana.projects.bugs.created_7d": 0,
            # Aging
            "asana.projects.bugs.avg_age_days": None,
            "asana.projects.bugs.oldest_bug_days": None,
        }

        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            seven_days_ago = now - datetime.timedelta(days=7)
            thirty_days_ago = now - datetime.timedelta(days=30)

            # Get all open tasks in Bugs project
            opts = {
                "project": self.bugs_project_gid,
                "completed_since": now,  # Only incomplete tasks
                "opt_fields": ["memberships.section.name", "created_at"],
            }

            response = self.tasks_api.get_tasks(opts)
            tasks = response.data if hasattr(response, "data") else []

            bug_ages = []

            for task in tasks:
                metrics["asana.projects.bugs.total_open"] += 1

                # Determine section
                section_name = "other"
                memberships = getattr(task, "memberships", []) if hasattr(task, "memberships") else []

                for membership in memberships:
                    section = (
                        membership.get("section")
                        if isinstance(membership, dict)
                        else getattr(membership, "section", None)
                    )
                    if section:
                        name = (
                            section.get("name", "").lower()
                            if isinstance(section, dict)
                            else getattr(section, "name", "").lower()
                        )
                        if "triage" in name:
                            section_name = "triage"
                        elif "active" in name:
                            section_name = "active"
                        elif "backlog" in name:
                            section_name = "backlog"
                        break

                # Increment section count
                if section_name == "triage":
                    metrics["asana.projects.bugs.triage_count"] += 1
                elif section_name == "active":
                    metrics["asana.projects.bugs.active_count"] += 1
                elif section_name == "backlog":
                    metrics["asana.projects.bugs.backlog_count"] += 1
                else:
                    metrics["asana.projects.bugs.other_count"] += 1

                # Calculate age
                created_at = getattr(task, "created_at", None) if hasattr(task, "created_at") else None
                if created_at:
                    if isinstance(created_at, str):
                        created_at = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    age_days = (now - created_at).total_seconds() / 86400
                    bug_ages.append(age_days)

                    # Track creation rate
                    if created_at >= seven_days_ago:
                        metrics["asana.projects.bugs.created_7d"] += 1

            # Calculate aging statistics
            if bug_ages:
                metrics["asana.projects.bugs.avg_age_days"] = sum(bug_ages) / len(bug_ages)
                metrics["asana.projects.bugs.oldest_bug_days"] = max(bug_ages)

            # Get completed bugs
            opts_completed = {
                "project": self.bugs_project_gid,
                "completed_since": thirty_days_ago,
                "opt_fields": ["completed_at"],
            }

            response_completed = self.tasks_api.get_tasks(opts_completed)
            completed_tasks = response_completed.data if hasattr(response_completed, "data") else []

            for task in completed_tasks:
                completed_at = getattr(task, "completed_at", None) if hasattr(task, "completed_at") else None
                if not completed_at:
                    continue

                if isinstance(completed_at, str):
                    completed_at = datetime.datetime.fromisoformat(completed_at.replace("Z", "+00:00"))

                if completed_at >= seven_days_ago:
                    metrics["asana.projects.bugs.completed_7d"] += 1
                if completed_at >= thirty_days_ago:
                    metrics["asana.projects.bugs.completed_30d"] += 1

        except Exception as e:
            self.logger.error(f"Failed to collect Bugs project metrics: {e}")
            # Set all metrics to None on error
            for key in metrics:
                metrics[key] = None

        return metrics
