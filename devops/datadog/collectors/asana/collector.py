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
        super().__init__(name="asana")
        configuration = asana.Configuration()
        configuration.access_token = access_token
        self.api_client = asana.ApiClient(configuration)

        self.tasks_api = asana.TasksApi(self.api_client)
        self.projects_api = asana.ProjectsApi(self.api_client)
        self.workspaces_api = asana.WorkspacesApi(self.api_client)

        self.workspace_gid = workspace_gid
        self.bugs_project_gid = bugs_project_gid

    def collect_metrics(self) -> dict[str, Any]:
        metrics = {}

        if self.bugs_project_gid:
            metrics.update(self._collect_bugs_project_metrics())
        else:
            self.logger.warning("No bugs_project_gid configured, skipping Bugs project metrics")

        metrics.update(self._collect_workspace_projects())

        return metrics

    def _collect_workspace_projects(self) -> dict[str, Any]:
        metrics = {}

        try:
            url = (
                f"/projects?workspace={self.workspace_gid}&archived=false&opt_fields=current_status_update.status_type"
            )
            response_data, status_code, headers = self.api_client.call_api(
                url, "GET", response_type="object", auth_settings=["token"]
            )

            projects = response_data.get("data", []) if isinstance(response_data, dict) else []

            active_count = 0
            on_track_count = 0
            at_risk_count = 0
            off_track_count = 0

            for project in projects:
                active_count += 1

                if "current_status_update" in project and project["current_status_update"]:
                    status_update = project["current_status_update"]
                    status_type = status_update.get("status_type")

                    if status_type == "on_track":
                        on_track_count += 1
                    elif status_type == "at_risk":
                        at_risk_count += 1
                    elif status_type == "off_track":
                        off_track_count += 1

            metrics["asana.projects"] = [
                (active_count, ["status:active"]),
                (on_track_count, ["status:on_track"]),
                (at_risk_count, ["status:at_risk"]),
                (off_track_count, ["status:off_track"]),
            ]

        except Exception as e:
            self.logger.error(f"Failed to collect workspace project metrics: {e}", exc_info=True)

        return metrics

    def _collect_bugs_project_metrics(self) -> dict[str, Any]:
        metrics = {}

        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            seven_days_ago = now - datetime.timedelta(days=7)
            thirty_days_ago = now - datetime.timedelta(days=30)

            opts = {
                "project": self.bugs_project_gid,
                "completed_since": now,
                "opt_fields": ["memberships.section.name", "created_at"],
            }

            response = self.tasks_api.get_tasks(opts)
            tasks = response.data if hasattr(response, "data") else []

            triage_count = 0
            active_count = 0
            backlog_count = 0
            other_count = 0
            total_open = 0
            created_7d = 0
            bug_ages = []

            for task in tasks:
                total_open += 1

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

                if section_name == "triage":
                    triage_count += 1
                elif section_name == "active":
                    active_count += 1
                elif section_name == "backlog":
                    backlog_count += 1
                else:
                    other_count += 1

                created_at = getattr(task, "created_at", None) if hasattr(task, "created_at") else None
                if created_at:
                    if isinstance(created_at, str):
                        created_at = datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    age_days = (now - created_at).total_seconds() / 86400
                    bug_ages.append(age_days)

                    if created_at >= seven_days_ago:
                        created_7d += 1

            metrics["asana.bugs"] = [
                (total_open, ["status:open"]),
                (triage_count, ["section:triage"]),
                (active_count, ["section:active"]),
                (backlog_count, ["section:backlog"]),
                (other_count, ["section:other"]),
                (created_7d, ["event:created", "timeframe:7d"]),
            ]

            if bug_ages:
                avg_age = sum(bug_ages) / len(bug_ages)
                max_age = max(bug_ages)
                metrics["asana.bugs.age_days"] = [
                    (avg_age, ["metric:avg"]),
                    (max_age, ["metric:max"]),
                ]

            opts_completed = {
                "project": self.bugs_project_gid,
                "completed_since": thirty_days_ago,
                "opt_fields": ["completed_at"],
            }

            response_completed = self.tasks_api.get_tasks(opts_completed)
            completed_tasks = response_completed.data if hasattr(response_completed, "data") else []

            completed_7d = 0
            completed_30d = 0

            for task in completed_tasks:
                completed_at = getattr(task, "completed_at", None) if hasattr(task, "completed_at") else None
                if not completed_at:
                    continue

                if isinstance(completed_at, str):
                    completed_at = datetime.datetime.fromisoformat(completed_at.replace("Z", "+00:00"))

                if completed_at >= seven_days_ago:
                    completed_7d += 1
                if completed_at >= thirty_days_ago:
                    completed_30d += 1

            if "asana.bugs" not in metrics:
                metrics["asana.bugs"] = []

            metrics["asana.bugs"].extend(
                [
                    (completed_7d, ["event:completed", "timeframe:7d"]),
                    (completed_30d, ["event:completed", "timeframe:30d"]),
                ]
            )

        except Exception as e:
            self.logger.error(f"Failed to collect Bugs project metrics: {e}")

        return metrics
