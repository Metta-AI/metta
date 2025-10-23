"""Asana metrics collector for Datadog monitoring."""

import datetime
from typing import Any

import asana

from devops.datadog.common.base import BaseCollector


class AsanaCollector(BaseCollector):
    """Collector for Asana task and project metrics.

    Collects comprehensive metrics about tasks, projects, team velocity, and
    specific project tracking (e.g., Bugs project with Triage/Active/Backlog).
    """

    def __init__(self, access_token: str, workspace_gid: str, bugs_project_gid: str | None = None):
        """Initialize Asana collector.

        Args:
            access_token: Asana API personal access token
            workspace_gid: Workspace ID to collect metrics from
            bugs_project_gid: Optional Bugs project ID for specific tracking
        """
        super().__init__(name="asana")
        self.client = asana.Client.access_token(access_token)
        self.workspace_gid = workspace_gid
        self.bugs_project_gid = bugs_project_gid

    def collect_metrics(self) -> dict[str, Any]:
        """Collect all Asana metrics."""
        metrics = {}
        metrics.update(self._collect_workspace_metrics())

        # Collect Bugs project metrics if configured
        if self.bugs_project_gid:
            metrics.update(self._collect_bugs_project_metrics())

        return metrics

    def _collect_workspace_metrics(self) -> dict[str, Any]:
        """Collect general workspace metrics."""
        metrics = {
            # Task status
            "asana.tasks.total": 0,
            "asana.tasks.open": 0,
            "asana.tasks.completed_7d": 0,
            "asana.tasks.completed_30d": 0,
            # Due date tracking
            "asana.tasks.overdue": 0,
            "asana.tasks.due_today": 0,
            "asana.tasks.due_this_week": 0,
            "asana.tasks.no_due_date": 0,
            # Assignment
            "asana.tasks.unassigned": 0,
            "asana.tasks.assigned": 0,
            # Velocity
            "asana.velocity.completed_per_day_7d": None,
            "asana.velocity.completion_rate_pct": None,
            # Cycle time (hours from creation to completion)
            "asana.cycle_time.avg_hours": None,
            "asana.cycle_time.p50_hours": None,
            "asana.cycle_time.p90_hours": None,
            # Team activity
            "asana.users.active_7d": 0,
            # Project health
            "asana.projects.active": 0,
            "asana.projects.on_track": 0,
            "asana.projects.at_risk": 0,
            "asana.projects.off_track": 0,
        }

        try:
            now = datetime.datetime.now(datetime.timezone.utc)
            seven_days_ago = now - datetime.timedelta(days=7)
            thirty_days_ago = now - datetime.timedelta(days=30)
            today_date = now.date()

            # Collect task data
            active_users = set()
            cycle_times = []

            # Get incomplete tasks
            tasks = self.client.tasks.find_all(
                {
                    "workspace": self.workspace_gid,
                    "completed_since": "now",  # Only incomplete tasks
                    "opt_fields": "completed,assignee,due_on,created_at,completed_at",
                }
            )

            for task in tasks:
                metrics["asana.tasks.total"] += 1
                metrics["asana.tasks.open"] += 1

                # Assignment tracking
                if task.get("assignee"):
                    metrics["asana.tasks.assigned"] += 1
                else:
                    metrics["asana.tasks.unassigned"] += 1

                # Due date tracking
                due_on = task.get("due_on")
                if due_on:
                    try:
                        due_date = datetime.datetime.strptime(due_on, "%Y-%m-%d").date()
                        if due_date < today_date:
                            metrics["asana.tasks.overdue"] += 1
                        elif due_date == today_date:
                            metrics["asana.tasks.due_today"] += 1
                        elif due_date <= today_date + datetime.timedelta(days=7):
                            metrics["asana.tasks.due_this_week"] += 1
                    except ValueError:
                        pass
                else:
                    metrics["asana.tasks.no_due_date"] += 1

            # Get completed tasks for velocity tracking
            completed_tasks = self.client.tasks.find_all(
                {
                    "workspace": self.workspace_gid,
                    "completed_since": seven_days_ago.isoformat(),
                    "opt_fields": "completed_at,created_at,assignee.name",
                }
            )

            completed_7d = 0
            completed_30d = 0

            for task in completed_tasks:
                completed_at_str = task.get("completed_at")
                if not completed_at_str:
                    continue

                completed_at = datetime.datetime.fromisoformat(completed_at_str.replace("Z", "+00:00"))

                if completed_at >= seven_days_ago:
                    completed_7d += 1

                    # Track active users
                    assignee = task.get("assignee")
                    if assignee and assignee.get("name"):
                        active_users.add(assignee["name"])

                if completed_at >= thirty_days_ago:
                    completed_30d += 1

                # Calculate cycle time
                created_at_str = task.get("created_at")
                if created_at_str:
                    created_at = datetime.datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                    cycle_time_hours = (completed_at - created_at).total_seconds() / 3600
                    cycle_times.append(cycle_time_hours)

            metrics["asana.tasks.completed_7d"] = completed_7d
            metrics["asana.tasks.completed_30d"] = completed_30d

            # Calculate velocity
            if completed_7d > 0:
                metrics["asana.velocity.completed_per_day_7d"] = completed_7d / 7.0

            # Calculate cycle time statistics
            if cycle_times:
                cycle_times.sort()
                n = len(cycle_times)
                metrics["asana.cycle_time.avg_hours"] = sum(cycle_times) / n
                metrics["asana.cycle_time.p50_hours"] = cycle_times[int(n * 0.50)]
                metrics["asana.cycle_time.p90_hours"] = cycle_times[int(n * 0.90)]

            # Team activity
            metrics["asana.users.active_7d"] = len(active_users)

            # Project health
            projects = self.client.projects.find_all(
                {
                    "workspace": self.workspace_gid,
                    "archived": False,
                    "opt_fields": "current_status_update.status_type",
                }
            )

            for project in projects:
                metrics["asana.projects.active"] += 1

                status_update = project.get("current_status_update")
                if status_update:
                    status_type = status_update.get("status_type")
                    if status_type == "on_track":
                        metrics["asana.projects.on_track"] += 1
                    elif status_type == "at_risk":
                        metrics["asana.projects.at_risk"] += 1
                    elif status_type == "off_track":
                        metrics["asana.projects.off_track"] += 1

        except Exception as e:
            self.logger.error(f"Failed to collect workspace metrics: {e}")
            # Set all metrics to None on error
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
            tasks = self.client.tasks.find_all(
                {
                    "project": self.bugs_project_gid,
                    "completed_since": "now",  # Only incomplete
                    "opt_fields": "memberships.section.name,created_at",
                }
            )

            bug_ages = []

            for task in tasks:
                metrics["asana.projects.bugs.total_open"] += 1

                # Determine section
                section_name = "other"
                memberships = task.get("memberships", [])
                for membership in memberships:
                    section = membership.get("section")
                    if section:
                        name = section.get("name", "").lower()
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
                created_at_str = task.get("created_at")
                if created_at_str:
                    created_at = datetime.datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
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
            completed_tasks = self.client.tasks.find_all(
                {
                    "project": self.bugs_project_gid,
                    "completed_since": thirty_days_ago.isoformat(),
                    "opt_fields": "completed_at",
                }
            )

            for task in completed_tasks:
                completed_at_str = task.get("completed_at")
                if not completed_at_str:
                    continue

                completed_at = datetime.datetime.fromisoformat(completed_at_str.replace("Z", "+00:00"))

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
