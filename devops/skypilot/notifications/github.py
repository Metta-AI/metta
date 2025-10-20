from typing import Any

from devops.skypilot.notifications.notifier import NotificationBase, NotificationConfig
from devops.skypilot.utils.job_config import JobConfig
from gitta import post_commit_status


class GitHubNotifier(NotificationBase):
    def __init__(self):
        super().__init__("GitHub")

    def _validate_config(self, job_config: JobConfig) -> str | None:
        if not job_config.github_pat:
            return "Missing required field: github_pat"
        if not job_config.github_repository:
            return "Missing required field: github_repository"

        # Validate SHA format
        commit_sha = job_config.metta_git_ref or ""
        if not commit_sha:
            return "Missing required field: metta_git_ref"
        if len(commit_sha) != 40:
            return f"GitHub requires full length commit hash (40 chars), got {len(commit_sha)} chars"

        return None

    def _make_payload(self, notification: NotificationConfig, job_config: JobConfig) -> dict[str, Any]:
        description = notification.description
        if job_config.skypilot_job_id:
            description += f" - [ jl {job_config.skypilot_job_id} ]"

        target_url = None
        if job_config.metta_run_id:
            target_url = f"https://wandb.ai/metta-research/metta/runs/{job_config.metta_run_id}"

        payload = {
            "commit_sha": job_config.metta_git_ref,
            "state": notification.github_state,
            "repo": job_config.github_repository,
            "context": job_config.github_status_context,
            "description": description,
            "target_url": target_url,
            "token": job_config.github_pat,
        }
        return payload

    def _send(self, payload: dict[str, Any]) -> None:
        post_commit_status(**payload)
