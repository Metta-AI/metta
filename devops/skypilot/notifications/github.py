import typing

import devops.skypilot.notifications.notifier
import devops.skypilot.utils.job_config
import gitta


class GitHubNotifier(devops.skypilot.notifications.notifier.NotificationBase):
    def __init__(self):
        super().__init__("GitHub")

    def _validate_config(self, job_config: devops.skypilot.utils.job_config.JobConfig) -> str | None:
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

    def _make_payload(
        self,
        notification: devops.skypilot.notifications.notifier.NotificationConfig,
        job_config: devops.skypilot.utils.job_config.JobConfig,
    ) -> dict[str, typing.Any]:
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

    def _send(self, payload: dict[str, typing.Any]) -> None:
        gitta.post_commit_status(**payload)
