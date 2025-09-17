#!/usr/bin/env python3
from typing import Any, Dict

from devops.skypilot.notifications.notifier import NotificationBase
from devops.skypilot.utils.job_config import JobConfig
from gitta import post_commit_status


class GitHubNotifier(NotificationBase):
    @property
    def name(self) -> str:
        return "GitHub"

    def get_required_fields(self, job_config: JobConfig) -> Dict[str, Any]:
        # Validate SHA format
        commit_sha = job_config.metta_git_ref or ""
        if commit_sha and len(commit_sha) < 40:
            raise ValueError(f'Github Status update requires a full length commit hash. len("{commit_sha}") < 40)')

        return {
            "commit_sha": commit_sha,
            "token": job_config.github_pat or "",
            "repository": job_config.github_repository or "",
            "context": job_config.github_status_context,
            "skypilot_job_id": job_config.skypilot_job_id,
            "metta_run_id": job_config.metta_run_id,
        }

    def format_notification(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        state = fields.get("state", "")
        description = fields.get("description", "")

        if not all([state, description]):
            raise ValueError("Missing state or description")

        # Build description with job ID
        desc = description
        if fields["skypilot_job_id"]:
            desc += f" - [ jl {fields['skypilot_job_id']} ]"

        # Build target URL
        target_url = None
        if fields["metta_run_id"]:
            target_url = f"https://wandb.ai/metta-research/metta/runs/{fields['metta_run_id']}"

        return {
            "commit_sha": fields["commit_sha"],
            "state": state,
            "repo": fields["repository"],
            "context": fields["context"],
            "description": desc,
            "target_url": target_url,
            "token": fields["token"],
        }

    def send(self, payload: Dict[str, Any]) -> None:
        post_commit_status(**payload)
