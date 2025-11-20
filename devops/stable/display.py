"""Display and formatting utilities for release validation.

Extracted from TaskRunner and release_stable.py to separate display concerns
from orchestration logic.
"""

from __future__ import annotations

import logging

from metta.common.util.text_styles import blue, cyan, green, magenta, red
from metta.jobs.job_config import JobConfig
from metta.jobs.job_display import format_artifact_link
from metta.jobs.job_state import JobState

logger = logging.getLogger("metta.jobs")


def format_artifact(value: str) -> str:
    """Format artifact URI with appropriate styling and icon.

    Deprecated: Use format_artifact_link from job_monitor instead.
    Kept for backward compatibility.
    """
    # Add colors to the basic format_artifact_link
    result = format_artifact_link(value)
    if value.startswith("wandb://") or value.startswith("s3://") or value.startswith("file://"):
        return magenta(result)
    elif value.startswith("http"):
        return cyan(result)
    return result


def format_job_result(
    job_state: JobState,
    job_config: JobConfig,
    acceptance_passed: bool,
    acceptance_error: str | None,
) -> str:
    """Format detailed job result for display.

    Args:
        job_state: Job state from JobManager
        job_config: Job configuration
        acceptance_passed: Whether acceptance criteria passed
        acceptance_error: Error message if acceptance failed

    Returns:
        Formatted multi-line string with job result details
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append(blue(f"📋 TASK RESULT: {job_config.name}"))
    lines.append("=" * 80)
    lines.append("")

    # Outcome
    if job_state.exit_code == 0 and acceptance_passed:
        lines.append(green("✅ Outcome: PASSED"))
    else:
        lines.append(red("❌ Outcome: FAILED"))

    # Exit code
    if job_state.exit_code != 0:
        lines.append(red(f"⚠️  Exit Code: {job_state.exit_code}"))
    else:
        lines.append(green(f"✓ Exit Code: {job_state.exit_code}"))

    # Acceptance criteria
    if not acceptance_passed:
        lines.append("")
        lines.append(red(f"❗ Acceptance Criteria Failed: {acceptance_error}"))

    # Metrics
    if job_state.metrics:
        lines.append("")
        lines.append("📊 Metrics:")
        for key, value in job_state.metrics.items():
            lines.append(f"   • {key}: {value:.4f}")

    # Artifacts
    artifacts = {}
    if job_state.wandb_run_id:
        artifacts["wandb_run_id"] = job_state.wandb_run_id
    if job_state.wandb_url:
        artifacts["wandb_url"] = job_state.wandb_url
    if job_state.checkpoint_uri:
        artifacts["checkpoint_uri"] = job_state.checkpoint_uri

    if artifacts:
        lines.append("")
        lines.append("📦 Artifacts:")
        for key, value in artifacts.items():
            highlighted = format_artifact(value)
            lines.append(f"   • {key}: {highlighted}")

    # Job ID and logs
    if job_state.job_id:
        lines.append("")
        lines.append(f"🆔 Job ID: {job_state.job_id}")

    if job_state.logs_path:
        lines.append(f"📝 Logs: {job_state.logs_path}")

    return "\n".join(lines)


def format_training_job_section(job_name: str, job_state: JobState) -> str:
    """Format training job information for release notes or summary.

    Args:
        job_name: Name of training job
        job_state: Job state from JobManager

    Returns:
        Formatted markdown-style training job section
    """
    lines = []

    lines.append(f"**{job_name}**")

    if job_state.wandb_url:
        lines.append(f"- WandB: {job_state.wandb_url}")

    if job_state.job_id:
        lines.append(f"- SkyPilot Job ID: {job_state.job_id}")
        lines.append(f"- View logs: `sky logs {job_state.job_id}`")

    if job_state.checkpoint_uri:
        lines.append(f"- Checkpoint: {job_state.checkpoint_uri}")

    if job_state.metrics:
        sps = job_state.metrics.get("overview/sps")
        if sps:
            lines.append(f"- Training throughput: {sps:.0f} SPS")

    return "\n".join(lines)
