"""Display and formatting utilities for release validation.

Extracted from TaskRunner and release_stable.py to separate display concerns
from orchestration logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from metta.common.util.text_styles import blue, cyan, green, magenta, red
from metta.jobs.job_config import JobConfig
from metta.jobs.job_monitor import format_artifact_link, format_job_status_line
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


def format_task_result(
    job_state: JobState,
    job_config: JobConfig,
    acceptance_passed: bool,
    acceptance_error: str | None,
) -> str:
    """Format detailed task result for display.

    Args:
        job_state: Job state from JobManager
        job_config: Job configuration
        acceptance_passed: Whether acceptance criteria passed
        acceptance_error: Error message if acceptance failed

    Returns:
        Formatted multi-line string with task result details
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


def format_failure_details(task_name: str, error: str | None, logs_path: str | None) -> str:
    """Format failure details for a task, including log tail.

    Args:
        task_name: Name of failed task
        error: Acceptance error message (if any)
        logs_path: Path to log file

    Returns:
        Formatted failure summary with log tail
    """
    lines = []

    # Task name and error
    if error:
        lines.append(f"  {red('✗')} {task_name}: {error}")
    else:
        lines.append(f"  {red('✗')} {task_name}: Job failed (see logs)")

    # Show last 5 lines of logs for quick debugging
    if logs_path:
        try:
            log_file = Path(logs_path)
            if log_file.exists():
                log_lines = log_file.read_text(errors="ignore").splitlines()
                if log_lines:
                    # Get last 5 non-empty lines
                    relevant_lines = [line for line in log_lines if line.strip()][-5:]
                    if relevant_lines:
                        lines.append(f"      Last {len(relevant_lines)} lines:")
                        for line in relevant_lines:
                            lines.append(f"      │ {line[:100]}")  # Truncate long lines
        except Exception:
            pass  # Don't crash if we can't read logs

    return "\n".join(lines)


def format_training_job_section(task_name: str, job_state: JobState) -> str:
    """Format training job information for release notes or summary.

    Args:
        task_name: Name of training task
        job_state: Job state from JobManager

    Returns:
        Formatted markdown-style training job section
    """
    lines = []

    lines.append(f"**{task_name}**")

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


def check_task_passed(job_state: JobState) -> bool:
    """Check if task passed (exit_code 0 + acceptance criteria).

    Args:
        job_state: Job state from JobManager

    Returns:
        True if task passed (exit code 0 and acceptance criteria met)
    """
    # Log the check for debugging
    passed = job_state.exit_code == 0 and job_state.acceptance_passed is not False

    logger.info(
        f"Task pass check for {job_state.name}: "
        f"exit_code={job_state.exit_code}, "
        f"acceptance_passed={job_state.acceptance_passed}, "
        f"result={'PASS' if passed else 'FAIL'}"
    )

    if job_state.exit_code != 0:
        logger.warning(
            f"Task {job_state.name} failed due to non-zero exit code: {job_state.exit_code} "
            f"(acceptance_passed={job_state.acceptance_passed})"
        )
        return False
    return job_state.acceptance_passed is not False


def format_task_with_acceptance(job_dict: dict, job_state: JobState) -> str:
    """Format job status integrated with acceptance criteria.

    Composes job_monitor primitives with job-level acceptance logic.

    Args:
        job_dict: Job dict from get_status_summary()
        job_state: Full job state for metrics and acceptance

    Returns:
        Multi-line formatted string with job status + acceptance + artifacts
    """
    lines = []

    # Job status line using primitive
    status_line = format_job_status_line(job_dict, show_duration=True)
    lines.append(status_line)

    # Check for launch/execution failures first (exit_code != 0 with no metrics)
    if job_dict["status"] == "completed" and job_dict["exit_code"] not in (0, None):
        has_metrics = job_state.metrics and any(k for k in job_state.metrics.keys() if not k.startswith("_"))

        if not has_metrics:
            # Job failed before producing metrics - likely a launch failure
            lines.append(red("  ✗ JOB FAILED TO LAUNCH"))
            lines.append(f"    Exit code: {job_dict['exit_code']}")
            lines.append(f"    Check logs for details: {job_dict.get('logs_path', 'N/A')}")
        elif job_state.config.acceptance_criteria:
            # Job ran but failed (has metrics) - show acceptance results
            if job_state.acceptance_passed:
                lines.append(green("  ✓ Acceptance criteria passed"))
            else:
                lines.append(red("  ✗ ACCEPTANCE CRITERIA FAILED"))

            # Show ALL criteria with pass/fail indicators
            for criterion in job_state.config.acceptance_criteria:
                if criterion.metric not in job_state.metrics:
                    lines.append(
                        red(f"    ✗ {criterion.metric}: MISSING (expected {criterion.operator} {criterion.threshold})")
                    )
                else:
                    actual = job_state.metrics[criterion.metric]
                    target_str = f"{criterion.operator} {criterion.threshold}"
                    if criterion.evaluate(actual):
                        lines.append(green(f"    ✓ {criterion.metric}: {actual:.1f} (target: {target_str})"))
                    else:
                        lines.append(red(f"    ✗ {criterion.metric}: {actual:.1f} (target: {target_str})"))
    # Acceptance criteria for successful jobs
    elif job_state.config.acceptance_criteria:
        if job_dict["status"] == "completed":
            # For completed jobs: show pass/fail with indicators
            if job_state.acceptance_passed:
                lines.append(green("  ✓ Acceptance criteria passed"))
            else:
                lines.append(red("  ✗ ACCEPTANCE CRITERIA FAILED"))

            # Show ALL criteria with pass/fail indicators
            for criterion in job_state.config.acceptance_criteria:
                if criterion.metric not in job_state.metrics:
                    lines.append(
                        red(f"    ✗ {criterion.metric}: MISSING (expected {criterion.operator} {criterion.threshold})")
                    )
                else:
                    actual = job_state.metrics[criterion.metric]
                    target_str = f"{criterion.operator} {criterion.threshold}"
                    # Use criterion's evaluate method
                    if criterion.evaluate(actual):
                        lines.append(green(f"    ✓ {criterion.metric}: {actual:.1f} (target: {target_str})"))
                    else:
                        lines.append(red(f"    ✗ {criterion.metric}: {actual:.1f} (target: {target_str})"))
        elif job_dict["status"] == "running":
            # For running jobs: show criteria without pass/fail (not decided yet)
            lines.append("  🎯 Acceptance criteria:")
            for criterion in job_state.config.acceptance_criteria:
                if criterion.metric in job_state.metrics:
                    actual = job_state.metrics[criterion.metric]
                    lines.append(
                        f"    • {criterion.metric}: {actual:.1f} (target: {criterion.operator} {criterion.threshold})"
                    )
                else:
                    lines.append(f"    • {criterion.metric}: (target: {criterion.operator} {criterion.threshold})")

    # Artifacts (using primitives)
    if job_dict.get("wandb_url"):
        artifact_str = format_artifact(job_dict["wandb_url"])
        lines.append(f"  {artifact_str}")

    if job_dict.get("checkpoint_uri"):
        artifact_str = format_artifact(job_dict["checkpoint_uri"])
        lines.append(f"  {artifact_str}")

    return "\n".join(lines)
