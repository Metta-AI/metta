"""Display and formatting utilities for release validation.

Extracted from TaskRunner and release_stable.py to separate display concerns
from orchestration logic.
"""


import logging

import metta.common.util.text_styles
import metta.jobs.job_config
import metta.jobs.job_display
import metta.jobs.job_state

logger = logging.getLogger("metta.jobs")


def format_artifact(value: str) -> str:
    """Format artifact URI with appropriate styling and icon.

    Deprecated: Use format_artifact_link from job_monitor instead.
    Kept for backward compatibility.
    """
    # Add colors to the basic format_artifact_link
    result = metta.jobs.job_display.format_artifact_link(value)
    if value.startswith("wandb://") or value.startswith("s3://") or value.startswith("file://"):
        return metta.common.util.text_styles.magenta(result)
    elif value.startswith("http"):
        return metta.common.util.text_styles.cyan(result)
    return result


def format_job_result(
    job_state: metta.jobs.job_state.JobState,
    job_config: metta.jobs.job_config.JobConfig,
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
    lines.append(metta.common.util.text_styles.blue(f"📋 TASK RESULT: {job_config.name}"))
    lines.append("=" * 80)
    lines.append("")

    # Outcome
    if job_state.exit_code == 0 and acceptance_passed:
        lines.append(metta.common.util.text_styles.green("✅ Outcome: PASSED"))
    else:
        lines.append(metta.common.util.text_styles.red("❌ Outcome: FAILED"))

    # Exit code
    if job_state.exit_code != 0:
        lines.append(metta.common.util.text_styles.red(f"⚠️  Exit Code: {job_state.exit_code}"))
    else:
        lines.append(metta.common.util.text_styles.green(f"✓ Exit Code: {job_state.exit_code}"))

    # Acceptance criteria
    if not acceptance_passed:
        lines.append("")
        lines.append(metta.common.util.text_styles.red(f"❗ Acceptance Criteria Failed: {acceptance_error}"))

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


def format_training_job_section(job_name: str, job_state: metta.jobs.job_state.JobState) -> str:
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


def format_job_with_acceptance(job_state: metta.jobs.job_state.JobState) -> str:
    """Format job status integrated with acceptance criteria.

    Composes job_monitor primitives with job-level acceptance logic.

    Args:
        job_state: Job state with status, metrics and acceptance

    Returns:
        Multi-line formatted string with job status + acceptance + artifacts
    """
    lines = []

    # Job status line using primitive
    status_line = metta.jobs.job_display.format_job_status_line(job_state, show_duration=True)
    lines.append(status_line)

    # Check for launch/execution failures first (exit_code != 0 with no metrics)
    if job_state.status == "completed" and job_state.exit_code not in (0, None):
        has_metrics = job_state.metrics and any(k for k in job_state.metrics.keys() if not k.startswith("_"))

        if not has_metrics:
            # Job failed before producing metrics - distinguish between cancelled and launch failure
            if job_state.exit_code == 130:
                lines.append(metta.common.util.text_styles.red("  ✗ JOB CANCELLED"))
                lines.append("    Job was interrupted (SIGINT)")
            else:
                lines.append(metta.common.util.text_styles.red("  ✗ JOB FAILED TO LAUNCH"))
                lines.append(f"    Exit code: {job_state.exit_code}")
                lines.append(f"    Check logs for details: {job_state.logs_path or 'N/A'}")
        elif job_state.config.acceptance_criteria:
            # Job ran but failed (has metrics) - show acceptance results
            if job_state.acceptance_passed:
                lines.append(metta.common.util.text_styles.green("  ✓ Acceptance criteria passed"))
            else:
                lines.append(metta.common.util.text_styles.red("  ✗ ACCEPTANCE CRITERIA FAILED"))

            # Show ALL criteria with pass/fail indicators
            for criterion in job_state.config.acceptance_criteria:
                if criterion.metric not in job_state.metrics:
                    lines.append(
                        metta.common.util.text_styles.red(
                            f"    ✗ {criterion.metric}: MISSING (expected {criterion.operator} {criterion.threshold})"
                        )
                    )
                else:
                    actual = job_state.metrics[criterion.metric]
                    target_str = f"{criterion.operator} {criterion.threshold}"
                    if criterion.evaluate(actual):
                        lines.append(
                            metta.common.util.text_styles.green(
                                f"    ✓ {criterion.metric}: {actual:.1f} (target: {target_str})"
                            )
                        )
                    else:
                        lines.append(
                            metta.common.util.text_styles.red(
                                f"    ✗ {criterion.metric}: {actual:.1f} (target: {target_str})"
                            )
                        )
    # Acceptance criteria for successful jobs
    elif job_state.config.acceptance_criteria:
        if job_state.status == "completed":
            # For completed jobs: show pass/fail with indicators
            if job_state.acceptance_passed:
                lines.append(metta.common.util.text_styles.green("  ✓ Acceptance criteria passed"))
            else:
                lines.append(metta.common.util.text_styles.red("  ✗ ACCEPTANCE CRITERIA FAILED"))

            # Show ALL criteria with pass/fail indicators
            for criterion in job_state.config.acceptance_criteria:
                if criterion.metric not in job_state.metrics:
                    lines.append(
                        metta.common.util.text_styles.red(
                            f"    ✗ {criterion.metric}: MISSING (expected {criterion.operator} {criterion.threshold})"
                        )
                    )
                else:
                    actual = job_state.metrics[criterion.metric]
                    target_str = f"{criterion.operator} {criterion.threshold}"
                    # Use criterion's evaluate method
                    if criterion.evaluate(actual):
                        lines.append(
                            metta.common.util.text_styles.green(
                                f"    ✓ {criterion.metric}: {actual:.1f} (target: {target_str})"
                            )
                        )
                    else:
                        lines.append(
                            metta.common.util.text_styles.red(
                                f"    ✗ {criterion.metric}: {actual:.1f} (target: {target_str})"
                            )
                        )
        elif job_state.status == "running":
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
    if job_state.wandb_url:
        artifact_str = format_artifact(job_state.wandb_url)
        lines.append(f"  {artifact_str}")

    if job_state.checkpoint_uri:
        artifact_str = format_artifact(job_state.checkpoint_uri)
        lines.append(f"  {artifact_str}")

    return "\n".join(lines)
