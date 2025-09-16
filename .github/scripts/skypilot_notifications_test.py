#!/usr/bin/env python3
"""Test script for SkyPilot notifications system."""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

this_file = Path(__file__).resolve()
project_root = this_file.parents[2]

if not (project_root / "pyproject.toml").is_file():
    raise RuntimeError(f"Project root validation failed: {project_root}")

sys.path.insert(0, str(project_root))

from devops.skypilot.notifications.notification import NotificationManager  # noqa: E402
from devops.skypilot.utils.job_config import JobConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_test_job_config(scenario: str) -> JobConfig:
    """Create a test JobConfig for the given scenario."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    config = JobConfig(
        metta_run_id=f"test_notify_{scenario}_{timestamp}",
        skypilot_task_id=f"test-task-{scenario}",
        skypilot_job_id=f"test-{os.environ.get('GITHUB_RUN_NUMBER', '0')}",
        metta_git_ref=os.environ.get("GITHUB_SHA", "unknown"),
    )

    # Notification settings from environment
    config.discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    config.github_pat = os.environ.get("GITHUB_PAT")
    config.enable_github_status = os.environ.get("SKIP_GITHUB_STATUS", "false").lower() != "true"
    config.enable_wandb_alerts = bool(os.environ.get("WANDB_API_KEY"))

    # Override defaults if provided
    if repo := os.environ.get("GITHUB_REPOSITORY"):
        config.github_repository = repo
    config.github_status_context = f"test/notifications/{scenario}"

    # W&B configuration
    if entity := os.environ.get("WANDB_ENTITY"):
        config.wandb_entity = entity
    if project := os.environ.get("WANDB_PROJECT"):
        config.wandb_project = project

    # Timing
    config.start_time = int(time.time()) - 3600

    # Scenario-specific settings
    if scenario == "heartbeat_timeout":
        config.heartbeat_timeout = 300
    elif scenario == "max_runtime_reached":
        config.max_runtime_hours = 1.0
    elif scenario == "rapid_restarts":
        config.restart_count = 5
        config.accumulated_runtime_sec = 600

    # Test metadata directory
    config.job_metadata_dir = f"/tmp/test_notifications_{scenario}"
    os.makedirs(config.job_metadata_dir, exist_ok=True)

    return config


def test_scenario(scenario: str) -> tuple[bool, str]:
    """Test a specific notification scenario."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing scenario: {scenario}")
    logger.info(f"{'=' * 60}")

    try:
        job_config = create_test_job_config(scenario)
        logger.info(f"Using GITHUB_SHA: {job_config.metta_git_ref}")

        manager = NotificationManager(job_config)
        manager.log_config()

        # Map scenarios to termination reasons
        termination_reasons = {
            "job_completed": "job_completed",
            "job_failed_137": "job_failed_137",
            "heartbeat_timeout": "heartbeat_timeout",
            "nccl_tests_failed": "nccl_tests_failed",
            "rapid_restarts": "rapid_restarts",
            "max_runtime_reached": "max_runtime_reached",
        }

        termination_reason = termination_reasons.get(scenario, scenario)

        # Send notifications
        results = manager.send_notifications(termination_reason)

        # Format results summary
        if results:
            summary = ", ".join(f"{name}: {'✅' if success else '❌'}" for name, success in results.items())
            message = f"Scenario '{scenario}' completed. Results: {summary}"
        else:
            message = f"Scenario '{scenario}' completed. No notifications sent."

        success = all(results.values()) if results else True
        return success, message

    except Exception as e:
        logger.error(f"Error testing scenario {scenario}: {e}", exc_info=True)
        return False, f"Exception: {str(e)}"


def generate_summary(results: list[tuple[str, bool, str]]) -> None:
    """Generate a summary of test results."""
    lines = []

    lines.append("### Test Configuration")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| Discord | {'✅ Configured' if os.environ.get('DISCORD_WEBHOOK_URL') else '❌ Not configured'} |")

    skip_github = os.environ.get("SKIP_GITHUB_STATUS", "false").lower() == "true"
    configured_msg = "✅ Configured" if os.environ.get("GITHUB_PAT") else "❌ Not configured"
    lines.append(f"| GitHub Status | {'⏭️ Skipped' if skip_github else configured_msg} |")
    lines.append(f"| W&B Alerts | {'✅ Configured' if os.environ.get('WANDB_API_KEY') else '❌ Not configured'} |")
    lines.append("")

    lines.append("### Test Results")
    lines.append("| Scenario | Result | Message |")
    lines.append("|----------|--------|---------|")

    for scenario, success, message in results:
        result_emoji = "✅" if success else "❌"
        lines.append(f"| {scenario} | {result_emoji} | {message} |")

    lines.append("")

    # Overall result
    all_passed = all(success for _, success, _ in results)
    if all_passed:
        lines.append("### ✅ All tests passed!")
    else:
        failed_count = sum(1 for _, success, _ in results if not success)
        lines.append(f"### ❌ {failed_count} test(s) failed")

    summary_text = "\n".join(lines)

    # Write to file for GitHub summary
    with open("/tmp/notification_test_results.md", "w") as f:
        f.write(summary_text)

    # Also print to console
    print(summary_text)


def main():
    """Run notification tests."""
    requested_scenario = os.environ.get("TEST_SCENARIO", "all")

    scenarios = [
        "job_completed",
        "job_failed_137",
        "heartbeat_timeout",
        "nccl_tests_failed",
        "rapid_restarts",
        "max_runtime_reached",
    ]

    if requested_scenario != "all":
        if requested_scenario.startswith("job_failed_"):
            # Handle specific exit codes
            scenarios = [requested_scenario]
        else:
            scenarios = [s for s in scenarios if s == requested_scenario]

    logger.info(f"Running notification tests for: {scenarios}")

    results = []
    for scenario_name in scenarios:
        success, message = test_scenario(scenario_name)
        results.append((scenario_name, success, message))

        # Small delay between tests
        if len(scenarios) > 1:
            time.sleep(2)

    generate_summary(results)

    # Exit with error if any test failed
    all_passed = all(success for _, success, _ in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
