#!/usr/bin/env python3
"""Test script for SkyPilot notifications system."""

import os
import random
import sys
import time
from pathlib import Path

this_file = Path(__file__).resolve()
project_root = this_file.parents[2]

if not (project_root / "pyproject.toml").is_file():
    raise RuntimeError(f"Project root validation failed: {project_root}")

sys.path.insert(0, str(project_root))


from devops.skypilot.notifications.notifier import send_notifications  # noqa
from devops.skypilot.utils.job_config import JobConfig  # noqa

SCENARIOS = [
    "job_completed",
    "job_failed_137",  # OOM kill
    "heartbeat_timeout",
    "nccl_tests_failed",
    "rapid_restarts",
    "max_runtime_reached",
]


def create_test_config():
    """Create a minimal test JobConfig."""
    return JobConfig(
        # Basic identifiers
        metta_run_id=f"test_{int(time.time())}",
        skypilot_job_id=f"gh-run-{os.environ.get('GITHUB_RUN_NUMBER', '0')}",
        skypilot_task_id="test-task",
        # Node settings
        is_master=True,
        node_index=0,
        total_nodes=1,
        start_time=int(time.time()) - 3600,  # 1 hour ago
        # Discord
        discord_webhook_url=os.environ.get("DISCORD_WEBHOOK_URL"),
        enable_discord_notification=bool(os.environ.get("DISCORD_WEBHOOK_URL")),
        # GitHub
        github_repository=os.environ.get("GITHUB_REPOSITORY"),
        metta_git_ref=os.environ.get("GITHUB_SHA"),
        github_pat=os.environ.get("GITHUB_PAT"),
        enable_github_status=bool(os.environ.get("GITHUB_PAT")),
        # W&B
        wandb_entity=os.environ.get("WANDB_ENTITY", "metta-research"),
        wandb_project=os.environ.get("WANDB_PROJECT", "metta"),
        enable_wandb_notification=bool(os.environ.get("WANDB_API_KEY")),
        # Scenario-specific values (these get used by the notifier)
        heartbeat_timeout=300,
        max_runtime_hours=1.0,
        restart_count=5,
    )


def main():
    """Run notification test."""
    # Get scenario - either from env or random
    scenario = os.environ.get("TEST_SCENARIO")
    if not scenario or scenario == "random":
        scenario = random.choice(SCENARIOS)

    print(f"Testing scenario: {scenario}")
    print("=" * 50)

    # Create config and send notifications
    config = create_test_config()
    results = send_notifications(scenario, config)

    # Print results
    print("\nResults:")
    for service, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {service}: {status}")

    # Generate summary for GitHub
    summary = f"## 🔔 Notification Test: {scenario}\n\n"
    summary += "| Service | Status |\n"
    summary += "|---------|--------|\n"

    for service, success in results.items():
        status = "✅ Passed" if success else "❌ Failed"
        summary += f"| {service.title()} | {status} |\n"

    with open("/tmp/test_summary.md", "w") as f:
        f.write(summary)

    # Exit based on results
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
