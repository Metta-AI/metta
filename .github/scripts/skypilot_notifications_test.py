#!/usr/bin/env python3
"""Test script for SkyPilot notifications system."""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

this_file = Path(__file__).resolve()
project_root = this_file.parents[2]  # 3 levels up

if not (project_root / "pyproject.toml").is_file():
    raise RuntimeError(f"Project root validation failed: {project_root}")

sys.path.insert(0, str(project_root))

from devops.skypilot.notifications.discord import DiscordNotifier  # noqa
from devops.skypilot.notifications.github import GitHubStatusUpdater  # noqa
from devops.skypilot.notifications.manager import NotificationManager  # noqa
from devops.skypilot.notifications.wandb import WandbAlertNotifier  # noqa
from devops.skypilot.utils.job_config import JobConfig  # noqa

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NotificationTester:
    """Tests the notification system with various scenarios."""

    def __init__(self):
        self.results = []
        self.custom_message = os.environ.get("CUSTOM_MESSAGE", "")
        self.skip_github = os.environ.get("SKIP_GITHUB_STATUS", "false").lower() == "true"

    def create_test_job_config(self, scenario: str) -> JobConfig:
        """Create a test JobConfig for the given scenario."""
        # Base configuration from environment
        config = JobConfig()

        # Set webhook URLs and tokens from environment
        config.discord_webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
        config.github_pat = os.environ.get("GITHUB_PAT")
        config.enable_github_status = not self.skip_github
        config.enable_wandb_alerts = bool(os.environ.get("WANDB_API_KEY"))

        # W&B configuration
        config.wandb_entity = os.environ.get("WANDB_ENTITY", "metta-research")
        config.wandb_project = os.environ.get("WANDB_PROJECT", "metta")

        # GitHub configuration
        config.github_repository = os.environ.get("GITHUB_REPOSITORY", "unknown/repo")
        config.metta_git_ref = os.environ.get("GITHUB_SHA", "unknown")
        config.github_status_context = f"test/notifications/{scenario}"

        # Job metadata
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        config.metta_run_id = f"test_notify_{scenario}_{timestamp}"
        config.skypilot_job_id = f"test-{os.environ.get('GITHUB_RUN_NUMBER', '0')}"
        config.skypilot_task_id = f"test-task-{scenario}"

        # Node configuration
        config.is_master = True
        config.node_index = 0
        config.total_nodes = 1

        # Timing configuration
        config.start_time = int(time.time()) - 3600  # Started 1 hour ago

        # Scenario-specific settings
        if scenario == "heartbeat_timeout":
            config.heartbeat_timeout = 300  # 5 minutes
        elif scenario == "max_runtime_reached":
            config.max_runtime_hours = 1.0
        elif scenario == "rapid_restarts":
            config.restart_count = 5
            config.accumulated_runtime_sec = 600  # 10 minutes total over 5 restarts
        elif scenario == "job_failed":
            config.restart_count = 0

        # Add test metadata directory
        config.job_metadata_dir = f"/tmp/test_notifications_{''.join(c for c in scenario if c.isalnum() or c in '_-')}"
        os.makedirs(config.job_metadata_dir, exist_ok=True)

        return config

    def test_scenario(self, scenario: str) -> Tuple[bool, str]:
        """Test a specific notification scenario."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing scenario: {scenario}")
        logger.info(f"{'=' * 60}")

        try:
            # Create test job config
            job_config = self.create_test_job_config(scenario)

            logger.info(f"Using GITHUB_SHA from origin/main: {job_config.metta_git_ref}")

            # Create notification manager
            manager = NotificationManager(job_config)

            # Log configuration
            manager.log_config()

            # Map scenarios to termination reasons
            termination_reasons = {
                "job_completed": "job_completed",
                "job_failed": "job_failed_137",  # OOM kill
                "heartbeat_timeout": "heartbeat_timeout",
                "nccl_tests_failed": "nccl_tests_failed",
                "rapid_restarts": "rapid_restarts",
                "max_runtime_reached": "max_runtime_reached",
            }

            termination_reason = termination_reasons.get(scenario, "job_completed")

            # Add custom message if provided
            if self.custom_message:
                logger.info(f"Adding custom message: {self.custom_message}")

            # Send notifications
            manager.send_notifications(termination_reason)

            # Test individual components directly for detailed results
            results = self._test_individual_components(job_config, scenario)

            success = all(r[1] for r in results if r[0] != "github" or not self.skip_github)
            message = f"Scenario '{scenario}' completed. Results: " + ", ".join(
                f"{comp}: {'✅' if ok else '❌'}" for comp, ok in results
            )

            return success, message

        except Exception as e:
            logger.error(f"Error testing scenario {scenario}: {e}", exc_info=True)
            return False, f"Exception: {str(e)}"

    def _test_individual_components(self, job_config: JobConfig, scenario: str) -> List[Tuple[str, bool]]:
        """Test each notification component individually."""
        results = []

        # Test Discord
        if job_config.discord_webhook_url:
            try:
                discord = DiscordNotifier()
                emoji = "🧪"
                title = f"Test Notification - {scenario}"
                status = (
                    f"Testing {scenario} scenario from GitHub Actions run #{os.environ.get('GITHUB_RUN_NUMBER', 'N/A')}"
                )

                success = discord.send_notification(
                    emoji=emoji,
                    title=title,
                    status_msg=status,
                    job_config=job_config,
                    additional_info=f"GitHub Run: #{os.environ.get('GITHUB_RUN_NUMBER', 'N/A')}",
                )
                results.append(("discord", success))
            except Exception as e:
                logger.error(f"Discord test failed: {e}")
                results.append(("discord", False))
        else:
            logger.warning("Discord webhook not configured")
            results.append(("discord", False))

        # Test GitHub
        if job_config.enable_github_status and job_config.github_pat:
            try:
                github = GitHubStatusUpdater()
                state = "failure" if "failed" in scenario else "success"
                description = f"Test: {scenario}"

                success = github.set_status(state=state, description=description, job_config=job_config)
                results.append(("github", success))
            except Exception as e:
                logger.error(f"GitHub status test failed: {e}")
                results.append(("github", False))
        else:
            logger.warning("GitHub status updates disabled or not configured")
            results.append(("github", False))

        # Test W&B
        if job_config.enable_wandb_alerts:
            try:
                wandb = WandbAlertNotifier()
                state = "failure" if "failed" in scenario or "timeout" in scenario else "success"
                description = f"Test notification for {scenario}"

                success = wandb.send_alert(state=state, description=description, job_config=job_config)
                results.append(("wandb", success))
            except Exception as e:
                logger.error(f"W&B alert test failed: {e}")
                results.append(("wandb", False))
        else:
            logger.warning("W&B alerts not configured")
            results.append(("wandb", False))

        return results

    def run_tests(self):
        """Run all requested tests."""
        test_scenario = os.environ.get("TEST_SCENARIO", "all")

        scenarios = [
            "job_completed",
            "job_failed",
            "heartbeat_timeout",
            "nccl_tests_failed",
            "rapid_restarts",
            "max_runtime_reached",
        ]

        if test_scenario != "all":
            scenarios = [test_scenario]

        logger.info(f"Running notification tests for: {scenarios}")
        logger.info(f"Skip GitHub status: {self.skip_github}")

        for scenario in scenarios:
            success, message = self.test_scenario(scenario)
            self.results.append((scenario, success, message))

            # Small delay between tests
            if len(scenarios) > 1:
                time.sleep(2)

        self.generate_summary()

        # Exit with error if any test failed
        all_passed = all(success for _, success, _ in self.results)
        sys.exit(0 if all_passed else 1)

    def generate_summary(self):
        """Generate a summary of test results."""
        summary_lines = []

        summary_lines.append("### Test Configuration")
        summary_lines.append("| Setting | Value |")
        summary_lines.append("|---------|-------|")
        summary_lines.append(
            f"| Discord | {'✅ Configured' if os.environ.get('DISCORD_WEBHOOK_URL') else '❌ Not configured'} |"
        )

        configured_msg = "✅ Configured" if os.environ.get("GITHUB_PAT") else "❌ Not configured"
        summary_lines.append(f"| GitHub Status | {'⏭️ Skipped' if self.skip_github else configured_msg} |")
        summary_lines.append(
            f"| W&B Alerts | {'✅ Configured' if os.environ.get('WANDB_API_KEY') else '❌ Not configured'} |"
        )
        summary_lines.append("")

        summary_lines.append("### Test Results")
        summary_lines.append("| Scenario | Result | Message |")
        summary_lines.append("|----------|--------|---------|")

        for scenario, success, message in self.results:
            result_emoji = "✅" if success else "❌"
            summary_lines.append(f"| {scenario} | {result_emoji} | {message} |")

        summary_lines.append("")

        # Overall result
        all_passed = all(success for _, success, _ in self.results)
        if all_passed:
            summary_lines.append("### ✅ All tests passed!")
        else:
            failed_count = sum(1 for _, success, _ in self.results if not success)
            summary_lines.append(f"### ❌ {failed_count} test(s) failed")

        # Write to file for GitHub summary
        with open("/tmp/notification_test_results.md", "w") as f:
            f.write("\n".join(summary_lines))

        # Also print to console
        print("\n".join(summary_lines))


if __name__ == "__main__":
    tester = NotificationTester()
    tester.run_tests()
