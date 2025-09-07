#!/usr/bin/env python3
import re

from devops.skypilot.notifications.config import NotificationConfig
from devops.skypilot.notifications.discord import DiscordNotifier
from devops.skypilot.notifications.github import GitHubStatusUpdater
from devops.skypilot.notifications.wandb import WandbAlertNotifier
from devops.skypilot.utils.job_config import JobConfig
from metta.common.util.log_config import getRankAwareLogger

logger = getRankAwareLogger(__name__)


class NotificationManager:
    """Manages all notification types for SkyPilot jobs."""

    def __init__(self, job_config: JobConfig):
        """Initialize notification manager with job configuration."""
        self.job_config = job_config
        self.discord = DiscordNotifier() if job_config.discord_webhook_url else None
        self.github = GitHubStatusUpdater() if job_config.enable_github_status else None
        self.wandb = WandbAlertNotifier() if job_config.enable_wandb_alerts else None

    def log_config(self):
        """Log the current configuration."""
        logger.info("Run Configuration:")
        logger.info(f"  - METTA_RUN_ID: {self.job_config.metta_run_id or ''}")
        logger.info(f"  - SKYPILOT_TASK_ID: {self.job_config.skypilot_task_id or ''}")
        logger.info(f"  - NODE_INDEX: {self.job_config.node_index}")
        logger.info(f"  - IS_MASTER: {self.job_config.is_master}")
        logger.info(f"  - TOTAL_NODES: {self.job_config.total_nodes}")
        logger.info(f"  - HEARTBEAT_TIMEOUT: {self.job_config.heartbeat_timeout or 'NOT SET'}")
        logger.info(f"  - HEARTBEAT_FILE: {self.job_config.heartbeat_file or 'NOT SET'}")
        logger.info(f"  - ACCUMULATED_RUNTIME_FILE: {self.job_config.accumulated_runtime_file or 'NOT SET'}")

        if self.job_config.accumulated_runtime_sec is not None:
            logger.info(f"  - ACCUMULATED_RUNTIME_SEC: {self.job_config.accumulated_runtime_sec}")

        logger.info(f"  - MAX_RUNTIME_HOURS: {self.job_config.max_runtime_hours or 'NOT SET'}")
        logger.info(f"  - RESTART_COUNT: {self.job_config.restart_count}")
        logger.info(f"  - TEST_NCCL: {self.job_config.test_nccl}")
        logger.info(f"  - DISCORD_ENABLED: {bool(self.discord)}")
        logger.info(f"  - GITHUB_STATUS_ENABLED: {bool(self.github)}")
        logger.info(f"  - WANDB_ALERTS_ENABLED: {bool(self.wandb)}")

    def log_final_summary(self, exit_code: int, termination_reason: str):
        """Log final job summary."""
        logger.info("[SUMMARY] ===== Job Summary =====")
        logger.info(f"[SUMMARY] Metta Run ID: {self.job_config.metta_run_id or 'N/A'}")
        logger.info(f"[SUMMARY] Skypilot Task ID: {self.job_config.skypilot_task_id or 'N/A'}")
        logger.info(f"[SUMMARY] Exit code: {exit_code}")
        logger.info(f"[SUMMARY] Termination reason: {termination_reason or 'unknown'}")
        logger.info("[SUMMARY] ======================")
        logger.info(f"Job complete with exit code: {exit_code} (reason: {termination_reason or 'unknown'})")

    @staticmethod
    def get_exit_code_description(exit_code: int) -> str:
        """Get human-readable description for process exit codes."""
        exit_code_descriptions = {
            # Standard exit codes
            1: "General error",
            2: "Misuse of shell command",
            126: "Command cannot execute",
            127: "Command not found",
            # Signal-based exit codes (128 + signal number)
            129: "Process terminated (SIGHUP)",
            130: "Script terminated by Ctrl-C (SIGINT)",
            131: "Core dumped (SIGQUIT)",
            134: "Assertion failed (SIGABRT)",
            135: "Process terminated (SIGTRAP)",
            136: "Floating point exception (SIGFPE)",
            137: "Process killed (SIGKILL) - likely OOM",
            139: "Segmentation fault (SIGSEGV)",
            140: "Bad system call (SIGSYS)",
            141: "Broken pipe (SIGPIPE)",
            143: "Process terminated (SIGTERM)",
            # Python/runtime errors
            255: "Uncaught exception or runtime error",
            # Special codes
            -1: "Unknown termination",
        }

        return exit_code_descriptions.get(exit_code, f"Unknown exit code ({exit_code})")

    def send_notifications(self, termination_reason: str):
        """Send notifications based on termination reason."""
        if not self.job_config.is_master:
            logger.debug("Skipping notifications on non-master node")
            return

        logger.info(f"Processing notifications for termination reason: {termination_reason}")

        notifications = {
            "heartbeat_timeout": NotificationConfig(
                emoji="🚨",
                title="SkyPilot Job Heartbeat Timeout",
                description=f"Job failed - no heartbeat for {self.job_config.heartbeat_timeout} seconds",
            ),
            "max_runtime_reached": NotificationConfig(
                emoji="✅",
                title="SkyPilot Job Completed",
                description=f"Job ran successfully for {self.job_config.max_runtime_hours} hours",
                github_state="success",
            ),
            "nccl_tests_failed": NotificationConfig(
                emoji="🔧",
                title="SkyPilot Job NCCL Config Error",
                description="NCCL tests failed",
            ),
            "rapid_restarts": NotificationConfig(
                emoji="⚠️",
                title="SkyPilot Job Failing Repeatedly",
                description=(
                    f"Job terminated after {self.job_config.restart_count} restarts with average runtime < 3 minutes"
                ),
            ),
            "force_restart_test": NotificationConfig(
                emoji="🔧",
                title="SkyPilot Job Restarted",
                description="Job restarted to test automatic recovery.",
            ),
            "job_completed": NotificationConfig(
                emoji="✅",
                title="SkyPilot Job Completed",
                description="Job ran to completion.",
            ),
        }

        # Parse exit code from termination reason if it matches the pattern
        exit_code = None
        if termination_reason.startswith("job_failed_"):
            match = re.match(r"job_failed_(-?\d+)", termination_reason)
            if match:
                exit_code = int(match.group(1))
                logger.info(f"Parsed exit code {exit_code} from termination reason")

                notifications.update(
                    {
                        termination_reason: NotificationConfig(
                            emoji="🚨",
                            title="SkyPilot Job Failed",
                            description=(
                                f"Job failed with code {exit_code}: {self.get_exit_code_description(exit_code)}"
                            ),
                        ),
                    }
                )

        if termination_reason not in notifications:
            logger.warning(f"No notification configuration found for termination reason: {termination_reason}")
            return

        config = notifications[termination_reason]
        notifications_sent = {}

        if config.discord and self.discord:
            success = self.discord.send_notification(
                emoji=config.emoji,
                title=config.title,
                status_msg=config.description,
                job_config=self.job_config,
            )
            notifications_sent["discord"] = success

        if config.wandb and self.wandb:
            success = self.wandb.send_alert(
                state="failure" if config.github_state == "failure" else "success",
                description=config.description,
                job_config=self.job_config,
            )
            notifications_sent["wandb"] = success

        if config.github and self.github:
            success = self.github.set_status(
                state=config.github_state,
                description=config.description,
                job_config=self.job_config,
            )
            notifications_sent["github"] = success

        logger.info(
            f"Notification summary: reason={termination_reason}, "
            f"discord={notifications_sent.get('discord', False)}, "
            f"wandb={notifications_sent.get('wandb', False)}, "
            f"github={notifications_sent.get('github', False)}"
        )
