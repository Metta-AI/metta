#!/usr/bin/env python3
"""
SkyPilot run manager that handles process groups and monitoring with integrated cleanup.
"""

import os
import subprocess
import sys
import time
from typing import Tuple

from devops.skypilot.utils.cost_monitor import get_cost_info
from devops.skypilot.utils.job_latency import calculate_queue_latency
from devops.skypilot.utils.nccl_tests import launch_nccl_tests
from devops.skypilot.utils.notifications import (
    log_config,
    log_final_summary,
    send_discord_notification,
    send_wandb_alert_notification,
    set_github_status,
)
from devops.skypilot.utils.runtime_monitors import HeartbeatMonitor, TimeoutMonitor
from metta.common.util.log_config import getRankAwareLogger
from metta.common.wandb.utils import log_to_wandb

logger = getRankAwareLogger(__name__)

# Exit code constants
EXIT_AND_STOP = 0
EXIT_AND_RESTART = 1

# Configuration
node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
is_master = node_index == 0
total_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", "1"))
max_runtime_hours = float(os.environ.get("MAX_RUNTIME_HOURS", "0")) or None
heartbeat_timeout = int(os.environ.get("HEARTBEAT_TIMEOUT", "0")) or None
restart_count = int(os.environ.get("RESTART_COUNT", "0"))
test_nccl = os.environ.get("TEST_NCCL", "false").lower() == "true"


def run_training_in_background():
    """Launch training process in the background and return immediately."""
    cmd = ["./devops/run.sh"]

    module_path = os.environ.get("METTA_MODULE_PATH")
    if not module_path:
        raise ValueError("METTA_MODULE_PATH is required")
    cmd.append(module_path)

    # Add args if present
    args = os.environ.get("METTA_ARGS", "").strip()
    if args:
        cmd.extend(["--args"] + args.split())

    # Add overrides if present
    overrides = os.environ.get("METTA_OVERRIDES", "").strip()
    if overrides:
        cmd.extend(["--overrides"] + overrides.split())

    logger.info(f"Launching training in background: {' '.join(cmd)}")

    # Launch and forget - let it run independently
    subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    logger.info("Training process launched in background")


def start_monitoring_loop() -> Tuple[int, str]:
    """
    Run monitoring loop until a termination condition is met.

    Returns:
        Tuple of (exit_code, termination_reason)
    """
    monitors = []

    # Initialize heartbeat monitor if configured
    if heartbeat_timeout:
        monitors.append(
            HeartbeatMonitor(
                rank=node_index,
                heartbeat_timeout_sec=heartbeat_timeout,
            )
        )

    # Initialize timeout monitor if configured
    if max_runtime_hours:
        monitors.append(TimeoutMonitor(rank=node_index, max_runtime_hours=max_runtime_hours))

    if not monitors:
        logger.info("No monitors configured, running indefinitely...")
        # Just sleep forever if no monitors are configured
        # In practice this shouldn't happen as we always have some monitoring
        while True:
            time.sleep(60)

    logger.info(f"Starting monitoring loop with {len(monitors)} monitor(s)")

    # Main monitoring loop
    while True:
        # Check all monitors
        for monitor in monitors:
            should_terminate, reason = monitor.check_condition()
            if should_terminate:
                logger.info(f"{monitor.name} triggered: {reason}")
                sys.exit(handle_master_cleanup(EXIT_AND_STOP, reason))

        # Sleep before next check
        time.sleep(10)


def determine_job_status(exit_code: int, termination_reason: str) -> Tuple[str, str, int]:
    """
    Determine job status based on exit code and termination reason.

    Returns:
        Tuple of (state, description, final_exit_code)
        - state: "success", "failure", "error", or "pending"
        - description: Human-readable description of what happened
        - final_exit_code: Exit code to use (may differ from input exit_code)
    """
    # Default values - assume failure unless proven otherwise
    state = "failure"
    description = f"Job failed with exit code {exit_code}"
    final_exit_code = exit_code

    if termination_reason == "heartbeat_timeout":
        logger.error("Job terminated due to heartbeat timeout")
        description = f"Job failed - no heartbeat for {heartbeat_timeout} seconds"
        final_exit_code = EXIT_AND_STOP  # Prevent SkyPilot restart

    elif termination_reason == "max_runtime_reached":
        logger.info("Job terminated due to max runtime limit")
        state = "success"
        description = f"Job ran successfully for {max_runtime_hours} hours"
        final_exit_code = EXIT_AND_STOP  # Prevent SkyPilot restart

    elif termination_reason == "nccl_tests_failed":
        logger.error("Job failed during NCCL tests")
        state = "error"  # Infrastructure issue
        description = "NCCL tests failed"

    elif not termination_reason and exit_code == EXIT_AND_STOP:
        logger.info("Job completed successfully")
        state = "success"
        description = "Job completed successfully"

    else:
        # Default case - just log the error
        logger.error(f"Job failed with exit code {exit_code}")

    return state, description, final_exit_code


def handle_master_cleanup(exit_code: int, termination_reason: str) -> int:
    """
    Handle master-specific cleanup tasks.

    Returns:
        Final exit code after cleanup
    """
    if not is_master:
        return exit_code

    log_final_summary(exit_code, termination_reason)

    # Determine job status (generic)
    state, description, final_exit_code = determine_job_status(exit_code, termination_reason)

    # Send notifications based on status
    if termination_reason == "heartbeat_timeout":
        send_discord_notification("🚨", "SkyPilot Job Heartbeat Timeout", description, "")
        send_wandb_alert_notification(state, description)
    elif termination_reason == "max_runtime_reached":
        send_discord_notification("✅", "SkyPilot Job Completed", description, "")
        send_wandb_alert_notification(state, description)
    elif termination_reason == "nccl_tests_failed":
        send_discord_notification("🔧", "SkyPilot Job NCCL Config Error", description, "")
        send_wandb_alert_notification(state, description)
    elif state == "success":
        # Only W&B gets success notifications
        send_wandb_alert_notification(state, description)

    # Update GitHub status
    set_github_status(exit_code, state, description)

    return final_exit_code


def main():
    log_config()

    if is_master:
        latency_sec = calculate_queue_latency()
        logger.info_master(f"SkyPilot queue latency: {latency_sec:.1f}s")

        cost_info = get_cost_info()
        total_hourly_cost = cost_info["total_hourly_cost"]
        logger.info_master(f"Total hourly cost: ${total_hourly_cost:.4f}")
        os.environ["METTA_HOURLY_COST"] = str(total_hourly_cost)

        log_to_wandb({"skypilot/hourly_cost": total_hourly_cost, "skypilot/queue_latency_s": latency_sec})

    if test_nccl and restart_count == 0:
        test_passed = launch_nccl_tests(logger, is_master)
        if not test_passed:
            sys.exit(handle_master_cleanup(EXIT_AND_STOP, "nccl_tests_failed"))

    run_training_in_background()

    start_monitoring_loop()


if __name__ == "__main__":
    sys.exit(main())
