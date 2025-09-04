#!/usr/bin/env python3
"""
SkyPilot run manager that handles process groups and monitoring with integrated cleanup.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from devops.skypilot.utils.cost_monitor import get_cost_info
from devops.skypilot.utils.job_latency import calculate_queue_latency
from devops.skypilot.utils.nccl_tests import launch_nccl_tests
from devops.skypilot.utils.notifications import (
    log_config,
    log_final_summary,
    send_notifications,
)
from devops.skypilot.utils.runtime_monitors import ForceRestartTestMonitor, HeartbeatMonitor, TimeoutMonitor
from metta.common.util.log_config import getRankAwareLogger
from metta.common.wandb.utils import ensure_wandb_run, log_to_wandb

logger = getRankAwareLogger(__name__)

EXIT_AND_STOP = 0

# Configuration
node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
is_master = node_index == 0
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

    subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    logger.info("Training process launched in background")


def monitor_until_termination() -> str:
    """
    Monitor until a termination condition is met.

    Returns:
        termination_reason: Why monitoring stopped
    """
    monitors = []

    if heartbeat_timeout:
        monitors.append(HeartbeatMonitor(rank=node_index, heartbeat_timeout_sec=heartbeat_timeout))

    if max_runtime_hours:
        monitors.append(TimeoutMonitor(rank=node_index, max_runtime_hours=max_runtime_hours))

        if is_master and restart_count == 0:
            monitors.append(ForceRestartTestMonitor(rank=node_index, restart_time_hours=max_runtime_hours / 2.0))

    logger.info(f"Starting monitoring loop with {len(monitors)} monitor(s)")

    # Main monitoring loop
    while True:
        for monitor in monitors:
            should_terminate, reason = monitor.check_condition()
            if should_terminate:
                logger.info(f"{monitor.name} triggered: {reason}")
                return reason
        time.sleep(10)

    # elif termination_reason == "force_restart_test":
    #     logger.info("Job restarting to simulate a node failure")
    #     state = "pending"
    #     description = f"Forced a restart test (restart count: {restart_count + 1})"
    #     final_exit_code = EXIT_FAILURE  # Cause SkyPilot restart


def main() -> int:
    log_config()

    if is_master:
        latency_sec = calculate_queue_latency()
        logger.info_master(f"SkyPilot queue latency: {latency_sec:.1f}s")

        cost_info = get_cost_info()
        total_hourly_cost = cost_info["total_hourly_cost"]
        logger.info_master(f"Total hourly cost: ${total_hourly_cost:.4f}")
        os.environ["METTA_HOURLY_COST"] = str(total_hourly_cost)  # used in system monitor

        metrics = {
            "skypilot/latency_collection_time": datetime.now(timezone.utc).isoformat(),
            "skypilot/task_id": os.environ.get("SKYPILOT_TASK_ID", "unknown"),
            "skypilot/hourly_cost": total_hourly_cost,
            "skypilot/queue_latency_s": latency_sec,
        }

        ensure_wandb_run()
        log_to_wandb(metrics)

    termination_reason = ""

    if test_nccl and restart_count == 0:
        if not launch_nccl_tests(logger, is_master):
            termination_reason = "nccl_tests_failed"

    if not termination_reason:
        run_training_in_background()
        termination_reason = monitor_until_termination()

    log_final_summary(0, termination_reason)
    send_notifications(termination_reason)
    return EXIT_AND_STOP


if __name__ == "__main__":
    sys.exit(main())
