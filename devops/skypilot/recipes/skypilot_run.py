#!/usr/bin/env -S uv run python3

"""
SkyPilot run manager that handles process groups and monitoring with integrated cleanup.
"""

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from devops.skypilot.notifications import send_notifications
from devops.skypilot.utils.cost_monitor import get_cost_info
from devops.skypilot.utils.job_config import JobConfig, log_job_config
from devops.skypilot.utils.job_latency import calculate_queue_latency
from devops.skypilot.utils.nccl_tests import launch_nccl_tests
from devops.skypilot.utils.runtime_monitors import ForceRestartTestMonitor, HeartbeatMonitor, TimeoutMonitor
from devops.skypilot.utils.subprocess_helpers import terminate_process_group
from metta.common.util.log_config import getRankAwareLogger
from metta.common.wandb.context import WandbConfig, WandbContext
from metta.common.wandb.utils import log_to_wandb_summary

logger = getRankAwareLogger(__name__)

EXIT_AND_STOP = 0
EXIT_AND_RESTART = 1


def create_job_config_from_environment() -> JobConfig:
    """Create JobConfig from environment variables."""
    # Handle accumulated runtime file
    accumulated_runtime_sec = None
    accumulated_runtime_file_path = os.environ.get("ACCUMULATED_RUNTIME_FILE", "")
    if accumulated_runtime_file_path:
        accumulated_runtime_file = Path(accumulated_runtime_file_path)
        if accumulated_runtime_file.exists():
            try:
                accumulated_runtime_sec = int(accumulated_runtime_file.read_text())
            except (ValueError, IOError) as e:
                logger.warning(f"Failed to load accumulated runtime: {e}")

    node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))

    return JobConfig(
        # Node configuration
        node_index=node_index,
        total_nodes=int(os.environ.get("SKYPILOT_NUM_NODES", "1")),
        is_master=node_index == 0,
        # Job identifiers
        metta_run_id=os.environ.get("METTA_RUN_ID"),
        skypilot_task_id=os.environ.get("SKYPILOT_TASK_ID"),
        skypilot_job_id=os.environ.get("SKYPILOT_JOB_ID"),
        # Runtime configuration
        max_runtime_hours=float(os.environ.get("MAX_RUNTIME_HOURS", "0")) or None,
        heartbeat_timeout=int(os.environ.get("HEARTBEAT_TIMEOUT", "0")) or None,
        restart_count=int(os.environ.get("RESTART_COUNT", "0")),
        test_nccl=os.environ.get("TEST_NCCL", "false").lower() == "true",
        test_job_restart=os.environ.get("TEST_JOB_RESTART", "false").lower() == "true",
        start_time=int(os.environ.get("START_TIME", "0")) or None,
        # File paths
        heartbeat_file=os.environ.get("HEARTBEAT_FILE"),
        accumulated_runtime_file=accumulated_runtime_file_path or None,
        accumulated_runtime_sec=accumulated_runtime_sec,
        job_metadata_dir=os.environ.get("JOB_METADATA_DIR"),
        # Discord
        discord_webhook_url=os.environ.get("DISCORD_WEBHOOK_URL", "").strip() or None,
        enable_discord_notification=bool(os.environ.get("DISCORD_WEBHOOK_URL", "").strip()),
        # GitHub
        github_repository=os.environ.get("GITHUB_REPOSITORY"),
        metta_git_ref=os.environ.get("METTA_GIT_REF"),
        github_pat=os.environ.get("GITHUB_PAT"),
        github_status_context=os.environ.get("GITHUB_STATUS_CONTEXT", "Skypilot/E2E"),
        enable_github_status=bool(os.environ.get("GITHUB_PAT", "").strip()),
        # W&B
        wandb_project=os.environ.get("WANDB_PROJECT"),
        wandb_entity=os.environ.get("WANDB_ENTITY"),
        enable_wandb_notification=os.environ.get("ENABLE_WANDB_ALERTS", "true").lower() == "true",
    )


def run_job_in_background() -> subprocess.Popen:
    """Launch training process in the background and return the process handle."""
    cmd = ["./devops/run.sh"]

    module_path = os.environ.get("METTA_MODULE_PATH")
    if not module_path:
        raise ValueError("METTA_MODULE_PATH is required")
    cmd.append(module_path)

    # Add args if present
    args = os.environ.get("METTA_ARGS", "").strip()
    if args:
        cmd.extend(args.split())

    logger.info(f"Launching training in background: {' '.join(cmd)}")

    process = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    logger.info(f"Training process launched in background (PID: {process.pid})")
    return process


def monitor_until_termination(job_config: JobConfig, job: subprocess.Popen) -> str:
    monitors = []

    if job_config.heartbeat_timeout:
        monitors.append(
            HeartbeatMonitor(rank=job_config.node_index, heartbeat_timeout_sec=job_config.heartbeat_timeout)
        )

    if job_config.max_runtime_hours:
        monitors.append(TimeoutMonitor(rank=job_config.node_index, max_runtime_hours=job_config.max_runtime_hours))

        if job_config.test_job_restart and job_config.is_master and job_config.restart_count == 0:
            restart_time_hours = job_config.max_runtime_hours / 2.0
            monitors.append(ForceRestartTestMonitor(restart_time_hours))

    logger.info(f"Starting monitoring loop with {len(monitors)} monitor(s)")

    while True:
        exit_code = job.poll()
        if exit_code is not None:
            logger.info(f"Subprocess exited with code {exit_code}")
            if exit_code == 0:
                return "job_completed"
            else:
                return f"job_failed_{exit_code}"

        for monitor in monitors:
            reason = monitor.check_condition()
            if reason:
                logger.info(f"{monitor.name} triggered: {reason}")
                terminate_process_group(job)

                return reason

        time.sleep(10)


def main() -> int:
    termination_reason = ""
    job_config = None

    try:
        job_config = create_job_config_from_environment()
        log_job_config(job_config)

        if job_config.is_master:
            latency_sec = calculate_queue_latency()
            logger.info_master(f"SkyPilot queue latency: {latency_sec:.1f}s")

            cost_info = get_cost_info()
            total_hourly_cost = cost_info["total_hourly_cost"]
            logger.info_master(f"Total hourly cost: ${total_hourly_cost:.4f}")
            os.environ["METTA_HOURLY_COST"] = str(total_hourly_cost)

            metrics = {
                "skypilot/latency_collection_time": datetime.now(timezone.utc).isoformat(),
                "skypilot/task_id": job_config.skypilot_task_id,
                "skypilot/hourly_cost": total_hourly_cost,
                "skypilot/queue_latency_s": latency_sec,
            }

            if job_config.wandb_project and job_config.wandb_entity:
                wandb_cfg = WandbConfig(
                    enabled=True,
                    project=job_config.wandb_project,
                    entity=job_config.wandb_entity,
                    run_id=job_config.metta_run_id,
                )
                with WandbContext(wandb_cfg, extra_cfg=job_config.to_filtered_dict()):
                    log_to_wandb_summary(metrics)

        termination_reason = ""

        if job_config.test_nccl and job_config.restart_count == 0:
            if not launch_nccl_tests(logger, job_config.is_master):
                termination_reason = "nccl_tests_failed"

        # If we've restarted 3+ times and average runtime is less than 3 minutes,
        if job_config.restart_count >= 3 and job_config.accumulated_runtime_sec is not None:
            average_runtime_minutes = (job_config.accumulated_runtime_sec / job_config.restart_count) / 60
            if average_runtime_minutes < 3:
                termination_reason = "rapid_restarts"

        if not termination_reason:
            subprocess = run_job_in_background()
            termination_reason = monitor_until_termination(job_config, subprocess)

    except BaseException as e:
        exit_code = getattr(e, "code", 1) if isinstance(e, SystemExit) else 1
        if isinstance(e, SystemExit) and e.code is None:
            exit_code = 1

        termination_reason = f"job_failed_{exit_code}"
        logger.error(f"Unexpected error in main: {type(e).__name__}: {str(e)}")

    finally:
        try:
            # Only log summary if we have job_config
            if job_config:
                logger.info("========= Job Summary =========")
                logger.info(f"{'Metta Run ID:':<20} {job_config.metta_run_id or 'N/A'}")
                logger.info(f"{'Skypilot Task ID:':<20} {job_config.skypilot_task_id or 'N/A'}")
                logger.info(
                    f"{'Restart Count:':<20} {'N/A' if job_config.restart_count is None else job_config.restart_count}"
                )
                logger.info(f"{'Termination Reason:':<20} {termination_reason or 'unknown'}")
                logger.info("==============================")

                send_notifications(termination_reason, job_config)
        except Exception as e:
            # if logging or notifications throw, we still want to exit cleanly
            logger.error(f"Error in finally block: {e}")

    return EXIT_AND_RESTART if termination_reason == "force_restart_test" else EXIT_AND_STOP


if __name__ == "__main__":
    sys.exit(main())
