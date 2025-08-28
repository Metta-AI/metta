#!/usr/bin/env python3
"""
SkyPilot run manager that handles process groups and monitoring with integrated cleanup.
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from metta.common.util.cost_monitor import get_cost_info
from metta.common.util.github import set_skypilot_test_status
from metta.common.util.skypilot_latency import calculate_queue_latency
from metta.common.wandb.log_wandb import log_to_wandb

from .runtime_monitors import start_monitors

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Exit code constants
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_NCCL_TEST_FAILURE = 42

# Global state
main_process: Optional[subprocess.Popen] = None
shutdown_event = threading.Event()
termination_reason_lock = threading.Lock()
exit_code = 0
termination_reason = ""
github_status_state = ""
github_status_description = ""

# Configuration
rank = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
is_master = rank == 0
total_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", "1"))
max_runtime_hours = float(os.environ.get("MAX_RUNTIME_HOURS", "0")) or None
heartbeat_timeout = int(os.environ.get("HEARTBEAT_TIMEOUT", "0")) or None
restart_count = int(os.environ.get("RESTART_COUNT", "0"))
test_nccl = os.environ.get("TEST_NCCL", "false").lower() == "true"
enable_discord = os.environ.get("ENABLE_DISCORD", "false").lower() == "true"
enable_github_status = os.environ.get("ENABLE_GITHUB_STATUS", "false").lower() == "true"

# Paths
job_metadata_dir = Path(os.environ.get("JOB_METADATA_DIR", "/tmp/metta"))
heartbeat_file = job_metadata_dir / "heartbeat"


def log_config():
    """Log the current configuration."""
    logger.info("Run Configuration:")
    logger.info(f"  - NODE_RANK: {rank}")
    logger.info(f"  - IS_MASTER: {is_master}")
    logger.info(f"  - TOTAL_NODES: {total_nodes}")
    logger.info(f"  - METTA_RUN_ID: {os.environ.get('METTA_RUN_ID', '')}")
    logger.info(f"  - SKYPILOT_TASK_ID: {os.environ.get('SKYPILOT_TASK_ID', '')}")
    logger.info(f"  - HEARTBEAT_TIMEOUT: {heartbeat_timeout or 'NOT SET'}")
    logger.info(f"  - MAX_RUNTIME_HOURS: {max_runtime_hours or 'NOT SET'}")
    logger.info(f"  - RESTART_COUNT: {restart_count}")
    logger.info(f"  - TEST_NCCL: {test_nccl}")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        shutdown_event.set()
        shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)


def trigger_shutdown(reason: str):
    """Callback function for monitors to trigger shutdown."""
    global termination_reason

    with termination_reason_lock:
        if not termination_reason:  # Only set if not already set
            termination_reason = reason
            logger.info(f"Shutdown triggered with reason: {reason}")

    shutdown_event.set()


def run_training():
    """Run the main training process."""
    global main_process, exit_code

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

    logger.info(f"Running command: {' '.join(cmd)}")

    # Create new process group
    main_process = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,  # Create new process group
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    logger.info(f"Started process with PID: {main_process.pid}")

    # Wait for process to complete or shutdown signal
    while main_process.poll() is None and not shutdown_event.is_set():
        time.sleep(1)

    if main_process.poll() is None:
        # Process still running, need to terminate
        logger.info("Terminating training process...")
        try:
            # Try graceful shutdown first
            os.killpg(os.getpgid(main_process.pid), signal.SIGTERM)
            main_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown fails
            logger.warning("Graceful shutdown failed, forcing termination")
            os.killpg(os.getpgid(main_process.pid), signal.SIGKILL)
            main_process.wait()
    else:
        # Process exited on its own
        if main_process.returncode != 0:
            logger.error(f"Training process failed with exit code: {main_process.returncode}")

    exit_code = main_process.returncode or 0
    logger.info(f"Training process exited with code: {exit_code}")


def send_discord_notification(emoji: str, title: str, status_msg: str, additional_info: str = ""):
    """Send Discord notification via the shell script."""
    if not is_master or not enable_discord:
        return

    try:
        # Set required environment variables for the script
        env = os.environ.copy()
        env.update(
            {
                "IS_MASTER": "true",
                "ENABLE_DISCORD": "true",
                "CMD_EXIT": str(exit_code),
            }
        )

        subprocess.run(
            [
                "bash",
                "./devops/skypilot/config/observability/send_discord_notification.sh",
                emoji,
                title,
                status_msg,
                additional_info,
            ],
            env=env,
            check=False,
        )
    except Exception as e:
        logger.warning(f"Failed to send Discord notification: {e}")


def set_github_status():
    """Update GitHub commit status using the new helper function."""
    if not is_master or not enable_github_status:
        return

    if not github_status_state or not github_status_description:
        return

    # Get required environment variables
    commit_sha = os.environ.get("METTA_GIT_REF", "").strip()
    token = os.environ.get("GITHUB_PAT", "").strip()
    context = os.environ.get("GITHUB_STATUS_CONTEXT", "Skypilot/E2E").strip()

    if not all([commit_sha, token]):
        logger.warning("Missing required environment variables for GitHub status")
        return

    # Get optional parameters
    job_id = os.environ.get("SKYPILOT_JOB_ID", "").strip()
    if not job_id and Path("/tmp/.sky_tmp/sky_job_id").exists():
        try:
            job_id = Path("/tmp/.sky_tmp/sky_job_id").read_text().strip()
        except Exception as e:
            logger.warning(f"Could not read SkyPilot job ID: {e}")

    wandb_run_id = os.environ.get("METTA_RUN_ID")

    success = set_skypilot_test_status(
        state=github_status_state,
        description=github_status_description,
        commit_sha=commit_sha,
        token=token,
        context=context,
        exit_code=exit_code,
        job_id=job_id,
        wandb_run_id=wandb_run_id,
    )

    if not success:
        logger.warning("Failed to set GitHub status")


def handle_master_cleanup():
    """Handle master-specific cleanup tasks."""
    global exit_code, github_status_state, github_status_description, termination_reason

    if not is_master:
        return

    # Check termination reason and set appropriate status
    if termination_reason == "heartbeat_timeout":
        logger.error("Job terminated due to heartbeat timeout")
        github_status_state = "failure"
        github_status_description = f"Job failed - no heartbeat for {heartbeat_timeout} seconds"
        send_discord_notification("❌", "SkyPilot Job Heartbeat Timeout", github_status_description)
        # Map to success exit code to prevent a SkyPilot restart
        exit_code = EXIT_SUCCESS

    elif termination_reason == "max_runtime_reached":
        logger.info("Job terminated due to max runtime limit")
        github_status_state = "success"
        github_status_description = f"Job ran successfully for {max_runtime_hours} hours"
        send_discord_notification("✅", "SkyPilot Job Completed", github_status_description)
        # Map to success exit code to prevent a SkyPilot restart
        exit_code = EXIT_SUCCESS

    elif not termination_reason:
        if exit_code == EXIT_SUCCESS:
            logger.info("[SUCCESS] Job completed successfully")
            termination_reason = "completed"
            github_status_state = "success"
            github_status_description = "Job completed successfully"
        else:
            logger.error(f"Job failed with exit code {exit_code}")
            termination_reason = f"exit_code_{exit_code}"
            github_status_state = "failure"
            github_status_description = f"Job failed with exit code {exit_code}"

    elif exit_code == EXIT_NCCL_TEST_FAILURE:
        logger.error("Job failed during NCCL tests")
        github_status_state = "error"  # Infrastructure issue
        github_status_description = "NCCL tests failed - GPU communication issue"
        termination_reason = "nccl_test_failure"
        send_discord_notification("❌", "SkyPilot Job NCCL Config Error", github_status_description)

    else:
        logger.error(f"Job failed with exit code {exit_code}")
        github_status_state = "failure"
        github_status_description = f"Job failed with exit code {exit_code}"
        termination_reason = f"exit_code_{exit_code}"
        send_discord_notification("❌", "SkyPilot Job Failed", github_status_description)

    # Update GitHub status
    set_github_status()


def print_final_summary():
    """Print final job summary."""
    logger.info("[SUMMARY] ===== Job Summary =====")
    logger.info(f"[SUMMARY] Metta Run ID: {os.environ.get('METTA_RUN_ID', 'N/A')}")
    logger.info(f"[SUMMARY] Skypilot Task ID: {os.environ.get('SKYPILOT_TASK_ID', 'N/A')}")
    logger.info(f"[SUMMARY] Exit code: {exit_code}")
    logger.info(f"[SUMMARY] Termination reason: {termination_reason or 'unknown'}")
    logger.info("[SUMMARY] ======================")

    logger.info(f"[RUN] Job complete with exit code: {exit_code} (reason: {termination_reason or 'unknown'})")


def shutdown():
    """Graceful shutdown of all processes."""
    if main_process and main_process.poll() is None:
        try:
            pgid = os.getpgid(main_process.pid)
            logger.info(f"Terminating process group {pgid}")
            os.killpg(pgid, signal.SIGTERM)

            # Wait for graceful shutdown
            try:
                main_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Process didn't terminate gracefully, using SIGKILL")
                os.killpg(pgid, signal.SIGKILL)
                main_process.wait()
        except ProcessLookupError:
            pass  # Process already terminated


def main():
    """Main entry point that runs the full lifecycle and returns exit code."""
    global exit_code, termination_reason

    # Setup environment
    log_config()
    setup_signal_handlers()

    if is_master:
        latency_sec = calculate_queue_latency()
        logger.info(f"SkyPilot queue latency: {latency_sec:.1f}s")

        cost_info = get_cost_info()
        total_hourly_cost = cost_info["total_hourly_cost"]
        logger.info(f"Total hourly cost: ${total_hourly_cost:.4f}")
        os.environ["METTA_HOURLY_COST"] = str(total_hourly_cost)

        log_to_wandb({"skypilot/hourly_cost": total_hourly_cost, "skypilot/queue_latency_s": latency_sec})

    # Run NCCL tests on all nodes
    if test_nccl and restart_count == 0:
        logger.info(f"Running GPU diagnostics and NCCL tests (node {rank})...")
        try:
            result = subprocess.run(
                ["uv", "run", "python", "./devops/skypilot/config/preflight/test_nccl.py"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"NCCL tests failed: {result.stderr}")
                sys.exit(EXIT_NCCL_TEST_FAILURE)
            else:
                logger.info("NCCL tests passed")

        except Exception as e:
            logger.error(f"Failed to run NCCL tests: {e}")
            sys.exit(EXIT_NCCL_TEST_FAILURE)

    try:
        start_monitors(shutdown_callback=trigger_shutdown)
        run_training()

    except SystemExit:
        # Re-raise system exit to be handled properly
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        exit_code = EXIT_FAILURE
        if not termination_reason:
            termination_reason = "unexpected_error"

    logger.info(f"[INFO] Termination reason: {termination_reason}")

    handle_master_cleanup()
    print_final_summary()

    # Sleep briefly before exit
    time.sleep(1)

    if termination_reason in ["max_runtime_reached", "completed", "heartbeat_timeout"]:
        logger.info("Will exit with code 0 to prevent SkyPilot restart")
        return EXIT_SUCCESS

    else:
        logger.info(f"Will exit with code: {exit_code}")
        return exit_code


if __name__ == "__main__":
    sys.exit(main())
