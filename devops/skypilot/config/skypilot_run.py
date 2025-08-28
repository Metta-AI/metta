#!/usr/bin/env python3
"""
SkyPilot run manager that handles process groups and monitoring with integrated cleanup.
"""

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from runtime_monitors import start_monitors
from skypilot_logging import setup_logger, log_all, log_master, log_error, log_warning

from gitta import set_skypilot_test_status
from metta.common.util.cost_monitor import get_cost_info
from metta.common.util.skypilot_latency import calculate_queue_latency
from metta.common.wandb.log_wandb import log_to_wandb
from metta.common.util.discord import send_to_discord

# Initialize logger for this module
logger = setup_logger()

# Exit code constants
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_NCCL_TEST_FAILURE = 42

# Global state - reduced to minimum necessary for signal handling
main_process: Optional[subprocess.Popen] = None
shutdown_event = threading.Event()
termination_reason_lock = threading.Lock()
_termination_reason = ""

# Configuration
node_index = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
is_master = node_index == 0
total_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", "1"))
max_runtime_hours = float(os.environ.get("MAX_RUNTIME_HOURS", "0")) or None
heartbeat_timeout = int(os.environ.get("HEARTBEAT_TIMEOUT", "0")) or None
restart_count = int(os.environ.get("RESTART_COUNT", "0"))
test_nccl = os.environ.get("TEST_NCCL", "false").lower() == "true"
enable_discord = os.environ.get("ENABLE_DISCORD", "false").lower() == "true"
enable_github_status = os.environ.get("ENABLE_GITHUB_STATUS", "false").lower() == "true"


def log_config():
    """Log the current configuration."""
    log_all("Run Configuration:")
    log_all(f"  - METTA_RUN_ID: {os.environ.get('METTA_RUN_ID', '')}")
    log_all(f"  - SKYPILOT_TASK_ID: {os.environ.get('SKYPILOT_TASK_ID', '')}")

    log_all(f"  - NODE_INDEX: {node_index}")
    log_all(f"  - IS_MASTER: {is_master}")
    log_all(f"  - TOTAL_NODES: {total_nodes}")

    log_all(f"  - HEARTBEAT_TIMEOUT: {heartbeat_timeout or 'NOT SET'}")
    heartbeat_file_path = os.environ.get("HEARTBEAT_FILE", "") or None
    log_all(f"  - HEARTBEAT_FILE: {heartbeat_file_path or 'NOT SET'}")

    accumulated_runtime_file_path = os.environ.get("ACCUMULATED_RUNTIME_FILE", "") or None
    log_all(f"  - ACCUMULATED_RUNTIME_FILE: {accumulated_runtime_file_path or 'NOT SET'}")

    if accumulated_runtime_file_path:
        accumulated_runtime_file = Path(accumulated_runtime_file_path)
        if accumulated_runtime_file.exists():
            try:
                accumulated_runtime_sec = int(accumulated_runtime_file.read_text())
                log_all(f"  - ACCUMULATED_RUNTIME_SEC: {accumulated_runtime_sec}")
            except (ValueError, IOError) as e:
                log_warning(f"Failed to load accumulated runtime: {e}")

    log_all(f"  - MAX_RUNTIME_HOURS: {max_runtime_hours or 'NOT SET'}")
    log_all(f"  - RESTART_COUNT: {restart_count}")

    log_all(f"  - TEST_NCCL: {test_nccl}")


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signal_number, frame):
        log_all(f"Received signal {signal_number}, initiating shutdown...")
        shutdown_event.set()
        shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)


def trigger_shutdown(reason: str):
    """Callback function for monitors to trigger shutdown."""
    global _termination_reason

    with termination_reason_lock:
        if not _termination_reason:  # Only set if not already set
            _termination_reason = reason
            log_all(f"Shutdown triggered with reason: {reason}")

    shutdown_event.set()


def get_termination_reason() -> str:
    """Thread-safe getter for termination reason."""
    with termination_reason_lock:
        return _termination_reason


def run_training() -> int:
    """Run the main training process and return exit code."""
    global main_process

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

    log_all(f"Running command: {' '.join(cmd)}")

    # Create new process group
    main_process = subprocess.Popen(
        cmd,
        preexec_fn=os.setsid,  # Create new process group
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    log_all(f"Started process with PID: {main_process.pid}")

    # Wait for process to complete or shutdown signal
    while main_process.poll() is None and not shutdown_event.is_set():
        time.sleep(1)

    if main_process.poll() is None:
        # Process still running, need to terminate
        log_all("Terminating training process...")
        try:
            # Try graceful shutdown first
            os.killpg(os.getpgid(main_process.pid), signal.SIGTERM)
            main_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Force kill if graceful shutdown fails
            log_warning("Graceful shutdown failed, forcing termination")
            os.killpg(os.getpgid(main_process.pid), signal.SIGKILL)
            main_process.wait()
    else:
        # Process exited on its own
        if main_process.returncode != 0:
            log_error(f"Training process failed with exit code: {main_process.returncode}")

    exit_code = main_process.returncode or 0
    log_all(f"Training process exited with code: {exit_code}")
    return exit_code


def send_discord_notification(emoji: str, title: str, status_msg: str, additional_info: str = "", exit_code: int = 0):
    """Send Discord notification directly using the discord module."""
    if not is_master or not enable_discord:
        return

    try:
        # Validate required environment variables
        required_env_vars = {
            "GITHUB_REPOSITORY": os.getenv("GITHUB_REPOSITORY"),
            "METTA_GIT_REF": os.getenv("METTA_GIT_REF"),
            "METTA_RUN_ID": os.getenv("METTA_RUN_ID"),
            "TOTAL_NODES": os.getenv("TOTAL_NODES"),
            "JOB_METADATA_DIR": os.getenv("JOB_METADATA_DIR"),
            "DISCORD_WEBHOOK_URL": os.getenv("DISCORD_WEBHOOK_URL"),
        }

        missing_vars = [k for k, v in required_env_vars.items() if not v]
        if missing_vars:
            log_warning(f"Missing required environment variables: {', '.join(missing_vars)}")
            return

        log_master(f"[RUN] Sending Discord notification: {title}")

        # Calculate runtime if START_TIME is set
        runtime_msg = ""
        start_time = os.getenv("START_TIME")
        if start_time and start_time != "0":
            try:
                current_time = int(time.time())
                duration = current_time - int(start_time)
                hours = duration // 3600
                minutes = (duration % 3600) // 60
                runtime_msg = f"**Runtime**: {hours}h {minutes}m"
            except (ValueError, TypeError):
                log_warning(f"Invalid START_TIME: {start_time}")

        # Build Discord message
        message_parts = [
            f"{emoji} **{title}**",
            "",
            f"**Repository**: {required_env_vars['GITHUB_REPOSITORY']}",
            f"**Git Ref**: {required_env_vars['METTA_GIT_REF']}",
            f"**Run ID**: {required_env_vars['METTA_RUN_ID'] or 'N/A'}",
            f"**Status**: {status_msg}",
        ]

        if runtime_msg:
            message_parts.append(runtime_msg)

        message_parts.extend([
            f"**Time**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Nodes**: {required_env_vars['TOTAL_NODES']}",
        ])

        if additional_info:
            message_parts.extend(["", additional_info])

        discord_content = "\n".join(message_parts)

        # Save to file (if still needed for debugging/logging purposes)
        assert required_env_vars['JOB_METADATA_DIR']
        assert required_env_vars['DISCORD_WEBHOOK_URL']

        discord_message_path = os.path.join(required_env_vars['JOB_METADATA_DIR'], "discord_message.txt")
        with open(discord_message_path, "w") as f:
            f.write(discord_content)

        # Send directly via Discord module
        success = send_to_discord(
            webhook_url=required_env_vars['DISCORD_WEBHOOK_URL'],
            content=discord_content,
            suppress_embeds=True
        )

        if not success:
            log_warning("[WARN] Discord notification failed; continuing")

    except Exception as e:
        log_warning(f"Failed to send Discord notification: {e}")


def set_github_status(exit_code: int, state: str, description: str):
    """Update GitHub commit status."""
    if not is_master or not enable_github_status:
        return

    if not state or not description:
        return

    # Get required environment variables
    commit_sha = os.environ.get("METTA_GIT_REF", "").strip()
    token = os.environ.get("GITHUB_PAT", "").strip()
    context = os.environ.get("GITHUB_STATUS_CONTEXT", "Skypilot/E2E").strip()

    if not all([commit_sha, token]):
        log_warning("Missing required environment variables for GitHub status")
        return

    # Get optional parameters
    job_id = os.environ.get("SKYPILOT_JOB_ID", "").strip()
    if not job_id and Path("/tmp/.sky_tmp/sky_job_id").exists():
        try:
            job_id = Path("/tmp/.sky_tmp/sky_job_id").read_text().strip()
        except Exception as e:
            log_warning(f"Could not read SkyPilot job ID: {e}")

    wandb_run_id = os.environ.get("METTA_RUN_ID")

    success = set_skypilot_test_status(
        state=state,
        description=description,
        commit_sha=commit_sha,
        token=token,
        context=context,
        exit_code=exit_code,
        job_id=job_id,
        wandb_run_id=wandb_run_id,
    )

    if not success:
        log_warning("Failed to set GitHub status")


def determine_job_status(exit_code: int, termination_reason: str) -> Tuple[str, str, int]:
    """
    Determine job status based on exit code and termination reason.

    Returns:
        Tuple of (github_state, github_description, final_exit_code)
    """
    # Default values - assume failure unless proven otherwise
    github_state = "failure"
    github_description = f"Job failed with exit code {exit_code}"
    final_exit_code = exit_code

    if termination_reason == "heartbeat_timeout":
        log_error("Job terminated due to heartbeat timeout")
        github_description = f"Job failed - no heartbeat for {heartbeat_timeout} seconds"
        final_exit_code = EXIT_SUCCESS  # Prevent SkyPilot restart

    elif termination_reason == "max_runtime_reached":
        log_all("Job terminated due to max runtime limit")
        github_state = "success"
        github_description = f"Job ran successfully for {max_runtime_hours} hours"
        final_exit_code = EXIT_SUCCESS  # Prevent SkyPilot restart

    elif termination_reason == "force_restart_test":
        log_all("Job restarting to simulate a node failure")
        github_state = "pending"
        github_description = f"Forced a restart test (restart count: {restart_count + 1})"
        final_exit_code = EXIT_FAILURE  # Cause SkyPilot restart

    elif not termination_reason and exit_code == EXIT_SUCCESS:
        log_master("[SUCCESS] Job completed successfully")
        github_state = "success"
        github_description = "Job completed successfully"

    elif exit_code == EXIT_NCCL_TEST_FAILURE:
        log_error("Job failed during NCCL tests")
        github_state = "error"  # Infrastructure issue
        github_description = "NCCL tests failed - GPU communication issue"

    else:
        # Default case - just log the error
        log_error(f"Job failed with exit code {exit_code}")

    return github_state, github_description, final_exit_code


def handle_master_cleanup(exit_code: int, termination_reason: str) -> int:
    """
    Handle master-specific cleanup tasks.

    Returns:
        Final exit code after cleanup
    """
    if not is_master:
        return exit_code

    # Determine job status
    github_state, github_description, final_exit_code = determine_job_status(
        exit_code, termination_reason
    )

    # Send notifications based on status
    if termination_reason == "heartbeat_timeout":
        send_discord_notification("❌", "SkyPilot Job Heartbeat Timeout", github_description, "", exit_code)
    elif termination_reason == "max_runtime_reached":
        send_discord_notification("✅", "SkyPilot Job Completed", github_description, "", exit_code)
    elif exit_code == EXIT_NCCL_TEST_FAILURE:
        send_discord_notification("❌", "SkyPilot Job NCCL Config Error", github_description, "", exit_code)
    elif exit_code != EXIT_SUCCESS and termination_reason != "force_restart_test":
        send_discord_notification("❌", "SkyPilot Job Failed", github_description, "", exit_code)

    # Update GitHub status
    set_github_status(exit_code, github_state, github_description)

    return final_exit_code


def print_final_summary(exit_code: int, termination_reason: str):
    """Print final job summary."""
    log_all("[SUMMARY] ===== Job Summary =====")
    log_all(f"[SUMMARY] Metta Run ID: {os.environ.get('METTA_RUN_ID', 'N/A')}")
    log_all(f"[SUMMARY] Skypilot Task ID: {os.environ.get('SKYPILOT_TASK_ID', 'N/A')}")
    log_all(f"[SUMMARY] Exit code: {exit_code}")
    log_all(f"[SUMMARY] Termination reason: {termination_reason or 'unknown'}")
    log_all("[SUMMARY] ======================")

    log_all(f"[RUN] Job complete with exit code: {exit_code} (reason: {termination_reason or 'unknown'})")


def shutdown():
    """Graceful shutdown of all processes."""
    if main_process and main_process.poll() is None:
        try:
            pgid = os.getpgid(main_process.pid)
            log_all(f"Terminating process group {pgid}")
            os.killpg(pgid, signal.SIGTERM)

            # Wait for graceful shutdown
            try:
                main_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                log_warning("Process didn't terminate gracefully, using SIGKILL")
                os.killpg(pgid, signal.SIGKILL)
                main_process.wait()
        except ProcessLookupError:
            pass  # Process already terminated


def main():
    """Main entry point that runs the full lifecycle and returns exit code."""
    # Setup environment
    log_config()
    setup_signal_handlers()

    if is_master:
        latency_sec = calculate_queue_latency()
        log_master(f"SkyPilot queue latency: {latency_sec:.1f}s")

        cost_info = get_cost_info()
        total_hourly_cost = cost_info["total_hourly_cost"]
        log_master(f"Total hourly cost: ${total_hourly_cost:.4f}")
        os.environ["METTA_HOURLY_COST"] = str(total_hourly_cost)

        log_to_wandb({"skypilot/hourly_cost": total_hourly_cost, "skypilot/queue_latency_s": latency_sec})

    # Run NCCL tests on all nodes
    if test_nccl and restart_count == 0:
        log_all(f"Running GPU diagnostics and NCCL tests...")
        try:
            result = subprocess.run(
                ["uv", "run", "python", "./devops/skypilot/config/test_nccl.py"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                log_error(f"NCCL tests failed: {result.stderr}")
                sys.exit(EXIT_NCCL_TEST_FAILURE)
            else:
                log_all("NCCL tests passed")

        except Exception as e:
            log_error(f"Failed to run NCCL tests: {e}")
            sys.exit(EXIT_NCCL_TEST_FAILURE)

    exit_code = EXIT_FAILURE
    termination_reason = ""

    try:
        start_monitors(shutdown_callback=trigger_shutdown)
        exit_code = run_training()
        termination_reason = get_termination_reason()

    except SystemExit:
        # Re-raise system exit to be handled properly
        raise
    except Exception as e:
        log_error(f"Unexpected error: {e}")
        exit_code = EXIT_FAILURE
        if not termination_reason:
            termination_reason = "unexpected_error"

    log_all(f"[INFO] Termination reason: {termination_reason}")

    # Handle cleanup and potentially modify exit code
    final_exit_code = handle_master_cleanup(exit_code, termination_reason)
    print_final_summary(exit_code, termination_reason)

    # Sleep briefly before exit
    time.sleep(1)

    if termination_reason in ["max_runtime_reached", "completed", "heartbeat_timeout"]:
        log_all("Will exit with code 0 to prevent SkyPilot restart")
        return EXIT_SUCCESS
    else:
        log_all(f"Will exit with code: {final_exit_code}")
        return final_exit_code


if __name__ == "__main__":
    sys.exit(main())
