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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Exit code constants
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_NCCL_TEST_FAILURE = 42


class SkypilotRunManager:
    """Manages a process group containing torchrun and monitors."""

    def __init__(self):
        self.rank = int(os.environ.get("SKYPILOT_NODE_RANK", "0"))
        self.is_master = self.rank == 0
        self.total_nodes = int(os.environ.get("SKYPILOT_NUM_NODES", "1"))

        # Runtime configuration
        self.max_runtime_hours = float(os.environ.get("MAX_RUNTIME_HOURS", "0")) or None
        self.heartbeat_timeout = int(os.environ.get("HEARTBEAT_TIMEOUT", "0")) or None
        self.restart_count = int(os.environ.get("RESTART_COUNT", "0"))
        self.test_nccl = os.environ.get("TEST_NCCL", "false").lower() == "true"
        self.enable_discord = os.environ.get("ENABLE_DISCORD", "false").lower() == "true"
        self.enable_github_status = os.environ.get("ENABLE_GITHUB_STATUS", "false").lower() == "true"

        # Paths
        self.job_metadata_dir = Path(os.environ.get("JOB_METADATA_DIR", "/tmp/metta"))
        self.termination_reason_file = self.job_metadata_dir / "termination_reason"
        self.heartbeat_file = self.job_metadata_dir / "heartbeat"
        self.accumulated_runtime_file = self.job_metadata_dir / "accumulated_runtime"

        # Process tracking
        self.main_process: Optional[subprocess.Popen] = None
        self.monitor_threads: list[threading.Thread] = []
        self.shutdown_event = threading.Event()
        self.start_time = None
        self.exit_code = 0
        self.termination_reason = ""

        # GitHub status environment variables
        self.github_status_state = ""
        self.github_status_description = ""

        # Ensure metadata directory exists
        self.job_metadata_dir.mkdir(parents=True, exist_ok=True)

    def log_config(self):
        """Log the current configuration."""
        logger.info("Run Configuration:")
        logger.info(f"  - NODE_RANK: {self.rank}")
        logger.info(f"  - IS_MASTER: {self.is_master}")
        logger.info(f"  - TOTAL_NODES: {self.total_nodes}")
        logger.info(f"  - METTA_RUN_ID: {os.environ.get('METTA_RUN_ID', '')}")
        logger.info(f"  - SKYPILOT_TASK_ID: {os.environ.get('SKYPILOT_TASK_ID', '')}")
        logger.info(f"  - HEARTBEAT_TIMEOUT: {self.heartbeat_timeout or 'NOT SET'}")
        logger.info(f"  - MAX_RUNTIME_HOURS: {self.max_runtime_hours or 'NOT SET'}")
        logger.info(f"  - RESTART_COUNT: {self.restart_count}")
        logger.info(f"  - TEST_NCCL: {self.test_nccl}")

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGHUP, signal_handler)

    def setup_environment(self):
        """Setup environment including sourcing the env file."""
        # Run configure_environment.sh
        subprocess.run(["bash", "./devops/skypilot/config/lifecycle/configure_environment.sh"], check=True)

        # Source environment file
        env_file = self._get_env_file_path()
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        # Handle shell variable expansion if needed
                        os.environ[key] = value

    def run_preflight_checks(self):
        """Run preflight checks including NCCL tests if needed."""
        if self.is_master:
            # Collect SkyPilot latency
            try:
                subprocess.run(["uv", "run", "python", "common/src/metta/common/util/skypilot_latency.py"], check=False)
            except Exception as e:
                logger.warning(f"Failed to collect SkyPilot latency: {e}")

            # Collect instance cost
            try:
                subprocess.run(["uv", "run", "python", "common/src/metta/common/util/cost_monitor.py"], check=False)
            except Exception as e:
                logger.warning(f"Failed to collect instance cost: {e}")

        # Run NCCL tests if needed
        if self.test_nccl and self.restart_count == 0:
            logger.info(f"Running GPU diagnostics and NCCL tests (node {self.rank})...")
            try:
                result = subprocess.run(
                    ["uv", "run", "python", "./devops/skypilot/config/preflight/test_nccl.py"],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    logger.error(f"NCCL tests failed: {result.stderr}")
                    self.termination_reason_file.write_text("nccl_test_failure")
                    sys.exit(EXIT_NCCL_TEST_FAILURE)
                else:
                    logger.info("NCCL tests passed")

            except Exception as e:
                logger.error(f"Failed to run NCCL tests: {e}")
                self.termination_reason_file.write_text("nccl_test_failure")
                sys.exit(EXIT_NCCL_TEST_FAILURE)

    def build_command(self) -> list[str]:
        """Build the training command."""
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

        return cmd

    def heartbeat_monitor(self):
        """Monitor heartbeat file and terminate if timeout exceeded."""
        if not self.heartbeat_timeout:
            return

        logger.info(f"Heartbeat monitor started (timeout: {self.heartbeat_timeout}s)")
        last_heartbeat = time.time()

        while not self.shutdown_event.is_set():
            try:
                if self.heartbeat_file.exists():
                    stat = os.stat(self.heartbeat_file)
                    last_heartbeat = stat.st_mtime

                elapsed = time.time() - last_heartbeat
                if elapsed > self.heartbeat_timeout:
                    logger.error(f"Heartbeat timeout exceeded ({elapsed:.0f}s > {self.heartbeat_timeout}s)")
                    self.termination_reason_file.write_text("heartbeat_timeout")
                    self.shutdown_event.set()
                    break

            except Exception as e:
                logger.warning(f"Heartbeat monitor error: {e}")

            time.sleep(15)  # Check every 15 seconds

    def timeout_monitor(self):
        """Monitor total runtime and terminate if max runtime exceeded."""
        if not self.max_runtime_hours or not self.is_master:
            return

        max_seconds = self.max_runtime_hours * 3600
        accumulated = (
            float(self.accumulated_runtime_file.read_text() or "0") if self.accumulated_runtime_file.exists() else 0
        )
        remaining = max_seconds - accumulated

        logger.info(f"Timeout monitor started (remaining: {remaining:.0f}s)")

        while not self.shutdown_event.is_set() and remaining > 0:
            time.sleep(30)  # Check every 30 seconds

            if self.start_time:
                elapsed = time.time() - self.start_time
                remaining = max_seconds - accumulated - elapsed

                if remaining <= 0:
                    logger.info("Max runtime exceeded, initiating shutdown")
                    self.termination_reason_file.write_text("max_runtime_reached")
                    self.shutdown_event.set()
                    break

    def start_monitors(self):
        """Start all monitor threads."""
        monitors = [
            ("heartbeat", self.heartbeat_monitor),
            ("timeout", self.timeout_monitor),
        ]

        for name, target in monitors:
            thread = threading.Thread(target=target, name=f"{name}_monitor", daemon=True)
            thread.start()
            self.monitor_threads.append(thread)
            logger.info(f"Started {name} monitor")

    def run_training(self):
        """Run the main training process."""
        cmd = self.build_command()
        logger.info(f"Running command: {' '.join(cmd)}")

        # Create new process group
        self.main_process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,  # Create new process group
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

        logger.info(f"Started process with PID: {self.main_process.pid}")

        # Wait for process to complete or shutdown signal
        while self.main_process.poll() is None and not self.shutdown_event.is_set():
            time.sleep(1)

        if self.main_process.poll() is None:
            # Process still running, need to terminate
            logger.info("Terminating training process...")
            try:
                # Try graceful shutdown first
                os.killpg(os.getpgid(self.main_process.pid), signal.SIGTERM)
                self.main_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning("Graceful shutdown failed, forcing termination")
                os.killpg(os.getpgid(self.main_process.pid), signal.SIGKILL)
                self.main_process.wait()
        else:
            # Process exited on its own
            if self.main_process.returncode != 0:
                logger.error(f"Training process failed with exit code: {self.main_process.returncode}")

        self.exit_code = self.main_process.returncode or 0
        logger.info(f"Training process exited with code: {self.exit_code}")

    def send_discord_notification(self, emoji: str, title: str, status_msg: str, additional_info: str = ""):
        """Send Discord notification via the shell script."""
        if not self.is_master or not self.enable_discord:
            return

        try:
            # Set required environment variables for the script
            env = os.environ.copy()
            env.update(
                {
                    "IS_MASTER": "true",
                    "ENABLE_DISCORD": "true",
                    "CMD_EXIT": str(self.exit_code),
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

    def set_github_status(self):
        """Update GitHub commit status via the Python script."""
        if not self.is_master or not self.enable_github_status:
            return

        if not self.github_status_state or not self.github_status_description:
            return

        try:
            # Set required environment variables
            env = os.environ.copy()
            env.update(
                {
                    "IS_MASTER": "true",
                    "ENABLE_GITHUB_STATUS": "true",
                    "CMD_EXIT": str(self.exit_code),
                    "GITHUB_STATUS_STATE": self.github_status_state,
                    "GITHUB_STATUS_DESCRIPTION": self.github_status_description,
                }
            )

            subprocess.run(
                [
                    "uv",
                    "run",
                    "devops/skypilot/config/observability/set_github_status.py",
                    self.github_status_state,
                    self.github_status_description,
                ],
                env=env,
                check=False,
            )
        except Exception as e:
            logger.warning(f"Failed to set GitHub status: {e}")

    def handle_master_cleanup(self):
        """Handle master-specific cleanup tasks."""
        if not self.is_master:
            return

        # Check termination reason and set appropriate status
        if self.termination_reason == "heartbeat_timeout":
            logger.error("Job terminated due to heartbeat timeout")
            self.github_status_state = "failure"
            self.github_status_description = f"Job failed - no heartbeat for {self.heartbeat_timeout} seconds"
            self.send_discord_notification("❌", "SkyPilot Job Failed", self.github_status_description)

        elif self.termination_reason == "max_runtime_reached":
            logger.info("Job terminated due to max runtime limit")
            self.github_status_state = "success"
            self.github_status_description = f"Job ran successfully for {self.max_runtime_hours} hours"
            # Map to success exit code
            self.exit_code = EXIT_SUCCESS

        elif self.termination_reason == "force_restart_test":
            logger.info("Job restarting for test purposes")
            self.github_status_state = "pending"
            self.github_status_description = f"Forced a restart test in run #{self.restart_count}"
            # Ensure exit code triggers restart
            self.exit_code = EXIT_FAILURE

        elif not self.termination_reason:
            if self.exit_code == EXIT_SUCCESS:
                logger.info("[SUCCESS] Job completed successfully")
                self.termination_reason = "completed"
                self.github_status_state = "success"
                self.github_status_description = "Job completed successfully"
            else:
                logger.error(f"Job failed with exit code {self.exit_code}")
                self.termination_reason = f"exit_code_{self.exit_code}"
                self.github_status_state = "failure"
                self.github_status_description = f"Job failed with exit code {self.exit_code}"

        elif self.exit_code == EXIT_NCCL_TEST_FAILURE:
            logger.error("Job failed during NCCL tests")
            self.github_status_state = "error"  # Infrastructure issue
            self.github_status_description = "NCCL tests failed - GPU communication issue"
            self.termination_reason = "nccl_test_failure"
            self.send_discord_notification("⚠️", "SkyPilot Job NCCL Config Error", self.github_status_description)

        else:
            logger.error(f"Job failed with exit code {self.exit_code}")
            self.github_status_state = "failure"
            self.github_status_description = f"Job failed with exit code {self.exit_code}"
            self.termination_reason = f"exit_code_{self.exit_code}"
            self.send_discord_notification("❌", "SkyPilot Job Failed", self.github_status_description)

        # Update GitHub status
        self.set_github_status()

    def print_final_summary(self):
        """Print final job summary."""
        logger.info("[SUMMARY] ===== Job Summary =====")
        logger.info(f"[SUMMARY] Metta Run ID: {os.environ.get('METTA_RUN_ID', 'N/A')}")
        logger.info(f"[SUMMARY] Skypilot Task ID: {os.environ.get('SKYPILOT_TASK_ID', 'N/A')}")
        logger.info(f"[SUMMARY] Exit code: {self.exit_code}")
        logger.info(f"[SUMMARY] Termination reason: {self.termination_reason or 'unknown'}")
        logger.info("[SUMMARY] ======================")

        logger.info(
            f"[RUN] Job complete with exit code: {self.exit_code} (reason: {self.termination_reason or 'unknown'})"
        )

    def determine_final_exit_code(self) -> int:
        """Determine the final exit code based on termination reason."""
        if self.termination_reason in ["max_runtime_reached", "completed", "heartbeat_timeout"]:
            logger.info("Will exit with code 0 to prevent SkyPilot restart")
            return EXIT_SUCCESS
        else:
            logger.info(f"Will exit with code: {self.exit_code}")
            return self.exit_code

    def update_accumulated_runtime(self):
        """Update the accumulated runtime file."""
        if self.start_time and self.is_master:
            runtime = time.time() - self.start_time
            accumulated = 0

            if self.accumulated_runtime_file.exists():
                try:
                    accumulated = float(self.accumulated_runtime_file.read_text())
                except ValueError:
                    pass

            total = accumulated + runtime
            self.accumulated_runtime_file.write_text(str(total))
            logger.info(f"Updated accumulated runtime: {total:.0f}s")

    def cleanup(self):
        """Perform full cleanup including notifications and status updates."""
        # Read termination reason
        if self.termination_reason_file.exists():
            self.termination_reason = self.termination_reason_file.read_text().strip()

        logger.info(f"[INFO] Termination reason: {self.termination_reason}")

        # Handle master-specific cleanup
        self.handle_master_cleanup()

        # Print final summary
        self.print_final_summary()

        # Update accumulated runtime
        self.update_accumulated_runtime()

        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"Total runtime: {duration:.0f} seconds ({duration / 60:.1f} minutes)")

        # Wait for monitor threads to finish
        for thread in self.monitor_threads:
            thread.join(timeout=5)

        # Sleep briefly before exit
        time.sleep(1)

    def shutdown(self):
        """Graceful shutdown of all processes."""
        if self.main_process and self.main_process.poll() is None:
            try:
                pgid = os.getpgid(self.main_process.pid)
                logger.info(f"Terminating process group {pgid}")
                os.killpg(pgid, signal.SIGTERM)

                # Wait for graceful shutdown
                try:
                    self.main_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    logger.warning("Process didn't terminate gracefully, using SIGKILL")
                    os.killpg(pgid, signal.SIGKILL)
                    self.main_process.wait()
            except ProcessLookupError:
                pass  # Process already terminated

    def run(self) -> int:
        """Main entry point that runs the full lifecycle and returns exit code."""
        self.log_config()
        self.setup_signal_handlers()

        try:
            # Setup environment
            subprocess.run(["bash", "./devops/skypilot/config/lifecycle/configure_environment.sh"], check=True)

            # Source environment file
            env_file = (
                subprocess.check_output(["uv", "run", "./common/src/metta/common/util/constants.py", "METTA_ENV_FILE"])
                .decode()
                .strip()
            )
            if Path(env_file).exists():
                with open(env_file) as f:
                    for line in f:
                        if "=" in line:
                            key, value = line.strip().split("=", 1)
                            os.environ[key] = value

            # Run preflight checks
            self.run_preflight_checks()

            # Start monitors
            self.start_monitors()

            # Run training
            self.start_time = time.time()
            self.run_training()

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            self.exit_code = EXIT_FAILURE
            if not self.termination_reason:
                self.termination_reason = "unexpected_error"
        finally:
            self.cleanup()

        return self.determine_final_exit_code()


def main():
    """Entry point."""
    manager = SkypilotRunManager()
    sys.exit(manager.run())


if __name__ == "__main__":
    main()
