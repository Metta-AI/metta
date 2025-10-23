"""Experiment monitoring - live status display and log streaming."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

from devops.skypilot.utils.job_helpers import check_job_statuses
from metta.experiment.state import ExperimentState
from metta.jobs.metrics import extract_wandb_info
from metta.jobs.runner import RemoteJob


class ExperimentMonitor:
    """Monitor experiment status and display live updates."""

    def __init__(self, instance_id: str, refresh_interval: int = 10):
        """Initialize experiment monitor.

        Args:
            instance_id: Experiment instance ID to monitor
            refresh_interval: Seconds between status updates

        Raises:
            ValueError: If instance not found
        """
        self.instance_id = instance_id
        self.refresh_interval = refresh_interval
        self.state = ExperimentState.load(instance_id)

        if self.state is None:
            raise ValueError(f"Experiment instance not found: {instance_id}")

    def run(self) -> int:
        """Run live monitoring loop.

        Returns:
            0 if experiment completed successfully, 1 otherwise
        """
        print(f"\nMonitoring experiment: {self.instance_id}")
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                self._refresh_and_display()

                # Check if complete
                if self.state.status in ("completed", "failed", "cancelled"):
                    print(f"\nExperiment {self.state.status.upper()}")
                    return 0 if self.state.status == "completed" else 1

                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            return 0

    def attach(self) -> int:
        """Attach to first running job and stream logs.

        Returns:
            Exit code of job (0 on success)
        """
        print(f"\nAttaching to experiment: {self.instance_id}\n")

        # Find first running job and attach to it
        for job_name, job_state in self.state.jobs.items():
            if job_state.status == "running" and job_state.job_id:
                print(f"Attaching to job: {job_name} (Job ID: {job_state.job_id})")
                print(f"{'=' * 80}\n")

                # Create appropriate job type and attach
                if job_state.config.execution == "remote":
                    job = RemoteJob(
                        name=job_name,
                        module=job_state.config.module,
                        args=[f"{k}={v}" for k, v in job_state.config.args.items()],
                        job_id=int(job_state.job_id),
                        log_dir=str(Path("experiments/logs") / self.instance_id),
                    )
                else:
                    # Local job - can't really attach to existing PID
                    print("Cannot attach to local jobs (would need to re-run)")
                    return 1

                # Stream logs
                result = job.wait(stream_output=True)

                # Update state with result
                self.state.update_job_status(
                    job_name,
                    status="completed" if result.exit_code == 0 else "failed",
                    exit_code=result.exit_code,
                    completed_at=datetime.utcnow().isoformat(timespec="seconds"),
                )

                return result.exit_code

        print("No running jobs to attach to")
        return 1

    def _refresh_and_display(self) -> None:
        """Refresh job states from SkyPilot and display status table."""
        # Reload state from disk (may have been updated by other process)
        self.state = ExperimentState.load(self.instance_id)

        # Poll job statuses and update state
        self._poll_job_statuses()

        # Clear screen and display table
        os.system("clear" if os.name == "posix" else "cls")

        print(f"\n{'=' * 80}")
        print(f"Experiment: {self.instance_id}")
        print(f"Status: {self.state.status.upper()}")
        print(f"Updated: {self.state.updated_at}")
        print(f"{'=' * 80}\n")

        # Display job table
        self._display_job_table()

        print(f"\nRefreshing every {self.refresh_interval}s (Ctrl+C to stop)")

    def _poll_job_statuses(self) -> None:
        """Poll SkyPilot for job statuses and update state."""
        try:
            # Collect all job IDs to check
            job_ids = []
            for job_state in self.state.jobs.values():
                if job_state.status == "running" and job_state.job_id and job_state.config.execution == "remote":
                    try:
                        job_ids.append(int(job_state.job_id))
                    except ValueError:
                        pass

            if not job_ids:
                return

            # Batch check all job statuses
            job_statuses = check_job_statuses(job_ids)

            # Update state for each job
            for job_name, job_state in self.state.jobs.items():
                if job_state.status != "running":
                    continue
                if not job_state.job_id:
                    continue
                if job_state.config.execution != "remote":
                    continue

                try:
                    job_id = int(job_state.job_id)
                except ValueError:
                    continue

                job_info = job_statuses.get(job_id)
                if not job_info:
                    # Job not found - might have been cleaned up
                    continue

                sky_status = job_info["status"]

                # Check if job completed (include all terminal states)
                if sky_status in (
                    "SUCCEEDED",
                    "FAILED",
                    "CANCELLED",
                    "FAILED_SETUP",
                    "FAILED_DRIVER",
                    "UNKNOWN",
                    "ERROR",
                ):
                    new_status = {
                        "SUCCEEDED": "completed",
                        "FAILED": "failed",
                        "FAILED_SETUP": "failed",
                        "FAILED_DRIVER": "failed",
                        "UNKNOWN": "failed",
                        "ERROR": "failed",
                        "CANCELLED": "cancelled",
                    }[sky_status]

                    self.state.update_job_status(
                        job_name,
                        status=new_status,
                        exit_code=0 if sky_status == "SUCCEEDED" else 1,
                        completed_at=datetime.utcnow().isoformat(timespec="seconds"),
                    )

                    # Try to extract WandB info from logs
                    if job_state.logs_path and Path(job_state.logs_path).exists():
                        log_text = Path(job_state.logs_path).read_text(errors="ignore")
                        wandb_info = extract_wandb_info(log_text)
                        if wandb_info:
                            self.state.update_job_status(
                                job_name,
                                wandb_url=wandb_info.url,
                                wandb_run_id=wandb_info.run_id,
                            )

        except Exception as e:
            # Log but don't crash monitoring on transient errors
            logging.warning(f"Failed to poll job statuses: {e}")

    def _display_job_table(self) -> None:
        """Display formatted table of job states."""
        # Header
        print(f"{'Job Name':<30} {'Status':<15} {'Job ID':<10} {'Execution':<10} {'WandB Run':<30}")
        print(f"{'─' * 30} {'─' * 15} {'─' * 10} {'─' * 10} {'─' * 30}")

        # Rows
        for job_name, job_state in self.state.jobs.items():
            # Status with symbol
            if job_state.status == "completed":
                status_str = f"✓ {job_state.status.upper()}"
            elif job_state.status == "failed":
                status_str = f"✗ {job_state.status.upper()}"
            elif job_state.status == "running":
                status_str = f"⋯ {job_state.status.upper()}"
            else:
                status_str = f"○ {job_state.status.upper()}"

            job_id_str = job_state.job_id or "-"
            execution_str = job_state.config.execution
            wandb_str = job_state.wandb_run_id or "-"

            print(f"{job_name:<30} {status_str:<15} {job_id_str:<10} {execution_str:<10} {wandb_str:<30}")

        print()
