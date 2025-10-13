"""Experiment lifecycle management - cancellation and cleanup."""

import os
import signal

import sky

from metta.experiment.state import ExperimentState


class ExperimentManager:
    """Manages experiment lifecycle operations."""

    def __init__(self, instance_id: str):
        """Initialize experiment manager.

        Args:
            instance_id: Experiment instance ID

        Raises:
            ValueError: If instance not found
        """
        self.instance_id = instance_id
        self.state = ExperimentState.load(instance_id)

        if self.state is None:
            raise ValueError(f"Experiment instance not found: {instance_id}")

    def cancel_all(self) -> int:
        """Cancel all running jobs in the experiment.

        Returns:
            0 on success, 1 if no jobs were cancelled
        """

        print(f"\nCancelling all jobs in experiment: {self.instance_id}\n")

        cancelled_count = 0
        failed_count = 0

        for job_name, job_state in self.state.jobs.items():
            if job_state.status != "running":
                continue

            if not job_state.job_id:
                continue

            # Cancel based on execution type
            if job_state.spec.execution == "remote":
                # Cancel SkyPilot job
                try:
                    job_id = int(job_state.job_id)
                    print(f"Cancelling remote job: {job_name} (Job ID: {job_id})")
                    sky.jobs.cancel(job_ids=[job_id])
                    self.state.update_job_status(job_name, status="cancelled")
                    cancelled_count += 1
                    print("  ✓ Cancelled")
                except Exception as e:
                    print(f"  ✗ Failed to cancel: {e}")
                    failed_count += 1

            elif job_state.spec.execution == "local":
                # Kill local process
                try:
                    pid = int(job_state.job_id)
                    print(f"Killing local job: {job_name} (PID: {pid})")
                    os.kill(pid, signal.SIGTERM)
                    self.state.update_job_status(job_name, status="cancelled")
                    cancelled_count += 1
                    print("  ✓ Killed")
                except Exception as e:
                    print(f"  ✗ Failed to kill: {e}")
                    failed_count += 1

        # Summary
        print("\nSummary:")
        print(f"  Cancelled: {cancelled_count}")
        print(f"  Failed: {failed_count}")

        if cancelled_count == 0:
            print("\nNo running jobs found to cancel")
            return 1

        return 0
