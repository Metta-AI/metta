#!/usr/bin/env -S uv run
# ruff: noqa: E402
import argparse
import subprocess
import time
from datetime import datetime

import sky

from metta.tools.utils.auto_config import auto_run_name


def wait_for_job(run_id: str, poll_interval: int = 60) -> bool:
    """
    Waits for the SkyPilot job with the given run_id to complete.
    Returns True if SUCCEEDED, False otherwise.
    """
    print(f"[{datetime.now()}] Waiting for job {run_id}...")

    while True:
        # refresh=True forces a fetch from the cluster controllers/cloud
        try:
            jobs = sky.jobs.queue(refresh=True)
        except Exception as e:
            print(f"Error fetching job queue: {e}")
            time.sleep(poll_interval)
            continue

        # Find our job
        target_job = None
        for job in jobs:
            if job["job_name"] == run_id:
                target_job = job
                break

        if not target_job:
            print(f"Job {run_id} not found in queue yet...")
        else:
            status = target_job["status"]
            # SkyPilot status strings: INIT, PROVISIONING, PENDING, RUNNING, STOPPING, SUCCEEDED, FAILED, CANCELLED
            if status == sky.JobStatus.SUCCEEDED:
                return True
            if status in [sky.JobStatus.FAILED, sky.JobStatus.CANCELLED]:
                return False

            print(f"[{datetime.now()}] Job {run_id} status: {status}")

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Continuous Kickstart Training Loop")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    # Configuration
    # Initial teacher from the original recipe
    current_teacher = "s3://softmax-public/policies/av.student.11.26.28/av.student.11.26.28:v4000.mpt"

    total_timesteps = 10_000_000_000  # 10B
    kickstart_steps = 1_000_000_000  # 1B (Kickstart phase length)

    module = "recipes.experiment.abes.kickstart.sliced.train"

    print("Starting Continuous Kickstart Training Loop")
    print(f"Initial Teacher: {current_teacher}")
    print(f"Total Timesteps per Iteration: {total_timesteps}")
    print("---------------------------------------------------")

    iteration = 0
    while True:
        iteration += 1
        # Generate a unique run ID
        run_id = f"continuous_student_iter{iteration}_{auto_run_name()}"

        print(f"\n>>> Starting Iteration {iteration}")
        print(f">>> Run ID: {run_id}")
        print(f">>> Teacher: {current_teacher}")

        # Construct launch command
        cmd = [
            "uv",
            "run",
            "devops/skypilot/launch.py",
            module,
            "--skip-git-check",
            f"run={run_id}",
            f"teacher_uri={current_teacher}",
            f"total_timesteps={total_timesteps}",
            f"kickstart_steps={kickstart_steps}",
        ]

        if args.dry_run:
            # Pass --dry-run to launch.py as well to verify configuration
            cmd.append("--dry-run")

        print(f"Running command: {' '.join(cmd)}")

        try:
            # Launch the job
            subprocess.check_call(cmd)
            print("Job submitted successfully (or dry-run completed).")

        except subprocess.CalledProcessError as e:
            print(f"Failed to submit job: {e}")
            break

        if args.dry_run:
            print("Dry run mode: skipping wait and loop continuation.")
            break

        # Wait for the job to complete
        success = wait_for_job(run_id)

        if not success:
            print(f"Job {run_id} failed. Stopping loop.")
            break

        print(f"Job {run_id} completed successfully.")

        # Set up next teacher
        # Assumption: Policy is saved at s3://softmax-public/policies/{run_id}/{run_id}:v{total_timesteps}.mpt
        # The trainer usually appends the step count to the version.
        # We assume total_timesteps is reached.

        current_teacher = f"s3://softmax-public/policies/{run_id}/{run_id}:v{total_timesteps}.mpt"
        print(f"Next iteration will use teacher: {current_teacher}")

        # Pause briefly before next iteration
        time.sleep(10)


if __name__ == "__main__":
    main()
