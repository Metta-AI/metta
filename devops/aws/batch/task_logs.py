#!/usr/bin/env python3
import argparse
import re
from datetime import datetime

import boto3

# Import functions from the job module
from devops.aws.batch.job import (
    format_time_difference,
    get_job_attempts,
    get_job_details,
    print_job_logs,
)


def get_batch_job_queues():
    """Get a list of all AWS Batch job queues."""
    batch = boto3.client("batch")
    response = batch.describe_job_queues()
    return [queue["jobQueueName"] for queue in response["jobQueues"]]


def list_recent_jobs(job_queue=None, max_jobs=10, interactive=False, debug=False):
    """
    List recent jobs and optionally allow the user to select one to view logs.

    Args:
        job_queue (str): The job queue to list jobs from
        max_jobs (int): The maximum number of jobs to list
        interactive (bool): Whether to allow the user to select a job
        debug (bool): Whether to show debug information

    Returns:
        str: The selected job ID if interactive is True, None otherwise
    """
    jobs = get_job_details(job_prefix=None, job_queue=job_queue, max_jobs=max_jobs)

    if not jobs:
        print("No recent jobs found" + (f" in queue {job_queue}" if job_queue else ""))
        return None

    print("\nRecent jobs" + (f" in queue {job_queue}" if job_queue else "") + ":")
    print(f"{'#':<3} {'Name':<30} {'ID':<36} {'Status':<10} {'Age':<10} {'Created At'}")
    print("-" * 100)

    for i, job in enumerate(jobs):
        job_id = job.get("jobId", "Unknown")
        job_name = job.get("jobName", "Unknown")
        status = job.get("status", "Unknown")
        created_at = job.get("createdAt", 0)

        # Format the created_at timestamp
        if created_at:
            created_at_str = datetime.fromtimestamp(created_at / 1000).strftime("%Y-%m-%d %H:%M:%S")
            age = format_time_difference(created_at / 1000)
        else:
            created_at_str = "Unknown"
            age = "Unknown"

        print(f"{i + 1:<3} {job_name[:30]:<30} {job_id:<36} {status:<10} {age:<10} {created_at_str}")

    if interactive:
        while True:
            try:
                choice = input("\nEnter job number to view logs (or press Enter to exit): ")
                if not choice:
                    return None

                choice = int(choice)
                if 1 <= choice <= len(jobs):
                    selected_job = jobs[choice - 1]
                    job_id = selected_job.get("jobId")

                    # Show logs for the selected job
                    print_job_logs(job_id, tail=False, debug=debug)

                    return job_id
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(jobs)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return None

    return None


def show_job_logs(job, tail=False, latest_attempt=False, attempt_index=None, node_index=None, debug=False):
    """
    Show logs for a specific job.

    Args:
        job (str): The job ID or name to show logs for
        tail (bool): Whether to continuously poll for new logs
        latest_attempt (bool): Whether to show logs only for the latest attempt
        attempt_index (int): The specific attempt index to show logs for
        node_index (int): The specific node index to show logs for (for multi-node jobs)
        debug (bool): Whether to show debug information
    """
    # Check if the input is a job ID or job name
    if re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", job):
        # Input is a job ID
        job_id = job
    else:
        # Input is a job name, try to find the job ID
        jobs = get_job_details(job_prefix=job)
        if not jobs:
            print(f"No job found with name or ID matching '{job}'")
            return

        if len(jobs) > 1:
            print(f"Multiple jobs found with name or ID matching '{job}':")
            for i, j in enumerate(jobs):
                job_id = j.get("jobId", "Unknown")
                job_name = j.get("jobName", "Unknown")
                status = j.get("status", "Unknown")
                print(f"{i + 1}. {job_name} (ID: {job_id}, Status: {status})")

            while True:
                try:
                    choice = input("\nEnter job number to view logs (or press Enter to exit): ")
                    if not choice:
                        return

                    choice = int(choice)
                    if 1 <= choice <= len(jobs):
                        job_id = jobs[choice - 1].get("jobId")
                        break
                    else:
                        print(f"Invalid choice. Please enter a number between 1 and {len(jobs)}")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                except KeyboardInterrupt:
                    print("\nOperation cancelled.")
                    return
        else:
            job_id = jobs[0].get("jobId")

    # Get job attempts
    attempts = get_job_attempts(job_id)

    # Determine which attempt to show logs for
    attempt_num = None

    if latest_attempt and attempts:
        # Show logs for the latest attempt
        attempt_num = len(attempts) - 1
    elif attempt_index is not None:
        # Show logs for the specified attempt
        if 0 <= attempt_index < len(attempts):
            attempt_num = attempt_index
        else:
            print(f"Invalid attempt index {attempt_index}. Job has {len(attempts)} attempts (0-{len(attempts) - 1})")
            return
    elif attempts and len(attempts) > 1:
        # If there are multiple attempts and no specific one was requested, ask the user
        print(f"\nJob has {len(attempts)} attempts:")
        for i, attempt in enumerate(attempts):
            container = attempt.get("container", {})
            exit_code = container.get("exitCode", "N/A")
            reason = container.get("reason", "N/A")

            # Format timestamps
            started_at = attempt.get("startedAt")
            if started_at:
                started_at_str = datetime.fromtimestamp(started_at / 1000).strftime("%Y-%m-%d %H:%M:%S")
            else:
                started_at_str = "N/A"

            stopped_at = attempt.get("stoppedAt")
            if stopped_at:
                stopped_at_str = datetime.fromtimestamp(stopped_at / 1000).strftime("%Y-%m-%d %H:%M:%S")
                duration = format_time_difference(started_at / 1000, stopped_at / 1000) if started_at else "N/A"
            else:
                stopped_at_str = "N/A"
                duration = "Running" if started_at else "N/A"

            print(
                f"Attempt {i + 1}: Started: {started_at_str}, Stopped: {stopped_at_str}, Duration: {duration}, Exit Code: {exit_code}, Reason: {reason}"
            )

        while True:
            try:
                choice = input("\nEnter attempt number to view logs (or press Enter to view all attempts): ")
                if not choice:
                    # View all attempts
                    break

                choice = int(choice)
                if 1 <= choice <= len(attempts):
                    attempt_num = choice - 1
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 1 and {len(attempts)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return

    # Show logs for the job
    print_job_logs(job_id, attempt_index=attempt_num, node_index=node_index, tail=tail, debug=debug)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="View logs for AWS Batch jobs")
    parser.add_argument("--job", help="Job ID or name to view logs for")
    parser.add_argument("--queue", default="metta-jq", help="Job queue to list jobs from")
    parser.add_argument("--tail", action="store_true", help="Continuously poll for new logs")
    parser.add_argument("--max-jobs", type=int, default=10, help="Maximum number of jobs to list")
    parser.add_argument("--latest", action="store_true", help="Show logs for the latest attempt only")
    parser.add_argument("--attempt", type=int, help="Show logs for a specific attempt (0-based index)")
    parser.add_argument("--node", type=int, help="Show logs for a specific node (for multi-node jobs)")
    parser.add_argument("--debug", action="store_true", help="Show debug information")

    args = parser.parse_args()

    if args.job:
        show_job_logs(
            args.job,
            tail=args.tail,
            latest_attempt=args.latest,
            attempt_index=args.attempt,
            node_index=args.node,
            debug=args.debug,
        )
    else:
        list_recent_jobs(job_queue=args.queue, max_jobs=args.max_jobs, debug=args.debug)


if __name__ == "__main__":
    main()
