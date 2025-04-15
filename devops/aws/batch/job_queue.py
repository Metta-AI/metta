#!/usr/bin/env python3
"""
AWS Batch Job Queue Utilities

This module provides functions for interacting with AWS Batch job queues.
"""

from datetime import datetime

import boto3
from botocore.config import Config
from tabulate import tabulate

from .job import format_time_difference

def get_boto3_client(service_name="batch"):
    """Get a boto3 client with standard configuration."""
    config = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=50)
    return boto3.client(service_name, config=config)


def list_job_queues():
    """List all available AWS Batch job queues."""
    batch = get_boto3_client()

    try:
        response = batch.describe_job_queues()
        queues = response["jobQueues"]

        # Format the output
        table_data = []
        for queue in queues:
            name = queue["jobQueueName"]
            state = queue["state"]
            status = queue["status"]
            priority = queue["priority"]

            # Get compute environment names
            compute_envs = [ce["computeEnvironment"].split("/")[-1] for ce in queue["computeEnvironmentOrder"]]
            compute_env_str = ", ".join(compute_envs)

            table_data.append([name, state, status, priority, compute_env_str])

        # Print the table
        headers = ["Queue Name", "State", "Status", "Priority", "Compute Environments"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        return [queue["jobQueueName"] for queue in queues]
    except Exception as e:
        print(f"Error retrieving job queues: {str(e)}")
        return []


def get_job_queue_info(queue_name, max_jobs=5):
    """Get detailed information about a specific job queue."""
    batch = get_boto3_client()

    try:
        response = batch.describe_job_queues(jobQueues=[queue_name])

        if not response["jobQueues"]:
            print(f"Job queue '{queue_name}' not found")
            return None

        queue = response["jobQueues"][0]

        # Print basic information
        print(f"\nJob Queue: {queue['jobQueueName']}")
        print(f"ARN: {queue['jobQueueArn']}")
        print(f"State: {queue['state']}")
        print(f"Status: {queue['status']}")
        print(f"Status Reason: {queue.get('statusReason', 'N/A')}")
        print(f"Priority: {queue['priority']}")

        # Print compute environments
        print("\nCompute Environments:")
        for ce in queue["computeEnvironmentOrder"]:
            print(f"  - {ce['computeEnvironment'].split('/')[-1]} (Order: {ce['order']})")

        # Get job statistics
        print("\nJob Statistics:")
        try:
            # Get jobs by status
            for status in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED"]:
                try:
                    response = batch.list_jobs(jobQueue=queue_name, jobStatus=status, maxResults=max_jobs)
                    job_summaries = response.get("jobSummaryList", [])
                    job_count = len(job_summaries)

                    # If there are more jobs, get the total count
                    if "nextToken" in response:
                        job_count = f"{job_count}+ (more available)"

                    print(f"  - {status}: {job_count}")

                    # If there are jobs, show details for the most recent ones
                    if job_summaries:
                        # Sort by creation time (newest first)
                        job_summaries.sort(key=lambda x: x.get("createdAt", 0), reverse=True)

                        # Limit to max_jobs
                        job_summaries = job_summaries[:max_jobs]

                        # Get job details
                        job_ids = [job["jobId"] for job in job_summaries]
                        job_details = batch.describe_jobs(jobs=job_ids)["jobs"]

                        # Create a lookup table for job details
                        job_details_map = {job["jobId"]: job for job in job_details}

                        # Format the output
                        table_data = []
                        for summary in job_summaries:
                            job_id = summary["jobId"]
                            job_name = summary["jobName"]

                            # Get additional details if available
                            details = job_details_map.get(job_id, {})
                            created_at = details.get("createdAt", summary.get("createdAt", 0))
                            started_at = details.get("startedAt", 0)
                            stopped_at = details.get("stoppedAt", 0)

                            # Format timestamps
                            created_str = (
                                datetime.fromtimestamp(created_at / 1000).strftime("%Y-%m-%d %H:%M:%S")
                                if created_at
                                else "N/A"
                            )

                            # Calculate duration
                            if started_at and stopped_at:
                                duration = format_time_difference(started_at, stopped_at)
                            elif started_at:
                                duration = format_time_difference(started_at)
                            else:
                                duration = "N/A"

                            # Get job definition
                            job_definition = details.get("jobDefinition", "").split("/")[-1].split(":")[0]

                            # Get number of attempts
                            attempts = len(details.get("attempts", []))

                            table_data.append([job_id, job_name, created_str, duration, job_definition, attempts])

                        # Print the table
                        if table_data:
                            print(f"\n    Recent {status} Jobs:")
                            headers = ["Job ID", "Name", "Created", "Duration", "Job Definition", "Attempts"]
                            print(
                                tabulate(
                                    table_data, headers=headers, tablefmt="grid", maxcolwidths=[20, 30, 20, 10, 20, 8]
                                )
                            )
                except Exception as e:
                    print(f"  Error retrieving {status} jobs: {str(e)}")
        except Exception as e:
            print(f"  Error retrieving job statistics: {str(e)}")

        return queue
    except Exception as e:
        print(f"Error retrieving job queue information: {str(e)}")
        return None
