#!/usr/bin/env python3
"""
AWS Batch Job Logs Utilities

This module provides functions for retrieving and displaying logs from AWS Batch jobs.
"""

import re
import time
from datetime import datetime

import boto3
from botocore.config import Config

def get_boto3_client(service_name):
    """Get a boto3 client with standard configuration."""
    config = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=50)
    return boto3.client(service_name, config=config)


def get_job_attempts(job_id):
    """Get all attempts for a job."""
    batch = get_boto3_client("batch")

    try:
        response = batch.describe_jobs(jobs=[job_id])
        if not response["jobs"]:
            print(f"Job '{job_id}' not found")
            return []

        job = response["jobs"][0]
        return job.get("attempts", [])
    except Exception as e:
        print(f"Error retrieving job attempts: {str(e)}")
        return []


def get_job_log_streams(job_id, attempt_index=None):
    """Get CloudWatch log streams for a job."""
    batch = get_boto3_client("batch")
    logs = get_boto3_client("logs")

    try:
        # Get job details
        response = batch.describe_jobs(jobs=[job_id])
        if not response["jobs"]:
            print(f"Job '{job_id}' not found")
            return []

        job = response["jobs"][0]
        attempts = job.get("attempts", [])

        if not attempts:
            print(f"No attempts found for job '{job_id}'")
            return []

        # If attempt_index is not specified, use the latest attempt
        if attempt_index is None:
            attempt_index = len(attempts) - 1
        elif attempt_index >= len(attempts):
            print(f"Attempt index {attempt_index} is out of range (max: {len(attempts) - 1})")
            return []

        attempt = attempts[attempt_index]
        container = attempt.get("container", {})
        log_stream_name = container.get("logStreamName")

        if not log_stream_name:
            print(f"No log stream found for job '{job_id}' attempt {attempt_index}")
            return []

        # For multi-node jobs, there might be multiple log streams
        if job.get("nodeProperties") and job.get("nodeProperties").get("numNodes", 0) > 1:
            # Extract the base log stream name (without the node index)
            base_log_stream = re.sub(r"/\d+$", "", log_stream_name)

            # List all log streams with the base name
            log_streams = []
            try:
                response = logs.describe_log_streams(logGroupName="/aws/batch/job", logStreamNamePrefix=base_log_stream)

                for stream in response.get("logStreams", []):
                    log_streams.append(stream["logStreamName"])

                return log_streams
            except Exception as e:
                print(f"Error listing log streams: {str(e)}")
                # Fall back to the single log stream
                return [log_stream_name]
        else:
            return [log_stream_name]
    except Exception as e:
        print(f"Error retrieving job log streams: {str(e)}")
        return []


def find_alternative_log_streams(job_id, attempt_index=None, debug=False):
    """Find alternative log streams for a job when the standard method fails."""
    logs = get_boto3_client("logs")

    try:
        # Try to find log streams based on job ID pattern
        log_streams = []

        # Pattern 1: Standard AWS Batch log stream pattern
        prefix = f"jobDefinition/*/job/{job_id}"
        if attempt_index is not None:
            prefix = f"{prefix}/attempt/{attempt_index}"

        if debug:
            print(f"Searching for log streams with prefix: {prefix}")

        try:
            response = logs.describe_log_streams(logGroupName="/aws/batch/job", logStreamNamePrefix=prefix)

            for stream in response.get("logStreams", []):
                log_streams.append(stream["logStreamName"])
        except Exception as e:
            if debug:
                print(f"Error searching for log streams with prefix '{prefix}': {str(e)}")

        # Pattern 2: Alternative pattern with just the job ID
        if not log_streams:
            try:
                response = logs.describe_log_streams(logGroupName="/aws/batch/job", logStreamNamePrefix=job_id)

                for stream in response.get("logStreams", []):
                    log_streams.append(stream["logStreamName"])
            except Exception as e:
                if debug:
                    print(f"Error searching for log streams with job ID '{job_id}': {str(e)}")

        return log_streams
    except Exception as e:
        if debug:
            print(f"Error finding alternative log streams: {str(e)}")
        return []


def format_time_difference(timestamp, end_timestamp=None):
    """Format the time difference between a timestamp and now (or another timestamp)."""
    if not timestamp:
        return "N/A"

    # Convert milliseconds to seconds if necessary
    if timestamp > 1000000000000:  # If timestamp is in milliseconds
        timestamp = timestamp / 1000

    # If end_timestamp is provided, use it; otherwise use current time
    if end_timestamp:
        # Convert milliseconds to seconds if necessary
        if end_timestamp > 1000000000000:  # If timestamp is in milliseconds
            end_timestamp = end_timestamp / 1000

        diff_seconds = end_timestamp - timestamp
    else:
        # Calculate difference from now
        now = time.time()
        diff_seconds = now - timestamp

    # Format the difference
    if diff_seconds < 0:
        return "Future"

    if diff_seconds < 60:
        return f"{int(diff_seconds)}s"

    if diff_seconds < 3600:
        minutes = int(diff_seconds / 60)
        seconds = int(diff_seconds % 60)
        return f"{minutes}m {seconds}s"

    if diff_seconds < 86400:
        hours = int(diff_seconds / 3600)
        minutes = int((diff_seconds % 3600) / 60)
        return f"{hours}h {minutes}m"

    days = int(diff_seconds / 86400)
    hours = int((diff_seconds % 86400) / 3600)
    return f"{days}d {hours}h"


def get_log_events(log_group, log_stream, start_time=None, tail=False):
    """Get log events from a CloudWatch log stream."""
    logs = get_boto3_client("logs")

    try:
        # Parameters for the get_log_events call
        params = {"logGroupName": log_group, "logStreamName": log_stream, "startFromHead": True}

        if start_time:
            # Convert to milliseconds if necessary
            if start_time < 1000000000000:  # If timestamp is in seconds
                start_time = start_time * 1000
            params["startTime"] = int(start_time)

        # Get the first batch of log events
        response = logs.get_log_events(**params)
        events = response.get("events", [])

        # If tail is True, we only want the last batch of events
        if tail and not events:
            # Try again with startFromHead=False to get the most recent events
            params["startFromHead"] = False
            response = logs.get_log_events(**params)
            events = response.get("events", [])

        # If not tailing, get all events
        if not tail:
            next_token = response.get("nextForwardToken")

            # Continue getting events until we've retrieved all of them
            while next_token:
                params["nextToken"] = next_token
                response = logs.get_log_events(**params)
                new_events = response.get("events", [])

                if not new_events:
                    break

                events.extend(new_events)

                # Check if we've reached the end
                if response.get("nextForwardToken") == next_token:
                    break

                next_token = response.get("nextForwardToken")

        return events
    except Exception as e:
        print(f"Error retrieving log events: {str(e)}")
        return []


def print_job_logs(job_id, attempt_index=None, node_index=None, tail=False, debug=False):
    """Print logs for a job."""
    batch = get_boto3_client("batch")

    try:
        # Get job details
        response = batch.describe_jobs(jobs=[job_id])
        if not response["jobs"]:
            print(f"Job '{job_id}' not found")
            return

        job = response["jobs"][0]
        job_name = job.get("jobName", "Unknown")
        job_status = job.get("status", "Unknown")

        print(f"Logs for job '{job_id}' (Name: {job_name}, Status: {job_status})")

        # Get job attempts
        attempts = job.get("attempts", [])

        if not attempts:
            print("No attempts found for this job")

            # Try to find alternative log streams
            log_streams = find_alternative_log_streams(job_id, attempt_index, debug)

            if log_streams:
                print(f"Found {len(log_streams)} alternative log streams")

                for i, log_stream in enumerate(log_streams):
                    if node_index is not None and i != node_index:
                        continue

                    print(f"\nLog Stream: {log_stream}")
                    events = get_log_events("/aws/batch/job", log_stream, tail=tail)

                    if not events:
                        print("No log events found")
                        continue

                    for event in events:
                        timestamp = event.get("timestamp", 0) / 1000  # Convert to seconds
                        time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        message = event.get("message", "")
                        print(f"[{time_str}] {message}")
            else:
                print("No log streams found")

            return

        # If attempt_index is not specified, use the latest attempt
        if attempt_index is None:
            attempt_index = len(attempts) - 1
        elif attempt_index >= len(attempts):
            print(f"Attempt index {attempt_index} is out of range (max: {len(attempts) - 1})")
            return

        attempt = attempts[attempt_index]
        container = attempt.get("container", {})
        log_stream_name = container.get("logStreamName")

        if not log_stream_name:
            print(f"No log stream found for attempt {attempt_index}")

            # Try to find alternative log streams
            log_streams = find_alternative_log_streams(job_id, attempt_index, debug)

            if log_streams:
                print(f"Found {len(log_streams)} alternative log streams")

                for i, log_stream in enumerate(log_streams):
                    if node_index is not None and i != node_index:
                        continue

                    print(f"\nLog Stream: {log_stream}")
                    events = get_log_events("/aws/batch/job", log_stream, tail=tail)

                    if not events:
                        print("No log events found")
                        continue

                    for event in events:
                        timestamp = event.get("timestamp", 0) / 1000  # Convert to seconds
                        time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        message = event.get("message", "")
                        print(f"[{time_str}] {message}")
            else:
                print("No log streams found")

            return

        # For multi-node jobs, there might be multiple log streams
        if job.get("nodeProperties") and job.get("nodeProperties").get("numNodes", 0) > 1:
            # Extract the base log stream name (without the node index)
            base_log_stream = re.sub(r"/\d+$", "", log_stream_name)

            # List all log streams with the base name
            log_streams = []
            try:
                logs = get_boto3_client("logs")
                response = logs.describe_log_streams(logGroupName="/aws/batch/job", logStreamNamePrefix=base_log_stream)

                for stream in response.get("logStreams", []):
                    log_streams.append(stream["logStreamName"])

                if not log_streams:
                    print("No log streams found")
                    return

                # Sort log streams by node index
                log_streams.sort(
                    key=lambda s: int(re.search(r"/(\d+)$", s).group(1)) if re.search(r"/(\d+)$", s) else 0
                )

                # If node_index is specified, only show logs for that node
                if node_index is not None:
                    node_stream = f"{base_log_stream}/{node_index}"
                    if node_stream in log_streams:
                        log_streams = [node_stream]
                    else:
                        print(f"No log stream found for node {node_index}")
                        return
                else:
                    # If node_index is not specified, only show logs for the first node (main node)
                    main_node = job.get("nodeProperties", {}).get("mainNode", 0)
                    node_stream = f"{base_log_stream}/{main_node}"
                    if node_stream in log_streams:
                        log_streams = [node_stream]
                        print(f"Showing logs for main node ({main_node}). Use --node=X to view logs for other nodes.")
                    else:
                        # If main node log stream not found, use the first available log stream
                        if log_streams:
                            log_streams = [log_streams[0]]
                            node_match = re.search(r"/(\d+)$", log_streams[0])
                            node_idx = int(node_match.group(1)) if node_match else 0
                            print(f"Showing logs for node {node_idx}. Use --node=X to view logs for other nodes.")

                # Print logs for each stream
                for log_stream in log_streams:
                    # Extract node index from log stream name
                    node_match = re.search(r"/(\d+)$", log_stream)
                    node_idx = int(node_match.group(1)) if node_match else 0

                    print(f"\nNode {node_idx} Log Stream: {log_stream}")
                    events = get_log_events("/aws/batch/job", log_stream, tail=tail)

                    if not events:
                        print("No log events found")
                        continue

                    for event in events:
                        timestamp = event.get("timestamp", 0) / 1000  # Convert to seconds
                        time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        message = event.get("message", "")
                        print(f"[{time_str}] {message}")
            except Exception as e:
                print(f"Error listing log streams: {str(e)}")
                # Fall back to the single log stream
                print(f"\nLog Stream: {log_stream_name}")
                events = get_log_events("/aws/batch/job", log_stream_name, tail=tail)

                if not events:
                    print("No log events found")
                    return

                for event in events:
                    timestamp = event.get("timestamp", 0) / 1000  # Convert to seconds
                    time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                    message = event.get("message", "")
                    print(f"[{time_str}] {message}")
        else:
            # Single node job
            print(f"\nLog Stream: {log_stream_name}")
            events = get_log_events("/aws/batch/job", log_stream_name, tail=tail)

            if not events:
                print("No log events found")
                return

            for event in events:
                timestamp = event.get("timestamp", 0) / 1000  # Convert to seconds
                time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                message = event.get("message", "")
                print(f"[{time_str}] {message}")
    except Exception as e:
        print(f"Error retrieving job logs: {str(e)}")


def get_job_logs(job_id_or_name, attempt_index=None, node_index=None, tail=False, debug=False):
    """Get logs for a job by ID or name."""
    batch = get_boto3_client("batch")

    try:
        # First try to get the job by ID
        response = batch.describe_jobs(jobs=[job_id_or_name])

        # If no job found by ID, try to find by name
        if not response["jobs"]:
            # We need to list jobs from all queues to find by name
            job_id = None

            # Get all job queues
            try:
                queues_response = batch.describe_job_queues()
                job_queues = [queue["jobQueueName"] for queue in queues_response["jobQueues"]]

                # Search for the job in each queue
                for queue in job_queues:
                    # Check all job statuses
                    for status in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED"]:
                        try:
                            jobs_response = batch.list_jobs(jobQueue=queue, jobStatus=status, maxResults=100)

                            # Look for a job with the specified name
                            for job_summary in jobs_response.get("jobSummaryList", []):
                                if job_summary["jobName"] == job_id_or_name:
                                    # Found a job with the specified name
                                    job_id = job_summary["jobId"]
                                    break

                            if job_id:
                                break
                        except Exception:
                            continue

                    if job_id:
                        break

                if not job_id:
                    print(f"No job found with ID or name '{job_id_or_name}'")
                    return
            except Exception as e:
                print(f"Error retrieving job queues: {str(e)}")
                print(f"Job '{job_id_or_name}' not found")
                return
        else:
            job_id = job_id_or_name

        # Now that we have the job ID, print the logs
        print_job_logs(job_id, attempt_index, node_index, tail, debug)
    except Exception as e:
        print(f"Error retrieving job logs: {str(e)}")
