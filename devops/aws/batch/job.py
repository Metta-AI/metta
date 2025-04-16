#!/usr/bin/env python3
"""
AWS Batch Job Utilities

This module provides functions for interacting with AWS Batch jobs.
"""

import subprocess
import time
from datetime import datetime

import boto3
from botocore.config import Config
from colorama import Fore, Style, init
from tabulate import tabulate

# Initialize colorama
init(autoreset=True)


def get_boto3_client(service_name="batch"):
    """Get a boto3 client with standard configuration."""
    config = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=50)
    return boto3.client(service_name, config=config)


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


def format_time_ago(timestamp):
    """Format a timestamp as a human-readable 'time ago' string."""
    if not timestamp:
        return "N/A"

    # Convert milliseconds to seconds if necessary
    if timestamp > 1000000000000:  # If timestamp is in milliseconds
        timestamp = timestamp / 1000

    # Calculate difference from now
    now = time.time()
    diff_seconds = now - timestamp

    # Format the difference
    if diff_seconds < 0:
        return "Future"

    if diff_seconds < 60:
        return f"({int(diff_seconds)}s ago)"

    if diff_seconds < 3600:
        minutes = int(diff_seconds / 60)
        return f"({minutes}m ago)"

    if diff_seconds < 86400:
        hours = int(diff_seconds / 3600)
        return f"({hours}h ago)"

    days = int(diff_seconds / 86400)
    return f"({days}d ago)"


def list_jobs(job_queue=None, max_jobs=100, no_color=False):
    """List jobs in a job queue."""
    # If no_color is True, disable colorama
    if no_color:
        global Fore, Style
        # Save the original values
        orig_fore, orig_style = Fore, Style

        # Create dummy objects that return empty strings
        class DummyFore:
            def __getattr__(self, _):
                return ""

        class DummyStyle:
            def __getattr__(self, _):
                return ""

        Fore, Style = DummyFore(), DummyStyle()

    batch = get_boto3_client()

    if not job_queue:
        # Get all job queues
        try:
            response = batch.describe_job_queues()
            job_queues = [queue["jobQueueName"] for queue in response["jobQueues"]]

            if not job_queues:
                print("No job queues found")
                return []

            print(f"Available job queues: {', '.join(job_queues)}")
            print("Please specify a job queue with --queue")
            print("Or use 'jobs' command to list jobs in the default queue (metta-jq)")
            return []
        except Exception as e:
            print(f"Error retrieving job queues: {str(e)}")
            return []

    all_jobs = []

    try:
        # Get jobs from the specified queue for each status
        for status in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED"]:
            try:
                response = batch.list_jobs(
                    jobQueue=job_queue,
                    jobStatus=status,
                    maxResults=min(100, max_jobs),  # AWS API limit is 100
                )

                job_summaries = response.get("jobSummaryList", [])

                # Process in batches of 100 to avoid API limits
                if job_summaries:
                    job_ids = [job["jobId"] for job in job_summaries]

                    # Only call describe_jobs if we have job IDs to process
                    if job_ids:
                        try:
                            job_details = batch.describe_jobs(jobs=job_ids)["jobs"]
                            all_jobs.extend(job_details)
                        except Exception as e:
                            print(f"Error retrieving job details: {str(e)}")
            except Exception as e:
                if "ArrayJob, Multi-node Job and job status are not applicable" not in str(e):
                    print(f"Error retrieving jobs from queue {job_queue} with status {status}: {str(e)}")
                continue
    except Exception as e:
        print(f"Error retrieving jobs: {str(e)}")

    # Sort jobs by creation time (newest first)
    all_jobs.sort(key=lambda x: x.get("createdAt", 0), reverse=True)

    # Limit to max_jobs
    all_jobs = all_jobs[:max_jobs]

    # Format the output
    table_data = []
    for job in all_jobs:
        job_name = job["jobName"]
        job_status = job["status"]

        # Get timestamps
        created_at = job.get("createdAt", 0)
        started_at = job.get("startedAt", 0)
        stopped_at = job.get("stoppedAt", 0)

        # Format timestamps
        created_str = datetime.fromtimestamp(created_at / 1000).strftime("%Y-%m-%d %H:%M:%S") if created_at else "N/A"
        created_ago = format_time_ago(created_at) if created_at else ""
        created_display = f"{created_str} {created_ago}"

        # Calculate duration
        if started_at and stopped_at:
            duration = format_time_difference(started_at, stopped_at)
        elif started_at:
            duration = format_time_difference(started_at)
        else:
            duration = "N/A"

        # Get number of attempts
        attempts = len(job.get("attempts", []))

        # Get number of nodes
        num_nodes = 1  # Default for single-node jobs
        if "nodeProperties" in job:
            num_nodes = job["nodeProperties"].get("numNodes", 1)

        # Calculate total GPUs
        num_gpus = 0
        container = job.get("container", {})
        if container:
            # Check if it's a GPU job
            resource_requirements = container.get("resourceRequirements", [])
            for resource in resource_requirements:
                if resource.get("type") == "GPU":
                    num_gpus = int(resource.get("value", 0))
                    # For single-node jobs, multiply by number of nodes
                    if "nodeProperties" not in job:
                        break

        # For multi-node jobs, calculate total GPUs across all nodes
        if "nodeProperties" in job:
            # Reset GPU count for multi-node jobs to avoid double counting
            num_gpus = 0
            node_ranges = job["nodeProperties"].get("nodeRangeProperties", [])
            total_nodes = job["nodeProperties"].get("numNodes", 1)

            # If no node ranges specified but we have numNodes, use the main container's GPU count
            if not node_ranges and container:
                resource_requirements = container.get("resourceRequirements", [])
                for resource in resource_requirements:
                    if resource.get("type") == "GPU":
                        num_gpus = int(resource.get("value", 0)) * total_nodes
                        break

            # Process each node range
            for node_range in node_ranges:
                node_container = node_range.get("container", {})
                if node_container:
                    # First check environment variables for NUM_GPUS
                    env_vars = node_container.get("environment", [])
                    gpus_per_node = 0
                    for env in env_vars:
                        if env.get("name") == "NUM_GPUS":
                            try:
                                gpus_per_node = int(env.get("value", 0))
                                break
                            except (ValueError, TypeError):
                                pass

                    # If not found in environment, check resource requirements
                    if gpus_per_node == 0:
                        node_resources = node_container.get("resourceRequirements", [])
                        for resource in node_resources:
                            if resource.get("type") == "GPU":
                                gpus_per_node = int(resource.get("value", 0))
                                break

                    # Get the target nodes range (e.g., "0:1" for nodes 0 and 1)
                    target_nodes = node_range.get("targetNodes", "")
                    try:
                        # Parse the range (e.g., "0:1" -> 2 nodes)
                        if ":" in target_nodes:
                            parts = target_nodes.split(":")
                            if len(parts) == 2:
                                start = int(parts[0])
                                # If end is empty (e.g., "0:"), use total_nodes
                                if parts[1] == "":
                                    node_count = total_nodes - start
                                else:
                                    end = int(parts[1])
                                    node_count = end - start + 1
                            else:
                                node_count = 1
                        else:
                            # Single node specified
                            node_count = 1
                        num_gpus += gpus_per_node * node_count
                    except (ValueError, TypeError):
                        # If we can't parse the range, assume 0 GPUs for this range
                        pass

        table_data.append([job_name, job_status, created_display, duration, attempts, num_nodes, num_gpus])

    # Print the table
    if table_data:
        # Add color to job status and other fields
        colored_table_data = []
        for row in table_data:
            job_name, status, created, duration, attempts, num_nodes, num_gpus = row

            # Color for job status
            if status == "RUNNING":
                colored_status = f"{Fore.GREEN}{status}{Style.RESET_ALL}"
            elif status == "SUCCEEDED":
                colored_status = f"{Fore.BLUE}{status}{Style.RESET_ALL}"
            elif status == "FAILED":
                colored_status = f"{Fore.RED}{status}{Style.RESET_ALL}"
            elif status in ["SUBMITTED", "PENDING", "RUNNABLE"]:
                colored_status = f"{Fore.YELLOW}{status}{Style.RESET_ALL}"
            elif status == "STARTING":
                colored_status = f"{Fore.CYAN}{status}{Style.RESET_ALL}"
            else:
                colored_status = status

            # Color for job name
            colored_job_name = f"{Fore.MAGENTA}{job_name}{Style.RESET_ALL}"

            # Color for duration
            if duration != "N/A":
                colored_duration = f"{Fore.CYAN}{duration}{Style.RESET_ALL}"
            else:
                colored_duration = duration

            # Color for GPU count
            if num_gpus > 0:
                colored_num_gpus = f"{Fore.YELLOW}{num_gpus}{Style.RESET_ALL}"
            else:
                colored_num_gpus = num_gpus

            colored_table_data.append(
                [colored_job_name, colored_status, created, colored_duration, attempts, num_nodes, colored_num_gpus]
            )

        print(f"Jobs in queue '{Fore.CYAN}{job_queue}{Style.RESET_ALL}':")
        headers = ["Name", "Status", "Created", "Duration", "Attempts", "NumNodes", "Num GPUs"]
        print(tabulate(colored_table_data, headers=headers, tablefmt="grid"))
    else:
        print(f"No jobs found in queue '{Fore.CYAN}{job_queue}{Style.RESET_ALL}'")

    # Restore colorama if it was disabled
    if no_color:
        Fore, Style = orig_fore, orig_style

    return all_jobs


def get_job_info(job_id_or_name, no_color=False):
    """Get detailed information about a specific job by ID or name."""
    # If no_color is True, disable colorama
    if no_color:
        global Fore, Style
        # Save the original values
        orig_fore, orig_style = Fore, Style

        # Create dummy objects that return empty strings
        class DummyFore:
            def __getattr__(self, _):
                return ""

        class DummyStyle:
            def __getattr__(self, _):
                return ""

        Fore, Style = DummyFore(), DummyStyle()

    batch = get_boto3_client()

    try:
        # First try to get the job by ID
        response = batch.describe_jobs(jobs=[job_id_or_name])

        # If no job found by ID, try to find by name
        if not response["jobs"]:
            # We need to list jobs from all queues to find by name
            job = None

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
                                    # Found a job with the specified name, get its details
                                    job_details_response = batch.describe_jobs(jobs=[job_summary["jobId"]])
                                    if job_details_response["jobs"]:
                                        job = job_details_response["jobs"][0]
                                        break

                            if job:
                                break
                        except Exception:
                            continue

                    if job:
                        break

                if not job:
                    print(f"No job found with ID or name '{job_id_or_name}'")
                    return None
            except Exception as e:
                print(f"Error retrieving job queues: {str(e)}")
                print(f"Job '{job_id_or_name}' not found")
                return None
        else:
            job = response["jobs"][0]

        # Print basic information
        print(f"\nJob: {Fore.CYAN}{job['jobId']}{Style.RESET_ALL}")
        print(f"Name: {Fore.MAGENTA}{job['jobName']}{Style.RESET_ALL}")

        # Color for job status
        status = job["status"]
        if status == "RUNNING":
            status_color = f"{Fore.GREEN}{status}{Style.RESET_ALL}"
        elif status == "SUCCEEDED":
            status_color = f"{Fore.BLUE}{status}{Style.RESET_ALL}"
        elif status == "FAILED":
            status_color = f"{Fore.RED}{status}{Style.RESET_ALL}"
        elif status in ["SUBMITTED", "PENDING", "RUNNABLE"]:
            status_color = f"{Fore.YELLOW}{status}{Style.RESET_ALL}"
        elif status == "STARTING":
            status_color = f"{Fore.CYAN}{status}{Style.RESET_ALL}"
        else:
            status_color = status

        print(f"Status: {status_color}")

        status_reason = job.get("statusReason", "N/A")
        if status_reason != "N/A":
            print(f"Status Reason: {Fore.RED}{status_reason}{Style.RESET_ALL}")
        else:
            print(f"Status Reason: {status_reason}")

        # Print timestamps
        created_at = job.get("createdAt", 0)
        started_at = job.get("startedAt", 0)
        stopped_at = job.get("stoppedAt", 0)

        if created_at:
            created_str = datetime.fromtimestamp(created_at / 1000).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Created: {Fore.YELLOW}{created_str}{Style.RESET_ALL}")

        if started_at:
            started_str = datetime.fromtimestamp(started_at / 1000).strftime("%Y-%m-%d %H:%M:%S")
            print(f"Started: {Fore.GREEN}{started_str}{Style.RESET_ALL}")

        if stopped_at:
            stopped_str = datetime.fromtimestamp(stopped_at / 1000).strftime("%Y-%m-%d %H:%M:%S")
            if status == "FAILED":
                print(f"Stopped: {Fore.RED}{stopped_str}{Style.RESET_ALL}")
            else:
                print(f"Stopped: {Fore.BLUE}{stopped_str}{Style.RESET_ALL}")

        # Calculate duration
        if started_at and stopped_at:
            duration = format_time_difference(started_at, stopped_at)
            print(f"Duration: {Fore.CYAN}{duration}{Style.RESET_ALL}")
        elif started_at:
            duration = format_time_difference(started_at)
            print(f"Running for: {Fore.CYAN}{duration}{Style.RESET_ALL}")

        # Print job definition
        job_definition = job.get("jobDefinition", "").split("/")[-1]
        print(f"Job Definition: {Fore.BLUE}{job_definition}{Style.RESET_ALL}")

        # Print job queue
        job_queue = job.get("jobQueue", "").split("/")[-1]
        print(f"Job Queue: {Fore.BLUE}{job_queue}{Style.RESET_ALL}")

        # Print container details
        if "container" in job:
            container = job["container"]
            print(f"\n{Fore.YELLOW}Container:{Style.RESET_ALL}")
            print(f"  Image: {Fore.CYAN}{container.get('image', 'N/A')}{Style.RESET_ALL}")
            print(f"  vCPUs: {Fore.GREEN}{container.get('vcpus', 'N/A')}{Style.RESET_ALL}")
            print(f"  Memory: {Fore.GREEN}{container.get('memory', 'N/A')} MiB{Style.RESET_ALL}")

            # Print command
            if "command" in container:
                command_str = " ".join(container["command"])
                print(f"  Command: {Fore.MAGENTA}{command_str}{Style.RESET_ALL}")

            # Print environment variables
            if "environment" in container:
                print(f"\n  {Fore.YELLOW}Environment Variables:{Style.RESET_ALL}")
                for env in container["environment"]:
                    print(f"    {Fore.CYAN}{env['name']}{Style.RESET_ALL}: {env['value']}")

            # Print exit code
            if "exitCode" in container:
                exit_code = container["exitCode"]
                if exit_code == 0:
                    print(f"  Exit Code: {Fore.GREEN}{exit_code}{Style.RESET_ALL}")
                else:
                    print(f"  Exit Code: {Fore.RED}{exit_code}{Style.RESET_ALL}")

            # Print reason
            if "reason" in container:
                print(f"  Reason: {Fore.RED}{container['reason']}{Style.RESET_ALL}")

        # Print attempts
        attempts = job.get("attempts", [])
        if attempts:
            print(f"\n{Fore.YELLOW}Attempts:{Style.RESET_ALL}")
            for i, attempt in enumerate(attempts):
                print(f"  {Fore.CYAN}Attempt {i}:{Style.RESET_ALL}")

                # Color for attempt status
                status = attempt.get("status", "N/A")
                if status == "RUNNING":
                    status_color = f"{Fore.GREEN}{status}{Style.RESET_ALL}"
                elif status == "SUCCEEDED":
                    status_color = f"{Fore.BLUE}{status}{Style.RESET_ALL}"
                elif status == "FAILED":
                    status_color = f"{Fore.RED}{status}{Style.RESET_ALL}"
                elif status in ["SUBMITTED", "PENDING", "RUNNABLE"]:
                    status_color = f"{Fore.YELLOW}{status}{Style.RESET_ALL}"
                elif status == "STARTING":
                    status_color = f"{Fore.CYAN}{status}{Style.RESET_ALL}"
                else:
                    status_color = status

                print(f"    Status: {status_color}")

                reason = attempt.get("statusReason", "N/A")
                if reason != "N/A":
                    print(f"    Reason: {Fore.RED}{reason}{Style.RESET_ALL}")
                else:
                    print(f"    Reason: {reason}")

                # Print container details
                container = attempt.get("container", {})
                if container:
                    if "exitCode" in container:
                        exit_code = container["exitCode"]
                        if exit_code == 0:
                            print(f"    Exit Code: {Fore.GREEN}{exit_code}{Style.RESET_ALL}")
                        else:
                            print(f"    Exit Code: {Fore.RED}{exit_code}{Style.RESET_ALL}")
                    if "reason" in container:
                        print(f"    Reason: {Fore.RED}{container['reason']}{Style.RESET_ALL}")
                    if "logStreamName" in container:
                        print(f"    Log Stream: {Fore.BLUE}{container['logStreamName']}{Style.RESET_ALL}")

        # Print node details for multi-node jobs
        if "nodeProperties" in job:
            node_props = job["nodeProperties"]
            print(f"\n{Fore.YELLOW}Node Properties:{Style.RESET_ALL}")
            print(f"  Number of Nodes: {Fore.GREEN}{node_props.get('numNodes', 'N/A')}{Style.RESET_ALL}")
            print(f"  Main Node: {Fore.CYAN}{node_props.get('mainNode', 'N/A')}{Style.RESET_ALL}")

            # Print node ranges
            if "nodeRangeProperties" in node_props:
                print(f"\n  {Fore.YELLOW}Node Ranges:{Style.RESET_ALL}")
                for i, node_range in enumerate(node_props["nodeRangeProperties"]):
                    print(f"    {Fore.CYAN}Range {i}:{Style.RESET_ALL}")
                    print(f"      Target Nodes: {Fore.MAGENTA}{node_range.get('targetNodes', 'N/A')}{Style.RESET_ALL}")

                    # Print container details
                    container = node_range.get("container", {})
                    if container:
                        print(f"      Image: {Fore.BLUE}{container.get('image', 'N/A')}{Style.RESET_ALL}")
                        print(f"      vCPUs: {Fore.GREEN}{container.get('vcpus', 'N/A')}{Style.RESET_ALL}")
                        print(f"      Memory: {Fore.GREEN}{container.get('memory', 'N/A')} MiB{Style.RESET_ALL}")

                        # Print command
                        if "command" in container:
                            command_str = " ".join(container["command"])
                            print(f"      Command: {Fore.MAGENTA}{command_str}{Style.RESET_ALL}")

        # Print dependencies
        dependencies = job.get("dependencies", [])
        if dependencies:
            print(f"\n{Fore.YELLOW}Dependencies:{Style.RESET_ALL}")
            for dep in dependencies:
                print(f"  {Fore.CYAN}{dep.get('jobId', 'N/A')}{Style.RESET_ALL}: {dep.get('type', 'N/A')}")

        # Print tags
        tags = job.get("tags", {})
        if tags:
            print(f"\n{Fore.YELLOW}Tags:{Style.RESET_ALL}")
            for key, value in tags.items():
                print(f"  {Fore.CYAN}{key}{Style.RESET_ALL}: {value}")

        # Restore colorama if it was disabled
        if no_color:
            Fore, Style = orig_fore, orig_style

        return job
    except Exception as e:
        print(f"Error retrieving job information: {str(e)}")
        return None


def find_jobs_by_prefix(prefix, max_results=5):
    """Find jobs that start with the given prefix."""
    batch = get_boto3_client()
    matching_jobs = []

    try:
        # Get all job queues
        queues_response = batch.describe_job_queues()
        job_queues = [queue["jobQueueName"] for queue in queues_response["jobQueues"]]

        # Search for jobs in each queue
        for queue in job_queues:
            # Check all job statuses
            for status in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]:
                try:
                    jobs_response = batch.list_jobs(jobQueue=queue, jobStatus=status, maxResults=100)

                    # Look for jobs with the specified prefix
                    for job_summary in jobs_response.get("jobSummaryList", []):
                        if job_summary["jobName"].startswith(prefix) or job_summary["jobId"].startswith(prefix):
                            # Get full job details
                            job_details_response = batch.describe_jobs(jobs=[job_summary["jobId"]])
                            if job_details_response["jobs"]:
                                matching_jobs.append(job_details_response["jobs"][0])

                                # Limit the number of results
                                if len(matching_jobs) >= max_results:
                                    return matching_jobs
                except Exception:
                    continue

        return matching_jobs
    except Exception as e:
        print(f"Error finding jobs by prefix: {str(e)}")
        return []


def stop_job(job_id_or_name, reason="Stopped by user", max_results=5):
    """Stop a running job by ID or name."""
    batch = get_boto3_client()

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
                    for status in ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]:
                        try:
                            jobs_response = batch.list_jobs(jobQueue=queue, jobStatus=status, maxResults=100)

                            # Look for a job with the specified name
                            for job_summary in jobs_response.get("jobSummaryList", []):
                                if job_summary["jobName"] == job_id_or_name:
                                    # Found a job with the specified name
                                    job_id = job_summary["jobId"]
                                    job_details_response = batch.describe_jobs(jobs=[job_id])
                                    if job_details_response["jobs"]:
                                        job = job_details_response["jobs"][0]
                                        break

                            if job_id:
                                break
                        except Exception:
                            continue

                    if job_id:
                        break

                if not job_id:
                    # If no exact match found, try to find jobs with the prefix
                    matching_jobs = find_jobs_by_prefix(job_id_or_name, max_results=max_results)
                    if matching_jobs:
                        return matching_jobs
                    else:
                        print(f"No job found with ID or name '{job_id_or_name}'")
                        return False
            except Exception as e:
                print(f"Error retrieving job queues: {str(e)}")
                print(f"Job '{job_id_or_name}' not found")
                return False
        else:
            job = response["jobs"][0]
            job_id = job["jobId"]

        # Check if the job is in a stoppable state
        stoppable_states = ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]
        if job["status"] not in stoppable_states:
            print(f"Job '{job_id}' is in state '{job['status']}' and cannot be stopped")
            return False

        # Stop the job
        batch.terminate_job(jobId=job_id, reason=reason)

        print(f"Job '{job_id}' has been stopped")
        return True
    except Exception as e:
        print(f"Error stopping job: {str(e)}")
        return False


def stop_jobs(job_ids, reason="Stopped by user"):
    """Stop multiple jobs by their IDs."""
    batch = get_boto3_client()
    success_count = 0

    for job_id in job_ids:
        try:
            # Check if the job is in a stoppable state
            response = batch.describe_jobs(jobs=[job_id])
            if not response["jobs"]:
                print(f"Job '{job_id}' not found")
                continue

            job = response["jobs"][0]
            stoppable_states = ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"]
            if job["status"] not in stoppable_states:
                print(f"Job '{job_id}' is in state '{job['status']}' and cannot be stopped")
                continue

            # Stop the job
            batch.terminate_job(jobId=job_id, reason=reason)

            print(f"Job '{job_id}' has been stopped")
            success_count += 1
        except Exception as e:
            print(f"Error stopping job '{job_id}': {str(e)}")

    return success_count > 0


def launch_job(job_queue=None):
    """Launch a new job."""
    print("The launch_job function is deprecated.")
    print("Please use the launch_cmd.py script directly:")
    print("  ../cmd.sh launch --run RUN_ID --cmd COMMAND [options]")
    return False


def get_job_ip(job_id_or_name):
    """Get the public IP address of the instance running a job."""
    batch = get_boto3_client("batch")
    ecs = get_boto3_client("ecs")
    ec2 = get_boto3_client("ec2")

    try:
        # First try to get the job by ID
        response = batch.describe_jobs(jobs=[job_id_or_name])

        # If no job found by ID, try to find by name
        if not response["jobs"]:
            # We need to list jobs from all queues to find by name
            job = None

            # Get all job queues
            queues_response = batch.describe_job_queues()
            job_queues = [queue["jobQueueName"] for queue in queues_response["jobQueues"]]

            # Search for the job in each queue
            for queue in job_queues:
                # Check all job statuses
                for status in ["RUNNING"]:  # Only look for running jobs
                    try:
                        jobs_response = batch.list_jobs(jobQueue=queue, jobStatus=status, maxResults=100)

                        # Look for a job with the specified name
                        for job_summary in jobs_response.get("jobSummaryList", []):
                            if job_summary["jobName"] == job_id_or_name:
                                # Found a job with the specified name, get its details
                                job_details_response = batch.describe_jobs(jobs=[job_summary["jobId"]])
                                if job_details_response["jobs"]:
                                    job = job_details_response["jobs"][0]
                                    break

                        if job:
                            break
                    except Exception:
                        continue

                if job:
                    break

            if not job:
                print(f"No running job found with ID or name '{job_id_or_name}'")
                return None
        else:
            job = response["jobs"][0]

        # Check if the job is running
        if job["status"] != "RUNNING":
            print(f"Job '{job['jobId']}' is in state '{job['status']}' and not running")
            return None

        # Check if it's a multi-node job
        if "nodeProperties" in job:
            print(f"Error: Job '{job['jobId']}' is a multi-node job. SSH is not supported for multi-node jobs.")
            return None

        # Get the task ARN and cluster
        container = job["container"]
        task_arn = container.get("taskArn")
        cluster_arn = container.get("containerInstanceArn")

        if not task_arn or not cluster_arn:
            print(f"Job '{job['jobId']}' does not have task or container instance information")
            return None

        # Extract the cluster name from the cluster ARN
        cluster_name = cluster_arn.split("/")[1]

        # Get the container instance ARN
        task_desc = ecs.describe_tasks(cluster=cluster_name, tasks=[task_arn])
        if not task_desc["tasks"]:
            print(f"No task found for job '{job['jobId']}'")
            return None

        container_instance_arn = task_desc["tasks"][0]["containerInstanceArn"]

        # Get the EC2 instance ID
        container_instance_desc = ecs.describe_container_instances(
            cluster=cluster_name, containerInstances=[container_instance_arn]
        )
        ec2_instance_id = container_instance_desc["containerInstances"][0]["ec2InstanceId"]

        # Get the public IP address
        instances = ec2.describe_instances(InstanceIds=[ec2_instance_id])
        if "PublicIpAddress" in instances["Reservations"][0]["Instances"][0]:
            public_ip = instances["Reservations"][0]["Instances"][0]["PublicIpAddress"]
            return public_ip
        else:
            print(f"No public IP address found for job '{job['jobId']}'")
            return None

    except Exception as e:
        print(f"Error retrieving job IP: {str(e)}")
        return None


def ssh_to_job(job_id_or_name, instance_only=False):
    """Connect to the instance running a job via SSH.

    Args:
        job_id_or_name: The job ID or name to connect to
        instance_only: If True, connect directly to the instance without attempting to connect to the container
    """

    # Get the IP address of the job
    ip = get_job_ip(job_id_or_name)
    if not ip:
        return False

    try:
        # Establish SSH connection and check if it's successful
        print(f"Checking SSH connection to {ip}...")
        ssh_check_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {ip} 'echo Connected'"
        ssh_check_output = subprocess.check_output(ssh_check_cmd, shell=True).decode().strip()
        if ssh_check_output != "Connected":
            raise subprocess.CalledProcessError(1, "SSH connection check failed")

        if instance_only:
            # Connect directly to the instance
            print(f"Connecting directly to the instance at {ip}...")
            ssh_cmd = f"ssh -o StrictHostKeyChecking=no -t {ip}"
            subprocess.run(ssh_cmd, shell=True)
        else:
            # Retrieve container ID
            print(f"Finding container on {ip}...")
            container_cmd = f"ssh -o StrictHostKeyChecking=no -t {ip} \"docker ps | grep 'mettaai/metta'\""
            container_id_output = subprocess.check_output(container_cmd, shell=True).decode().strip()

            if container_id_output:
                container_id = container_id_output.split()[0]
                print(f"Connecting to container {container_id} on {ip}...")
                exec_cmd = f'ssh -o StrictHostKeyChecking=no -t {ip} "docker exec -it {container_id} /bin/bash"'
                subprocess.run(exec_cmd, shell=True)
            else:
                print(f"No container running the 'mettaai/metta' image found on the instance {ip}.")
                print("Connecting to the instance directly...")
                ssh_cmd = f"ssh -o StrictHostKeyChecking=no -t {ip}"
                subprocess.run(ssh_cmd, shell=True)

        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {str(e)}")
        if "Connection timed out" in str(e):
            print(f"SSH connection to {ip} timed out. Please check the instance status and network connectivity.")
        elif "Connection refused" in str(e):
            print(
                f"SSH connection to {ip} was refused. Please check if the instance is running and accepts SSH "
                "connections."
            )
        else:
            print(
                f"An error occurred while connecting to {ip}. Please check the instance status and SSH configuration."
            )
        return False
