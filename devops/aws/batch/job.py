#!/usr/bin/env python3
"""
AWS Batch Job Utilities

This module provides functions for interacting with AWS Batch jobs.
"""

import boto3
from botocore.config import Config
from datetime import datetime
import time
from tabulate import tabulate

def get_boto3_client(service_name='batch'):
    """Get a boto3 client with standard configuration."""
    config = Config(retries={'max_attempts': 10, 'mode': 'standard'}, max_pool_connections=50)
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

def list_jobs(job_queue=None, max_jobs=100):
    """List jobs in a job queue."""
    batch = get_boto3_client()

    if not job_queue:
        # Get all job queues
        try:
            response = batch.describe_job_queues()
            job_queues = [queue['jobQueueName'] for queue in response['jobQueues']]

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
        for status in ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING', 'SUCCEEDED', 'FAILED']:
            try:
                response = batch.list_jobs(
                    jobQueue=job_queue,
                    jobStatus=status,
                    maxResults=min(100, max_jobs)  # AWS API limit is 100
                )

                job_summaries = response.get('jobSummaryList', [])

                # Process in batches of 100 to avoid API limits
                if job_summaries:
                    job_ids = [job['jobId'] for job in job_summaries]

                    # Only call describe_jobs if we have job IDs to process
                    if job_ids:
                        try:
                            job_details = batch.describe_jobs(jobs=job_ids)['jobs']
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
    all_jobs.sort(key=lambda x: x.get('createdAt', 0), reverse=True)

    # Limit to max_jobs
    all_jobs = all_jobs[:max_jobs]

    # Format the output
    table_data = []
    for job in all_jobs:
        job_name = job['jobName']
        job_status = job['status']

        # Get timestamps
        created_at = job.get('createdAt', 0)
        started_at = job.get('startedAt', 0)
        stopped_at = job.get('stoppedAt', 0)

        # Format timestamps
        created_str = datetime.fromtimestamp(created_at / 1000).strftime('%Y-%m-%d %H:%M:%S') if created_at else 'N/A'
        created_ago = format_time_ago(created_at) if created_at else ''
        created_display = f"{created_str} {created_ago}"

        # Calculate duration
        if started_at and stopped_at:
            duration = format_time_difference(started_at, stopped_at)
        elif started_at:
            duration = format_time_difference(started_at)
        else:
            duration = 'N/A'

        # Get number of attempts
        attempts = len(job.get('attempts', []))

        # Get number of nodes
        num_nodes = 1  # Default for single-node jobs
        if 'nodeProperties' in job:
            num_nodes = job['nodeProperties'].get('numNodes', 1)

        # Calculate total GPUs
        num_gpus = 0
        container = job.get('container', {})
        if container:
            # Check if it's a GPU job
            resource_requirements = container.get('resourceRequirements', [])
            for resource in resource_requirements:
                if resource.get('type') == 'GPU':
                    num_gpus = int(resource.get('value', 0))
                    # For single-node jobs, multiply by number of nodes
                    if 'nodeProperties' not in job:
                        break

        # For multi-node jobs, calculate total GPUs across all nodes
        if 'nodeProperties' in job:
            # Reset GPU count for multi-node jobs to avoid double counting
            num_gpus = 0
            node_ranges = job['nodeProperties'].get('nodeRangeProperties', [])
            total_nodes = job['nodeProperties'].get('numNodes', 1)

            # If no node ranges specified but we have numNodes, use the main container's GPU count
            if not node_ranges and container:
                resource_requirements = container.get('resourceRequirements', [])
                for resource in resource_requirements:
                    if resource.get('type') == 'GPU':
                        num_gpus = int(resource.get('value', 0)) * total_nodes
                        break

            # Process each node range
            for node_range in node_ranges:
                node_container = node_range.get('container', {})
                if node_container:
                    # First check environment variables for NUM_GPUS
                    env_vars = node_container.get('environment', [])
                    gpus_per_node = 0
                    for env in env_vars:
                        if env.get('name') == 'NUM_GPUS':
                            try:
                                gpus_per_node = int(env.get('value', 0))
                                break
                            except (ValueError, TypeError):
                                pass

                    # If not found in environment, check resource requirements
                    if gpus_per_node == 0:
                        node_resources = node_container.get('resourceRequirements', [])
                        for resource in node_resources:
                            if resource.get('type') == 'GPU':
                                gpus_per_node = int(resource.get('value', 0))
                                break

                    # Get the target nodes range (e.g., "0:1" for nodes 0 and 1)
                    target_nodes = node_range.get('targetNodes', '')
                    try:
                        # Parse the range (e.g., "0:1" -> 2 nodes)
                        if ':' in target_nodes:
                            parts = target_nodes.split(':')
                            if len(parts) == 2:
                                start = int(parts[0])
                                # If end is empty (e.g., "0:"), use total_nodes
                                if parts[1] == '':
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
        print(f"Jobs in queue '{job_queue}':")
        headers = ['Name', 'Status', 'Created', 'Duration', 'Attempts', 'NumNodes', 'Num GPUs']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    else:
        print(f"No jobs found in queue '{job_queue}'")

    return all_jobs

def get_job_info(job_id_or_name):
    """Get detailed information about a specific job by ID or name."""
    batch = get_boto3_client()

    try:
        # First try to get the job by ID
        response = batch.describe_jobs(jobs=[job_id_or_name])

        # If no job found by ID, try to find by name
        if not response['jobs']:
            # We need to list jobs from all queues to find by name
            job = None

            # Get all job queues
            try:
                queues_response = batch.describe_job_queues()
                job_queues = [queue['jobQueueName'] for queue in queues_response['jobQueues']]

                # Search for the job in each queue
                for queue in job_queues:
                    # Check all job statuses
                    for status in ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING', 'SUCCEEDED', 'FAILED']:
                        try:
                            jobs_response = batch.list_jobs(
                                jobQueue=queue,
                                jobStatus=status,
                                maxResults=100
                            )

                            # Look for a job with the specified name
                            for job_summary in jobs_response.get('jobSummaryList', []):
                                if job_summary['jobName'] == job_id_or_name:
                                    # Found a job with the specified name, get its details
                                    job_details_response = batch.describe_jobs(jobs=[job_summary['jobId']])
                                    if job_details_response['jobs']:
                                        job = job_details_response['jobs'][0]
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
            job = response['jobs'][0]

        # Print basic information
        print(f"\nJob: {job['jobId']}")
        print(f"Name: {job['jobName']}")
        print(f"Status: {job['status']}")
        print(f"Status Reason: {job.get('statusReason', 'N/A')}")

        # Print timestamps
        created_at = job.get('createdAt', 0)
        started_at = job.get('startedAt', 0)
        stopped_at = job.get('stoppedAt', 0)

        if created_at:
            created_str = datetime.fromtimestamp(created_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Created: {created_str}")

        if started_at:
            started_str = datetime.fromtimestamp(started_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Started: {started_str}")

        if stopped_at:
            stopped_str = datetime.fromtimestamp(stopped_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Stopped: {stopped_str}")

        # Calculate duration
        if started_at and stopped_at:
            duration = format_time_difference(started_at, stopped_at)
            print(f"Duration: {duration}")
        elif started_at:
            duration = format_time_difference(started_at)
            print(f"Running for: {duration}")

        # Print job definition
        job_definition = job.get('jobDefinition', '').split('/')[-1]
        print(f"Job Definition: {job_definition}")

        # Print job queue
        job_queue = job.get('jobQueue', '').split('/')[-1]
        print(f"Job Queue: {job_queue}")

        # Print container details
        if 'container' in job:
            container = job['container']
            print("\nContainer:")
            print(f"  Image: {container.get('image', 'N/A')}")
            print(f"  vCPUs: {container.get('vcpus', 'N/A')}")
            print(f"  Memory: {container.get('memory', 'N/A')} MiB")

            # Print command
            if 'command' in container:
                command_str = ' '.join(container['command'])
                print(f"  Command: {command_str}")

            # Print environment variables
            if 'environment' in container:
                print("\n  Environment Variables:")
                for env in container['environment']:
                    print(f"    {env['name']}: {env['value']}")

            # Print exit code
            if 'exitCode' in container:
                print(f"  Exit Code: {container['exitCode']}")

            # Print reason
            if 'reason' in container:
                print(f"  Reason: {container['reason']}")

        # Print attempts
        attempts = job.get('attempts', [])
        if attempts:
            print("\nAttempts:")
            for i, attempt in enumerate(attempts):
                print(f"  Attempt {i}:")
                print(f"    Status: {attempt.get('status', 'N/A')}")
                print(f"    Reason: {attempt.get('statusReason', 'N/A')}")

                # Print container details
                container = attempt.get('container', {})
                if container:
                    if 'exitCode' in container:
                        print(f"    Exit Code: {container['exitCode']}")
                    if 'reason' in container:
                        print(f"    Reason: {container['reason']}")
                    if 'logStreamName' in container:
                        print(f"    Log Stream: {container['logStreamName']}")

        # Print node details for multi-node jobs
        if 'nodeProperties' in job:
            node_props = job['nodeProperties']
            print("\nNode Properties:")
            print(f"  Number of Nodes: {node_props.get('numNodes', 'N/A')}")
            print(f"  Main Node: {node_props.get('mainNode', 'N/A')}")

            # Print node ranges
            if 'nodeRangeProperties' in node_props:
                print("\n  Node Ranges:")
                for i, node_range in enumerate(node_props['nodeRangeProperties']):
                    print(f"    Range {i}:")
                    print(f"      Target Nodes: {node_range.get('targetNodes', 'N/A')}")

                    # Print container details
                    container = node_range.get('container', {})
                    if container:
                        print(f"      Image: {container.get('image', 'N/A')}")
                        print(f"      vCPUs: {container.get('vcpus', 'N/A')}")
                        print(f"      Memory: {container.get('memory', 'N/A')} MiB")

                        # Print command
                        if 'command' in container:
                            command_str = ' '.join(container['command'])
                            print(f"      Command: {command_str}")

        # Print dependencies
        dependencies = job.get('dependencies', [])
        if dependencies:
            print("\nDependencies:")
            for dep in dependencies:
                print(f"  {dep.get('jobId', 'N/A')}: {dep.get('type', 'N/A')}")

        # Print tags
        tags = job.get('tags', {})
        if tags:
            print("\nTags:")
            for key, value in tags.items():
                print(f"  {key}: {value}")

        return job
    except Exception as e:
        print(f"Error retrieving job information: {str(e)}")
        return None

def stop_job(job_id_or_name, reason="Stopped by user"):
    """Stop a running job by ID or name."""
    batch = get_boto3_client()

    try:
        # First try to get the job by ID
        response = batch.describe_jobs(jobs=[job_id_or_name])

        # If no job found by ID, try to find by name
        if not response['jobs']:
            # We need to list jobs from all queues to find by name
            job_id = None

            # Get all job queues
            try:
                queues_response = batch.describe_job_queues()
                job_queues = [queue['jobQueueName'] for queue in queues_response['jobQueues']]

                # Search for the job in each queue
                for queue in job_queues:
                    # Check all job statuses
                    for status in ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING']:
                        try:
                            jobs_response = batch.list_jobs(
                                jobQueue=queue,
                                jobStatus=status,
                                maxResults=100
                            )

                            # Look for a job with the specified name
                            for job_summary in jobs_response.get('jobSummaryList', []):
                                if job_summary['jobName'] == job_id_or_name:
                                    # Found a job with the specified name
                                    job_id = job_summary['jobId']
                                    job_details_response = batch.describe_jobs(jobs=[job_id])
                                    if job_details_response['jobs']:
                                        job = job_details_response['jobs'][0]
                                        break

                            if job_id:
                                break
                        except Exception:
                            continue

                    if job_id:
                        break

                if not job_id:
                    print(f"No job found with ID or name '{job_id_or_name}'")
                    return False
            except Exception as e:
                print(f"Error retrieving job queues: {str(e)}")
                print(f"Job '{job_id_or_name}' not found")
                return False
        else:
            job = response['jobs'][0]
            job_id = job['jobId']

        # Check if the job is in a stoppable state
        stoppable_states = ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING']
        if job['status'] not in stoppable_states:
            print(f"Job '{job_id}' is in state '{job['status']}' and cannot be stopped")
            return False

        # Stop the job
        batch.terminate_job(
            jobId=job_id,
            reason=reason
        )

        print(f"Job '{job_id}' has been stopped")
        return True
    except Exception as e:
        print(f"Error stopping job: {str(e)}")
        return False

def launch_job(job_queue=None):
    """Launch a new job."""
    if not job_queue:
        print("Error: Job queue is required for launching a job")
        return False

    print("Job launch functionality is not implemented yet")
    print("Please use the AWS Management Console or AWS CLI to launch jobs")
    return False
