#!/usr/bin/env python3
import argparse
import boto3
import sys
import time
import os
from datetime import datetime, timedelta
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
import netrc

# Add the project root to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from metta.devops.aws.cluster_info import get_batch_job_queues, get_batch_jobs

# Configure boto3 to use a higher max_pool_connections
config = Config(
    retries={'max_attempts': 10, 'mode': 'standard'},
    max_pool_connections=50
)

def get_batch_job_queues():
    """Get a list of all available AWS Batch job queues."""
    config = Config(retries={'max_attempts': 10, 'mode': 'standard'})
    batch = boto3.client('batch', config=config)

    try:
        response = batch.describe_job_queues()
        return [queue['jobQueueName'] for queue in response['jobQueues']]
    except Exception as e:
        print(f"Error retrieving job queues: {str(e)}")
        return []

def get_job_details(job_id=None, job_prefix=None, job_queue=None, max_jobs=100):
    """Get details for a specific job or jobs matching a prefix."""
    config = Config(retries={'max_attempts': 10, 'mode': 'standard'})
    batch = boto3.client('batch', config=config)

    # If a specific job ID is provided, try to get it directly first
    if job_id:
        try:
            response = batch.describe_jobs(jobs=[job_id])
            if response['jobs']:
                return response['jobs']
        except Exception as e:
            print(f"Error retrieving job with ID '{job_id}': {str(e)}")

    all_jobs = []

    # Determine which job queues to search
    job_queues = []
    if job_queue:
        job_queues.append(job_queue)
    else:
        job_queues = get_batch_job_queues()

    if not job_queues:
        print("No job queues found. Please check your AWS configuration.")
        return all_jobs

    # Get jobs from each queue
    for queue in job_queues:
        try:
            # Get all jobs from the queue without filtering by status
            for status in ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING', 'SUCCEEDED', 'FAILED']:
                try:
                    response = batch.list_jobs(
                        jobQueue=queue,
                        jobStatus=status,
                        maxResults=min(max_jobs, 100)  # AWS API limit is 100
                    )

                    job_summaries = response.get('jobSummaryList', [])

                    # If we're looking for a specific job prefix, filter the results
                    if job_prefix:
                        job_prefix_lower = job_prefix.lower()
                        filtered_summaries = []

                        for job_summary in job_summaries:
                            job_name = job_summary.get('jobName', '')

                            # Check for exact match
                            if job_name == job_prefix:
                                filtered_summaries.append(job_summary)
                                continue

                            # Check for case-insensitive match
                            if job_name.lower() == job_prefix_lower:
                                filtered_summaries.append(job_summary)
                                continue

                            # Check for substring match
                            if job_prefix_lower in job_name.lower():
                                filtered_summaries.append(job_summary)
                                continue

                        job_summaries = filtered_summaries

                    # Get job details for each job summary
                    if job_summaries:
                        job_ids = [job['jobId'] for job in job_summaries]

                        # Process in batches of 100 to avoid API limits
                        for i in range(0, len(job_ids), 100):
                            batch_ids = job_ids[i:i+100]
                            try:
                                job_details = batch.describe_jobs(jobs=batch_ids)['jobs']
                                all_jobs.extend(job_details)
                                if job_prefix:
                                    print(f"Found {len(job_details)} jobs matching '{job_prefix}' in queue {queue} with status {status}")
                            except Exception as e:
                                print(f"Error retrieving job details: {str(e)}")

                    # Handle pagination if there are more jobs
                    while 'nextToken' in response:
                        response = batch.list_jobs(
                            jobQueue=queue,
                            jobStatus=status,
                            maxResults=min(max_jobs, 100),
                            nextToken=response['nextToken']
                        )

                        job_summaries = response.get('jobSummaryList', [])

                        # If we're looking for a specific job prefix, filter the results
                        if job_prefix:
                            job_prefix_lower = job_prefix.lower()
                            filtered_summaries = []

                            for job_summary in job_summaries:
                                job_name = job_summary.get('jobName', '')

                                # Check for exact match
                                if job_name == job_prefix:
                                    filtered_summaries.append(job_summary)
                                    continue

                                # Check for case-insensitive match
                                if job_name.lower() == job_prefix_lower:
                                    filtered_summaries.append(job_summary)
                                    continue

                                # Check for substring match
                                if job_prefix_lower in job_name.lower():
                                    filtered_summaries.append(job_summary)
                                    continue

                            job_summaries = filtered_summaries

                        # Get job details for each job summary
                        if job_summaries:
                            job_ids = [job['jobId'] for job in job_summaries]

                            # Process in batches of 100 to avoid API limits
                            for i in range(0, len(job_ids), 100):
                                batch_ids = job_ids[i:i+100]
                                try:
                                    job_details = batch.describe_jobs(jobs=batch_ids)['jobs']
                                    all_jobs.extend(job_details)
                                    if job_prefix:
                                        print(f"Found {len(job_details)} additional jobs matching '{job_prefix}' in queue {queue} with status {status}")
                                except Exception as e:
                                    print(f"Error retrieving job details: {str(e)}")
                except Exception as e:
                    if "ArrayJob, Multi-node Job and job status are not applicable" not in str(e):
                        print(f"Error retrieving jobs from queue {queue} with status {status}: {str(e)}")
                    continue
        except Exception as e:
            print(f"Error retrieving jobs from queue {queue}: {str(e)}")

    return all_jobs

def get_job_log_streams(job_id):
    """
    Get CloudWatch log streams for a specific job.
    Returns a list of log stream names.
    """
    batch = boto3.client('batch', config=config)
    logs = boto3.client('logs', config=config)

    try:
        # Get job details
        response = batch.describe_jobs(jobs=[job_id])
        if not response['jobs']:
            return []

        job = response['jobs'][0]
        attempts = job.get('attempts', [])

        # Get log streams for each attempt
        log_streams = []

        # List of log groups to check
        log_groups = ['/aws/batch/job', 'metta-batch-dist-train']

        for attempt in attempts:
            container = attempt.get('container', {})
            log_stream_prefix = container.get('logStreamName')

            if log_stream_prefix:
                # Check each log group for this stream prefix
                for log_group in log_groups:
                    try:
                        # For multi-node jobs, there will be multiple log streams with the same prefix
                        # but different node indices
                        response = logs.describe_log_streams(
                            logGroupName=log_group,
                            logStreamNamePrefix=log_stream_prefix,
                            orderBy='LogStreamName',
                            descending=False
                        )

                        if response.get('logStreams'):
                            # Add log group information to each stream
                            for stream in response.get('logStreams', []):
                                stream['logGroupName'] = log_group

                            log_streams.extend(response.get('logStreams', []))

                            # Handle pagination if there are more log streams
                            while response.get('nextToken'):
                                response = logs.describe_log_streams(
                                    logGroupName=log_group,
                                    logStreamNamePrefix=log_stream_prefix,
                                    orderBy='LogStreamName',
                                    descending=False,
                                    nextToken=response['nextToken']
                                )

                                # Add log group information to each stream
                                for stream in response.get('logStreams', []):
                                    stream['logGroupName'] = log_group

                                log_streams.extend(response.get('logStreams', []))
                    except Exception as e:
                        # Just continue to the next log group if this one fails
                        continue

            # Also check for job ID in the default stream format
            for log_group in log_groups:
                try:
                    # Try with job ID as prefix
                    response = logs.describe_log_streams(
                        logGroupName=log_group,
                        logStreamNamePrefix=f"default/{job_id}",
                        orderBy='LogStreamName',
                        descending=False
                    )

                    if response.get('logStreams'):
                        # Add log group information to each stream
                        for stream in response.get('logStreams', []):
                            stream['logGroupName'] = log_group

                        log_streams.extend(response.get('logStreams', []))
                except Exception:
                    # Just continue to the next log group if this one fails
                    continue

        return log_streams
    except Exception as e:
        print(f"Error retrieving log streams for job {job_id}: {str(e)}")
        return []

def find_alternative_log_streams(job_id):
    """
    Try to find alternative log streams for a job when the standard approach fails.
    This checks various log groups and patterns that might contain logs for the job.
    """
    logs = boto3.client('logs', config=config)
    log_streams = []

    # List of log groups to check
    log_groups = ['metta-batch-dist-train', '/aws/batch/job']

    # List of patterns to try
    patterns = [
        f"default/{job_id}",  # Direct job ID pattern
        "default/",           # Check all default streams
    ]

    for log_group in log_groups:
        for pattern in patterns:
            try:
                response = logs.describe_log_streams(
                    logGroupName=log_group,
                    logStreamNamePrefix=pattern,
                    orderBy='LogStreamName',
                    descending=False,
                    limit=50  # Increase limit to find more potential matches
                )

                if response.get('logStreams'):
                    # Add log group information to each stream
                    for stream in response.get('logStreams', []):
                        stream['logGroupName'] = log_group
                        # Add to list if it's not already there
                        if not any(s.get('logStreamName') == stream.get('logStreamName') for s in log_streams):
                            log_streams.append(stream)
            except Exception as e:
                print(f"Error checking {log_group}/{pattern}: {str(e)}")
                continue

    return log_streams

def get_log_events(log_group, log_stream, start_time=None, tail=False):
    """
    Get log events from a specific log stream.
    If tail is True, only return the most recent events.
    """
    logs = boto3.client('logs', config=config)

    params = {
        'logGroupName': log_group,
        'logStreamName': log_stream,
        'startFromHead': True
    }

    if start_time:
        params['startTime'] = start_time

    if tail:
        # Only get the most recent events if tailing
        try:
            response = logs.describe_log_streams(
                logGroupName=log_group,
                logStreamNamePrefix=log_stream,
                orderBy='LogStreamName',
                limit=1
            )
            if response.get('logStreams'):
                stream_info = response['logStreams'][0]
                # Get events from the last 10 minutes or 1000 events, whichever is less
                ten_min_ago = int((datetime.now() - timedelta(minutes=10)).timestamp() * 1000)
                params['startTime'] = max(stream_info.get('firstEventTimestamp', ten_min_ago), ten_min_ago)
                params['limit'] = 1000
        except Exception as e:
            print(f"Error setting up tail for log stream {log_stream}: {str(e)}")

    try:
        response = logs.get_log_events(**params)
        events = response.get('events', [])

        # Handle pagination if not tailing
        if not tail:
            while response.get('nextForwardToken'):
                next_token = response['nextForwardToken']
                response = logs.get_log_events(
                    logGroupName=log_group,
                    logStreamName=log_stream,
                    nextToken=next_token,
                    startFromHead=True
                )
                events.extend(response.get('events', []))

                # Break if we've reached the end or if the token hasn't changed
                if next_token == response.get('nextForwardToken'):
                    break

        return events
    except Exception as e:
        print(f"Error retrieving log events for stream {log_stream}: {str(e)}")
        return []

def print_logs(job_id, node=None, tail=False):
    """
    Print logs for a specific job and optionally a specific node.
    If tail is True, continuously poll for new logs.
    """
    log_streams = get_job_log_streams(job_id)

    if not log_streams:
        print(f"No log streams found for job {job_id}")
        return

    # Filter log streams by node if specified
    if node is not None and node != 'all':
        try:
            node_idx = int(node)
            # Filter streams that contain the node index
            log_streams = [stream for stream in log_streams if f'/node-{node_idx}/' in stream['logStreamName']]
            if not log_streams:
                print(f"No log streams found for node {node} in job {job_id}")
                return
        except ValueError:
            print(f"Invalid node index: {node}. Must be an integer or 'all'.")
            return

    # Sort log streams by name to ensure consistent ordering
    log_streams.sort(key=lambda x: x['logStreamName'])

    # Print logs for each stream
    last_timestamps = {}

    def print_stream_events(stream, start_time=None):
        stream_name = stream['logStreamName']
        log_group = stream.get('logGroupName', '/aws/batch/job')  # Default to /aws/batch/job if not specified
        node_info = "unknown"

        # Extract node information from the stream name
        if '/node-' in stream_name:
            node_info = stream_name.split('/node-')[1].split('/')[0]

        events = get_log_events(log_group, stream_name, start_time, tail)

        if events:
            last_timestamp = events[-1]['timestamp']
            last_timestamps[stream_name] = last_timestamp

            for event in events:
                timestamp = datetime.fromtimestamp(event['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                message = event['message'].rstrip()
                print(f"[Node {node_info}] {timestamp}: {message}")

            return True
        return False

    # Initial print of all logs
    for stream in log_streams:
        print_stream_events(stream)

    # If tail is True, continuously poll for new logs
    if tail:
        print("\nTailing logs... (Press Ctrl+C to stop)")
        try:
            while True:
                time.sleep(5)  # Poll every 5 seconds

                # Check for new logs in each stream
                with ThreadPoolExecutor(max_workers=min(10, len(log_streams))) as executor:
                    futures = []
                    for stream in log_streams:
                        stream_name = stream['logStreamName']
                        start_time = last_timestamps.get(stream_name)
                        if start_time:
                            # Add 1ms to avoid duplicate logs
                            start_time += 1
                        futures.append(executor.submit(print_stream_events, stream, start_time))

                    # Wait for all futures to complete
                    for future in as_completed(futures):
                        future.result()
        except KeyboardInterrupt:
            print("\nStopped tailing logs.")

def print_log_stream(log_group, log_stream, tail=False):
    """
    Print logs from a specific log group and log stream.
    If tail is True, continuously poll for new logs.
    """
    logs = boto3.client('logs', config=config)

    # Check if the log stream exists
    try:
        response = logs.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=log_stream,
            limit=1
        )

        if not response.get('logStreams'):
            print(f"Log stream '{log_stream}' not found in log group '{log_group}'")
            return

        stream = response['logStreams'][0]
        last_timestamp = None

        # Function to print events from the stream
        def print_stream_events(start_time=None):
            nonlocal last_timestamp
            events = get_log_events(log_group, stream['logStreamName'], start_time, tail)

            if events:
                last_timestamp = events[-1]['timestamp']

                for event in events:
                    timestamp = datetime.fromtimestamp(event['timestamp'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    message = event['message'].rstrip()
                    print(f"{timestamp}: {message}")

                return True
            return False

        # Initial print of all logs
        print_stream_events()

        # If tail is True, continuously poll for new logs
        if tail:
            print("\nTailing logs... (Press Ctrl+C to stop)")
            try:
                while True:
                    time.sleep(5)  # Poll every 5 seconds

                    # Check for new logs
                    if last_timestamp:
                        # Add 1ms to avoid duplicate logs
                        print_stream_events(last_timestamp + 1)
            except KeyboardInterrupt:
                print("\nStopped tailing logs.")

    except Exception as e:
        print(f"Error retrieving log stream: {str(e)}")

def list_recent_jobs(job_queue=None, max_jobs=10, interactive=True):
    """List recent jobs and optionally allow selection of a job to view logs for."""
    # Get job details using the improved function
    all_jobs = get_job_details(job_queue=job_queue, max_jobs=max_jobs)

    # Sort jobs by creation time (newest first)
    all_jobs.sort(key=lambda x: x.get('createdAt', 0), reverse=True)

    # Limit to max_jobs
    all_jobs = all_jobs[:max_jobs]

    if not all_jobs:
        print("No recent jobs found.")
        return None

    print("\nRecent Jobs (showing up to {}, newest first):".format(max_jobs))
    print("------------------------------------------------------------------------------------------------------------------------")
    print(f"#    Job Name                                 Job ID                    Status          Age             Created At          ")
    print("------------------------------------------------------------------------------------------------------------------------")

    for i, job in enumerate(all_jobs, 1):
        job_name = job.get('jobName', 'Unknown')
        job_id = job.get('jobId', 'Unknown')
        status = job.get('status', 'Unknown')
        created_at = job.get('createdAt', 0)

        # Format the timestamp
        if created_at:
            created_at_str = datetime.fromtimestamp(created_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
            age = format_time_difference(created_at / 1000)
        else:
            created_at_str = 'Unknown'
            age = 'Unknown'

        print(f"{i:<4} {job_name[:40]:<40} {job_id[:25]:<25} {status:<15} {age:<15} {created_at_str}")

    # If interactive mode is enabled, prompt for selection
    if interactive:
        try:
            choice = input("\nEnter job number to view logs (or press Enter to exit): ")
            if choice.strip():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(all_jobs):
                    selected_job = all_jobs[choice_idx]
                    show_job_logs(selected_job, tail=False)
                    return selected_job.get('jobId')
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(all_jobs)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")

    return None

def format_time_difference(timestamp):
    """
    Format the time difference between now and the given timestamp in a human-readable format.
    Returns a string like "1d 15h 12m" or "45m 30s" for more recent times.
    """
    if not timestamp:
        return "Unknown"

    try:
        # Convert timestamp from milliseconds to seconds if needed
        if timestamp > 1000000000000:  # If timestamp is in milliseconds
            timestamp = timestamp / 1000

        # Calculate time difference
        created_time = datetime.fromtimestamp(timestamp)
        now = datetime.now()
        diff = now - created_time

        # Calculate components
        days = diff.days
        hours = diff.seconds // 3600
        minutes = (diff.seconds % 3600) // 60
        seconds = diff.seconds % 60

        # Format the string based on the time difference
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception:
        return "Unknown"

def show_job_logs(job, tail=False):
    """Show logs for a specific job."""
    job_id = job.get('jobId')
    if not job_id:
        print("Error: Job ID not found in job details.")
        return

    job_name = job.get('jobName', 'Unknown')
    status = job.get('status', 'Unknown')
    print(f"\nShowing logs for job {job_name} (ID: {job_id}):")

    # If the job is in RUNNABLE or SUBMITTED state, logs might not be available yet
    if status in ['RUNNABLE', 'SUBMITTED', 'PENDING']:
        print(f"Job is in {status} state. Logs may not be available until the job starts running.")

    # Get log streams for the job
    log_streams = []
    try:
        logs = boto3.client('logs')

        # Try to find log streams in the default AWS Batch log group
        log_group = '/aws/batch/job'
        paginator = logs.get_paginator('describe_log_streams')

        # Use pagination to get all log streams
        for page in paginator.paginate(
            logGroupName=log_group,
            logStreamNamePrefix=job_id
        ):
            log_streams.extend(page.get('logStreams', []))

        if not log_streams:
            # If no logs found in default group, try alternative log groups
            alternative_log_groups = [
                f'/aws/batch/job/{job_id}',
                f'/aws/batch/{job_id}'
            ]

            for alt_group in alternative_log_groups:
                try:
                    for page in paginator.paginate(
                        logGroupName=alt_group
                    ):
                        streams = page.get('logStreams', [])
                        for stream in streams:
                            stream['logGroupName'] = alt_group  # Add log group name to stream info
                        log_streams.extend(streams)

                    if log_streams:
                        break  # Stop if we found logs in this group
                except logs.exceptions.ResourceNotFoundException:
                    continue

        if not log_streams:
            print(f"No log streams found for job {job_id}.")

            # Provide more information based on job status
            if status == 'SUCCEEDED':
                print("The job has completed successfully, but no logs were found. This can happen if the job didn't produce any output.")
            elif status == 'FAILED':
                print("The job has failed, but no logs were found. This can happen if the job failed before producing any output.")
                print("You might want to check the job definition and container configuration.")
            elif status in ['RUNNING', 'STARTING']:
                print("The job is running, but logs might not be available yet. Try again in a few moments.")

            return

        # Sort log streams by name
        log_streams.sort(key=lambda x: x.get('logStreamName', ''))

        # Get and print log events for each stream
        for stream in log_streams:
            stream_name = stream.get('logStreamName', '')
            log_group_name = stream.get('logGroupName', log_group)

            print(f"\nLog stream: {log_group_name}/{stream_name}")
            print("-" * 80)

            # Get log events
            try:
                response = logs.get_log_events(
                    logGroupName=log_group_name,
                    logStreamName=stream_name,
                    startFromHead=True
                )

                events = response.get('events', [])
                if not events:
                    print("No log events found in this stream.")
                    continue

                for event in events:
                    timestamp = datetime.fromtimestamp(event.get('timestamp', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    message = event.get('message', '')
                    print(f"[{timestamp}] {message}")

                # If tail is True, continuously poll for new logs
                if tail:
                    print("\nTailing logs... (Press Ctrl+C to stop)")
                    next_token = response.get('nextForwardToken')

                    try:
                        while True:
                            time.sleep(2)  # Poll every 2 seconds

                            response = logs.get_log_events(
                                logGroupName=log_group_name,
                                logStreamName=stream_name,
                                nextToken=next_token
                            )

                            events = response.get('events', [])
                            for event in events:
                                timestamp = datetime.fromtimestamp(event.get('timestamp', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                                message = event.get('message', '')
                                print(f"[{timestamp}] {message}")

                            next_token = response.get('nextForwardToken')
                    except KeyboardInterrupt:
                        print("\nStopped tailing logs.")
            except Exception as e:
                print(f"Error retrieving log events: {str(e)}")
    except Exception as e:
        print(f"Error retrieving log streams: {str(e)}")

def main():
    """Main function to handle command line arguments and execute appropriate actions."""
    parser = argparse.ArgumentParser(description='Get AWS Batch job logs')
    parser.add_argument('--job-queue', help='AWS Batch job queue name')
    parser.add_argument('--job', help='Job name or ID to search for')
    parser.add_argument('--max-jobs', type=int, default=10, help='Maximum number of jobs to list')
    parser.add_argument('--list-jobs', action='store_true', help='List recent jobs')
    parser.add_argument('--list-queues', action='store_true', help='List available job queues')
    parser.add_argument('--tail', action='store_true', help='Tail logs')
    parser.add_argument('--no-logs', action='store_true', help='Do not show logs')
    parser.add_argument('--no-interactive', action='store_false', dest='interactive', help='Do not use interactive mode')
    parser.add_argument('--job-index', type=int, help='Index of job to show logs for (from list)')
    args = parser.parse_args()

    if args.list_queues:
        queues = get_batch_job_queues()
        print("\nAvailable Job Queues:")
        print("---------------------")
        for i, queue in enumerate(queues, 1):
            print(f"{i}. {queue}")
        return

    # If a job is specified, try to find it by ID first, then by name
    if args.job:
        # Check if the job argument looks like a job ID (typically starts with a specific pattern or is a UUID)
        is_job_id = args.job.startswith('job-') or args.job.startswith('aws:') or (len(args.job) >= 32 and '-' in args.job)

        if is_job_id:
            # If it looks like a job ID, search directly by ID
            try:
                batch = boto3.client('batch')
                response = batch.describe_jobs(jobs=[args.job])
                jobs = response.get('jobs', [])

                if jobs:
                    # Sort jobs by creation time (newest first)
                    jobs.sort(key=lambda x: x.get('createdAt', 0), reverse=True)

                    # Display the jobs with age
                    print("\nJob Details:")
                    print("------------------------------------------------------------------------------------------------------------------------")
                    print(f"#    Job Name                                 Job ID                    Status          Age             Created At          ")
                    print("------------------------------------------------------------------------------------------------------------------------")

                    for i, job in enumerate(jobs, 1):
                        job_name = job.get('jobName', 'Unknown')
                        job_id = job.get('jobId', 'Unknown')
                        status = job.get('status', 'Unknown')
                        created_at = job.get('createdAt', 0)

                        # Format the timestamp
                        if created_at:
                            created_at_str = datetime.fromtimestamp(created_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
                            age = format_time_difference(created_at / 1000)
                        else:
                            created_at_str = 'Unknown'
                            age = 'Unknown'

                        print(f"{i:<4} {job_name[:40]:<40} {job_id[:25]:<25} {status:<15} {age:<15} {created_at_str}")

                    # If only one job is found, show its logs
                    if len(jobs) == 1 and not args.no_logs:
                        job = jobs[0]
                        show_job_logs(job, tail=args.tail)
                    return
            except Exception as e:
                print(f"Error retrieving job with ID '{args.job}': {str(e)}")

        # If not a job ID or job ID search failed, try by job name
        jobs = get_job_details(job_prefix=args.job, job_queue=args.job_queue, max_jobs=args.max_jobs)

        if not jobs:
            print(f"No jobs found matching '{args.job}'. Listing recent jobs:")
            list_recent_jobs(job_queue=args.job_queue, max_jobs=args.max_jobs, interactive=args.interactive)
            return

        # Sort jobs by creation time (newest first)
        jobs.sort(key=lambda x: x.get('createdAt', 0), reverse=True)

        # Display the jobs with age
        print("\nMatching Jobs:")
        print("------------------------------------------------------------------------------------------------------------------------")
        print(f"#    Job Name                                 Job ID                    Status          Age             Created At          ")
        print("------------------------------------------------------------------------------------------------------------------------")

        for i, job in enumerate(jobs, 1):
            job_name = job.get('jobName', 'Unknown')
            job_id = job.get('jobId', 'Unknown')
            status = job.get('status', 'Unknown')
            created_at = job.get('createdAt', 0)

            # Format the timestamp
            if created_at:
                created_at_str = datetime.fromtimestamp(created_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
                age = format_time_difference(created_at / 1000)
            else:
                created_at_str = 'Unknown'
                age = 'Unknown'

            print(f"{i:<4} {job_name[:40]:<40} {job_id[:25]:<25} {status:<15} {age:<15} {created_at_str}")

        # If only one job is found, show its logs
        if len(jobs) == 1 and not args.no_logs:
            job = jobs[0]
            show_job_logs(job, tail=args.tail)
        # If multiple jobs are found and interactive mode is enabled, prompt for selection
        elif len(jobs) > 1 and args.interactive and not args.no_logs:
            try:
                job_index = args.job_index if args.job_index is not None else int(input("\nEnter job number to view logs: "))
                if 1 <= job_index <= len(jobs):
                    show_job_logs(jobs[job_index - 1], tail=args.tail)
                else:
                    print(f"Invalid job index. Please enter a number between 1 and {len(jobs)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
        return

    # If no specific job is requested, list recent jobs
    list_recent_jobs(job_queue=args.job_queue, max_jobs=args.max_jobs, interactive=args.interactive)

if __name__ == "__main__":
    main()
