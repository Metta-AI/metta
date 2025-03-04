#!/usr/bin/env python3
import argparse
import boto3
import sys
import time
from datetime import datetime, timedelta
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from devops.aws.cluster_info import get_batch_job_queues, get_batch_jobs

# Configure boto3 to use a higher max_pool_connections
config = Config(
    retries={'max_attempts': 10, 'mode': 'standard'},
    max_pool_connections=50
)

def get_job_details(job_id=None, job_prefix=None):
    """
    Get details for a specific job by ID or jobs matching a prefix.
    Returns a list of job details.
    """
    batch = boto3.client('batch', config=config)

    if job_id:
        try:
            response = batch.describe_jobs(jobs=[job_id])
            if response['jobs']:
                return response['jobs']
            return []
        except Exception as e:
            print(f"Error retrieving job details for {job_id}: {str(e)}")
            return []

    # If no specific job ID, get jobs from all queues
    job_queues = get_batch_job_queues()
    all_jobs = []

    for queue in job_queues:
        jobs = get_batch_jobs(queue, max_jobs=100)
        if job_prefix:
            # Filter jobs by prefix
            jobs = [job for job in jobs if job['name'].startswith(job_prefix)]
        all_jobs.extend(jobs)

    # Sort by creation time, newest first
    all_jobs.sort(key=lambda x: x.get('createdAt', 0), reverse=True)
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
        for attempt in attempts:
            container = attempt.get('container', {})
            log_stream_prefix = container.get('logStreamName')
            if log_stream_prefix:
                # For multi-node jobs, there will be multiple log streams with the same prefix
                # but different node indices
                response = logs.describe_log_streams(
                    logGroupName='/aws/batch/job',
                    logStreamNamePrefix=log_stream_prefix,
                    orderBy='LogStreamName',
                    descending=False
                )
                log_streams.extend(response.get('logStreams', []))

                # Handle pagination if there are more log streams
                while response.get('nextToken'):
                    response = logs.describe_log_streams(
                        logGroupName='/aws/batch/job',
                        logStreamNamePrefix=log_stream_prefix,
                        orderBy='LogStreamName',
                        descending=False,
                        nextToken=response['nextToken']
                    )
                    log_streams.extend(response.get('logStreams', []))

        return log_streams
    except Exception as e:
        print(f"Error retrieving log streams for job {job_id}: {str(e)}")
        return []

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
        node_info = "unknown"

        # Extract node information from the stream name
        if '/node-' in stream_name:
            node_info = stream_name.split('/node-')[1].split('/')[0]

        events = get_log_events('/aws/batch/job', stream_name, start_time, tail)

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

def list_recent_jobs():
    """List recent jobs from all queues."""
    jobs = get_job_details()

    if not jobs:
        print("No recent jobs found.")
        return

    print("\nRecent Jobs:")
    print("-" * 80)
    print(f"{'Job Name':<40} {'Job ID':<25} {'Status':<15} {'Created At':<20}")
    print("-" * 80)

    for job in jobs:
        if isinstance(job, dict) and 'jobId' in job:
            # Handle job details from get_batch_jobs
            job_id = job['jobId'] if 'jobId' in job else job.get('stop_command', '').split()[-1]
            job_name = job['name']
            status = job['status']
            created_at = datetime.fromtimestamp(job.get('createdAt', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S') if 'createdAt' in job else 'Unknown'
        else:
            # Handle job details from describe_jobs
            job_id = job.get('jobId', 'Unknown')
            job_name = job.get('jobName', 'Unknown')
            status = job.get('status', 'Unknown')
            created_at = datetime.fromtimestamp(job.get('createdAt', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S') if 'createdAt' in job else 'Unknown'

        print(f"{job_name:<40} {job_id:<25} {status:<15} {created_at:<20}")

def main():
    parser = argparse.ArgumentParser(description='Retrieve and display logs from AWS Batch jobs')
    parser.add_argument('--job', type=str, help='Job ID or name prefix to retrieve logs for')
    parser.add_argument('--node', type=str, default='all', help='Node index to retrieve logs for (default: all)')
    parser.add_argument('--tail', action='store_true', help='Continuously poll for new logs')
    parser.add_argument('--profile', type=str, help='AWS profile to use')

    args = parser.parse_args()

    # Set AWS profile if specified
    if args.profile:
        boto3.setup_default_session(profile_name=args.profile)

    if not args.job:
        # If no job specified, list recent jobs
        list_recent_jobs()
        return

    # Check if the input is a job ID or a job name prefix
    if args.job.startswith('job-'):
        # It's a job ID
        job_id = args.job
        job_details = get_job_details(job_id=job_id)

        if not job_details:
            print(f"Job {job_id} not found. Listing jobs with similar IDs:")
            similar_jobs = get_job_details(job_prefix=job_id.split('-')[1])
            if similar_jobs:
                for job in similar_jobs:
                    print(f"Job Name: {job['name']}, Job ID: {job.get('jobId', job.get('stop_command', '').split()[-1])}")
            else:
                print("No similar jobs found.")
            return

        print(f"Retrieving logs for job {job_id}...")
        print_logs(job_id, args.node, args.tail)
    else:
        # It's a job name prefix
        jobs = get_job_details(job_prefix=args.job)

        if not jobs:
            print(f"No jobs found with prefix '{args.job}'. Listing recent jobs:")
            list_recent_jobs()
            return

        if len(jobs) == 1:
            # If only one job matches, show its logs
            job_id = jobs[0].get('jobId', jobs[0].get('stop_command', '').split()[-1])
            print(f"Found one job matching '{args.job}': {job_id}")
            print_logs(job_id, args.node, args.tail)
        else:
            # If multiple jobs match, list them
            print(f"Multiple jobs found with prefix '{args.job}':")
            print("-" * 80)
            print(f"{'Job Name':<40} {'Job ID':<25} {'Status':<15} {'Created At':<20}")
            print("-" * 80)

            for job in jobs:
                job_id = job.get('jobId', job.get('stop_command', '').split()[-1])
                job_name = job['name']
                status = job['status']
                created_at = datetime.fromtimestamp(job.get('createdAt', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S') if 'createdAt' in job else 'Unknown'

                print(f"{job_name:<40} {job_id:<25} {status:<15} {created_at:<20}")

if __name__ == "__main__":
    main()
