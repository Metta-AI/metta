#!/usr/bin/env python3
import argparse
import boto3
import sys
import time
import os
from datetime import datetime, timedelta
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the project root to the Python path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from metta.devops.aws.cluster_info import get_batch_job_queues, get_batch_jobs

# Configure boto3 to use a higher max_pool_connections
config = Config(
    retries={'max_attempts': 10, 'mode': 'standard'},
    max_pool_connections=50
)

def get_job_details(job_id=None, job_prefix=None, job_queue=None, max_jobs=100):
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

    # If no specific job ID, get jobs from all queues or specified queue
    if job_queue:
        job_queues = [job_queue]
    else:
        job_queues = get_batch_job_queues()

    all_jobs = []

    for queue in job_queues:
        try:
            jobs = get_batch_jobs(queue, max_jobs=max_jobs)
            if job_prefix:
                # Filter jobs by prefix
                jobs = [job for job in jobs if job['name'].startswith(job_prefix)]
            all_jobs.extend(jobs)
        except Exception as e:
            print(f"Error retrieving jobs from queue {queue}: {str(e)}")

    # Sort by creation time, newest first
    all_jobs.sort(key=lambda x: x.get('createdAt', 0) if isinstance(x.get('createdAt'), int) else 0, reverse=True)

    # Limit to max_jobs
    return all_jobs[:max_jobs]

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

def list_recent_jobs(job_queue=None, max_jobs=10, interactive=True):
    """
    List recent jobs from all queues or a specific queue.
    If interactive is True, allow the user to select a job by number.
    Returns the selected job ID if interactive and a selection is made, otherwise None.
    """
    jobs = get_job_details(job_queue=job_queue, max_jobs=max_jobs)

    if not jobs:
        print("No recent jobs found.")
        return None

    print(f"\nRecent Jobs (showing up to {max_jobs}):")
    print("-" * 100)
    print(f"{'#':<4} {'Job Name':<40} {'Job ID':<25} {'Status':<15} {'Created At':<20}")
    print("-" * 100)

    job_ids = []
    for i, job in enumerate(jobs, 1):
        if isinstance(job, dict):
            # Handle job details from get_batch_jobs
            job_id = job.get('jobId', job.get('stop_command', '').split()[-1] if job.get('stop_command') else 'Unknown')
            job_name = job.get('name', 'Unknown')
            status = job.get('status', 'Unknown')

            # Handle creation time
            created_at = 'Unknown'
            if 'createdAt' in job and job['createdAt']:
                try:
                    created_at = datetime.fromtimestamp(job['createdAt'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                except (TypeError, ValueError):
                    pass
        else:
            # Fallback for unexpected data type
            job_id = 'Unknown'
            job_name = 'Unknown'
            status = 'Unknown'
            created_at = 'Unknown'

        job_ids.append(job_id)
        print(f"{i:<4} {job_name:<40} {job_id:<25} {status:<15} {created_at:<20}")

    if interactive and job_ids:
        try:
            choice = input("\nEnter job number to view logs (or press Enter to exit): ")
            if choice.strip():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(job_ids):
                    return job_ids[choice_idx]
                else:
                    print(f"Invalid selection. Please enter a number between 1 and {len(job_ids)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return None

def main():
    parser = argparse.ArgumentParser(description='Retrieve and display logs from AWS Batch jobs')
    parser.add_argument('--job', type=str, help='Job ID or name prefix to retrieve logs for')
    parser.add_argument('--node', type=str, default='all', help='Node index to retrieve logs for (default: all)')
    parser.add_argument('--tail', action='store_true', help='Continuously poll for new logs')
    parser.add_argument('--profile', type=str, help='AWS profile to use')
    parser.add_argument('--job-queue', type=str, help='Specific AWS Batch job queue to use')
    parser.add_argument('--max', type=int, default=10, help='Maximum number of jobs to display (default: 10)')
    parser.add_argument('--non-interactive', action='store_true', help='Disable interactive job selection')

    args = parser.parse_args()

    # Set AWS profile if specified
    if args.profile:
        boto3.setup_default_session(profile_name=args.profile)

    if not args.job:
        # If no job specified, list recent jobs and allow selection
        selected_job_id = list_recent_jobs(args.job_queue, args.max, not args.non_interactive)
        if selected_job_id:
            print(f"Retrieving logs for job {selected_job_id}...")
            print_logs(selected_job_id, args.node, args.tail)
        return

    # Check if the input is a job ID or a job name prefix
    if args.job.startswith('job-'):
        # It's a job ID
        job_id = args.job
        job_details = get_job_details(job_id=job_id)

        if not job_details:
            print(f"Job {job_id} not found. Listing jobs with similar IDs:")
            similar_jobs = get_job_details(job_prefix=job_id.split('-')[1], job_queue=args.job_queue, max_jobs=args.max)
            if similar_jobs:
                # Display numbered list of similar jobs
                print("-" * 100)
                print(f"{'#':<4} {'Job Name':<40} {'Job ID':<25} {'Status':<15} {'Created At':<20}")
                print("-" * 100)

                job_ids = []
                for i, job in enumerate(similar_jobs, 1):
                    job_id = job.get('jobId', job.get('stop_command', '').split()[-1] if job.get('stop_command') else 'Unknown')
                    job_name = job.get('name', 'Unknown')
                    status = job.get('status', 'Unknown')

                    # Handle creation time
                    created_at = 'Unknown'
                    if 'createdAt' in job and job['createdAt']:
                        try:
                            created_at = datetime.fromtimestamp(job['createdAt'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                        except (TypeError, ValueError):
                            pass

                    job_ids.append(job_id)
                    print(f"{i:<4} {job_name:<40} {job_id:<25} {status:<15} {created_at:<20}")

                # Allow selection if interactive
                if not args.non_interactive:
                    try:
                        choice = input("\nEnter job number to view logs (or press Enter to exit): ")
                        if choice.strip():
                            choice_idx = int(choice) - 1
                            if 0 <= choice_idx < len(job_ids):
                                selected_job_id = job_ids[choice_idx]
                                print(f"Retrieving logs for job {selected_job_id}...")
                                print_logs(selected_job_id, args.node, args.tail)
                                return
                            else:
                                print(f"Invalid selection. Please enter a number between 1 and {len(job_ids)}.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            else:
                print("No similar jobs found.")
            return

        print(f"Retrieving logs for job {job_id}...")
        print_logs(job_id, args.node, args.tail)
    else:
        # It's a job name prefix
        jobs = get_job_details(job_prefix=args.job, job_queue=args.job_queue, max_jobs=args.max)

        if not jobs:
            print(f"No jobs found with prefix '{args.job}'. Listing recent jobs:")
            selected_job_id = list_recent_jobs(args.job_queue, args.max, not args.non_interactive)
            if selected_job_id:
                print(f"Retrieving logs for job {selected_job_id}...")
                print_logs(selected_job_id, args.node, args.tail)
            return

        if len(jobs) == 1:
            # If only one job matches, show its logs
            job_id = jobs[0].get('jobId', jobs[0].get('stop_command', '').split()[-1] if jobs[0].get('stop_command') else 'Unknown')
            job_name = jobs[0].get('name', 'Unknown')
            print(f"Found one job matching '{args.job}': {job_name} ({job_id})")
            print_logs(job_id, args.node, args.tail)
        else:
            # If multiple jobs match, list them with numbers
            print(f"Multiple jobs found with prefix '{args.job}' (showing up to {args.max}):")
            print("-" * 100)
            print(f"{'#':<4} {'Job Name':<40} {'Job ID':<25} {'Status':<15} {'Created At':<20}")
            print("-" * 100)

            job_ids = []
            for i, job in enumerate(jobs, 1):
                job_id = job.get('jobId', job.get('stop_command', '').split()[-1] if job.get('stop_command') else 'Unknown')
                job_name = job.get('name', 'Unknown')
                status = job.get('status', 'Unknown')

                # Handle creation time
                created_at = 'Unknown'
                if 'createdAt' in job and job['createdAt']:
                    try:
                        created_at = datetime.fromtimestamp(job['createdAt'] / 1000).strftime('%Y-%m-%d %H:%M:%S')
                    except (TypeError, ValueError):
                        pass

                job_ids.append(job_id)
                print(f"{i:<4} {job_name:<40} {job_id:<25} {status:<15} {created_at:<20}")

            # Allow selection if interactive
            if not args.non_interactive:
                try:
                    choice = input("\nEnter job number to view logs (or press Enter to exit): ")
                    if choice.strip():
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(job_ids):
                            selected_job_id = job_ids[choice_idx]
                            print(f"Retrieving logs for job {selected_job_id}...")
                            print_logs(selected_job_id, args.node, args.tail)
                        else:
                            print(f"Invalid selection. Please enter a number between 1 and {len(job_ids)}.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()
