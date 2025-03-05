"""
AWS Batch job utilities for interacting with AWS Batch jobs.
"""

import boto3
from botocore.config import Config
from datetime import datetime

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

def get_job_attempts(job_id):
    """Get all attempts for a specific job."""
    if not job_id:
        return []

    config = Config(retries={'max_attempts': 10, 'mode': 'standard'})
    batch = boto3.client('batch', config=config)

    try:
        response = batch.describe_jobs(jobs=[job_id])
        if not response['jobs']:
            return []

        job = response['jobs'][0]
        attempts = job.get('attempts', [])

        # Sort attempts by startedAt time (newest first)
        attempts.sort(key=lambda x: x.get('startedAt', 0) if x.get('startedAt') else 0, reverse=True)

        return attempts
    except Exception as e:
        print(f"Error retrieving job attempts for job {job_id}: {str(e)}")
        return []

def get_job_log_streams(job_id, attempt_index=None):
    """
    Get CloudWatch log streams for a specific job.

    Args:
        job_id (str): The job ID to get log streams for
        attempt_index (int, optional): The specific attempt index to get logs for.
            If None, returns log streams for all attempts.

    Returns:
        list: A list of log stream objects
    """
    if not job_id:
        return []

    log_streams = []
    try:
        logs = boto3.client('logs')

        # Try to find log streams in the default AWS Batch log group
        log_group = '/aws/batch/job'

        # Try different patterns for log stream names
        patterns = [
            job_id,  # Direct job ID pattern
            f"default/{job_id}",  # Standard AWS Batch pattern
            f"metta-batch-dist-train/default/{job_id}",  # Pattern seen in the CloudWatch console
            f"metta-batch-dist-train/default/"  # For multi-node jobs, search broader pattern
        ]

        for pattern in patterns:
            try:
                paginator = logs.get_paginator('describe_log_streams')

                # Use pagination to get all log streams
                for page in paginator.paginate(
                    logGroupName=log_group,
                    logStreamNamePrefix=pattern
                ):
                    for stream in page.get('logStreams', []):
                        stream['logGroupName'] = log_group  # Add log group name to stream info

                        # For multi-node jobs, check if the stream name contains the job ID
                        # This is important because the broader pattern might return unrelated streams
                        stream_name = stream.get('logStreamName', '')
                        if pattern == f"metta-batch-dist-train/default/" and job_id not in stream_name:
                            continue

                        # Extract attempt information from the stream name if possible
                        if '/attempt/' in stream_name:
                            try:
                                attempt_num = int(stream_name.split('/attempt/')[1].split('/')[0])
                                stream['attemptNum'] = attempt_num
                            except (ValueError, IndexError):
                                stream['attemptNum'] = -1
                        else:
                            stream['attemptNum'] = -1

                        # Add to list if not already there
                        if not any(s.get('logStreamName') == stream.get('logStreamName') for s in log_streams):
                            log_streams.append(stream)
            except Exception as e:
                # Just continue to the next pattern if this one fails
                continue

        # Filter by attempt index if specified
        if attempt_index is not None:
            log_streams = [stream for stream in log_streams if stream.get('attemptNum') == attempt_index]
    except Exception as e:
        print(f"Error retrieving log streams for job {job_id}: {str(e)}")

    return log_streams

def find_alternative_log_streams(job_id, attempt_index=None, debug=False):
    """
    Find alternative CloudWatch log streams for a job if standard ones aren't found.

    Args:
        job_id (str): The job ID to find log streams for
        attempt_index (int, optional): The specific attempt index to get logs for.
            If None, returns log streams for all attempts.
        debug (bool): Whether to show debug information

    Returns:
        list: A list of log stream objects
    """
    if not job_id:
        return []

    log_streams = []
    job_definition_name = None

    # First, try to get log stream names directly from the job details
    try:
        batch = boto3.client('batch')
        response = batch.describe_jobs(jobs=[job_id])

        if response['jobs']:
            job = response['jobs'][0]
            job_name = job.get('jobName')
            job_definition_name = job.get('jobDefinition', '').split('/')[-1].split(':')[0]

            if debug:
                print(f"Debug: Job name is {job_name}")
                if job_definition_name:
                    print(f"Debug: Job definition name is {job_definition_name}")

            # For multi-node jobs, check the container details in the nodes
            if 'nodeDetails' in job:
                if debug:
                    print(f"Debug: This is a multi-node job with {len(job['nodeDetails'].get('nodes', []))} nodes")

                for node in job['nodeDetails'].get('nodes', []):
                    for container in node.get('containers', []):
                        if 'logStreamName' in container:
                            log_stream_name = container['logStreamName']

                            if debug:
                                print(f"Debug: Found log stream name in node container: {log_stream_name}")

                            # Create a stream object
                            stream = {
                                'logStreamName': log_stream_name,
                                'logGroupName': '/aws/batch/job',
                                'attemptNum': -1  # Default attempt number
                            }

                            # Add to list if not already there
                            if not any(s.get('logStreamName') == stream.get('logStreamName') and
                                      s.get('logGroupName') == stream.get('logGroupName') for s in log_streams):
                                log_streams.append(stream)

            # Check the container details in the attempts
            for attempt in job.get('attempts', []):
                container = attempt.get('container', {})
                if 'logStreamName' in container:
                    log_stream_name = container['logStreamName']

                    if debug:
                        print(f"Debug: Found log stream name in attempt container: {log_stream_name}")

                    # Create a stream object
                    stream = {
                        'logStreamName': log_stream_name,
                        'logGroupName': '/aws/batch/job',
                        'attemptNum': job.get('attempts', []).index(attempt)  # Set attempt number based on index
                    }

                    # Add to list if not already there
                    if not any(s.get('logStreamName') == stream.get('logStreamName') and
                              s.get('logGroupName') == stream.get('logGroupName') for s in log_streams):
                        log_streams.append(stream)

            # For multi-node jobs, we need to search more broadly
            # Check if this is a multi-node job
            is_multi_node = False
            if 'nodeProperties' in job:
                num_nodes = job.get('nodeProperties', {}).get('numNodes', 0)
                if num_nodes > 1:
                    is_multi_node = True
                    if debug:
                        print(f"Debug: This is a multi-node job with {num_nodes} nodes")

            # If this is a multi-node job and we haven't found any log streams yet,
            # try to search for log streams with a broader pattern
            if is_multi_node and not log_streams and job_name:
                try:
                    logs = boto3.client('logs')

                    # Try to find log streams with the job name in the stream name
                    paginator = logs.get_paginator('describe_log_streams')

                    # Use pagination to get all log streams
                    for page in paginator.paginate(
                        logGroupName='/aws/batch/job',
                        logStreamNamePrefix='metta-batch-dist-train/default/'
                    ):
                        for stream in page.get('logStreams', []):
                            stream['logGroupName'] = '/aws/batch/job'  # Add log group name to stream info

                            # Check if the stream was created around the time the job was created
                            stream_creation_time = stream.get('creationTime', 0)
                            job_creation_time = job.get('createdAt', 0)

                            # If the stream was created within 5 minutes of the job, it's likely related
                            if abs(stream_creation_time - job_creation_time) < 300000:  # 5 minutes in milliseconds
                                if debug:
                                    print(f"Debug: Found log stream by creation time: {stream.get('logStreamName')}")

                                # Add to list if not already there
                                if not any(s.get('logStreamName') == stream.get('logStreamName') and
                                          s.get('logGroupName') == stream.get('logGroupName') for s in log_streams):
                                    stream['attemptNum'] = -1  # Default attempt number
                                    log_streams.append(stream)
                except Exception as e:
                    if debug:
                        print(f"Debug: Error searching for log streams by creation time: {str(e)}")

    except Exception as e:
        if debug:
            print(f"Debug: Error getting log stream names from job details: {str(e)}")

    # If we found log streams from the job details, return them
    if log_streams:
        # Filter by attempt index if specified
        if attempt_index is not None:
            log_streams = [stream for stream in log_streams if stream.get('attemptNum') == attempt_index]

        return log_streams

    # If we didn't find log streams from the job details, try the alternative search
    try:
        logs = boto3.client('logs')

        # Try alternative log groups
        alternative_log_groups = [
            f'/aws/batch/job/{job_id}',
            f'/aws/batch/{job_id}',
            '/aws/batch/job'  # Standard AWS Batch log group
        ]

        # If we have a job definition name, add its log group
        if job_definition_name:
            alternative_log_groups.insert(0, f'/aws/batch/{job_definition_name}')
            if debug:
                print(f"Debug: Adding job definition log group: /aws/batch/{job_definition_name}")

        # Try alternative patterns for log stream names
        alternative_patterns = [
            f"default/{job_id}",  # Standard AWS Batch pattern
            f"metta-batch-dist-train/default/{job_id}",  # Pattern seen in the CloudWatch console
            "metta-batch-dist-train/default/",  # Broader pattern to catch variations
            "metta-batch-jq-test/default/",  # Another queue pattern
            "metta-batch-jq-2/default/",  # Another queue pattern
            "batch-job/",  # Job definition log group pattern
            "default/"  # Most general pattern
        ]

        # For multi-node jobs, we need to search more broadly
        # Get job details to determine job name
        job_name = None
        try:
            batch = boto3.client('batch')
            response = batch.describe_jobs(jobs=[job_id])
            if response['jobs']:
                job_name = response['jobs'][0].get('jobName')
                if debug:
                    print(f"Debug: Job name is {job_name}")
        except Exception as e:
            if debug:
                print(f"Debug: Error getting job name: {str(e)}")

        for log_group in alternative_log_groups:
            try:
                for pattern in alternative_patterns:
                    try:
                        paginator = logs.get_paginator('describe_log_streams')

                        # Use pagination to get all log streams
                        for page in paginator.paginate(
                            logGroupName=log_group,
                            logStreamNamePrefix=pattern
                        ):
                            for stream in page.get('logStreams', []):
                                stream['logGroupName'] = log_group  # Add log group name to stream info
                                stream_name = stream.get('logStreamName', '')

                                # For broader patterns, we need to filter the results
                                if pattern in ["metta-batch-dist-train/default/", "metta-batch-jq-test/default/",
                                              "metta-batch-jq-2/default/", "batch-job/", "default/"]:
                                    # Check if the stream name contains the job ID
                                    if job_id in stream_name:
                                        # This is definitely related to our job
                                        pass
                                    # If we have a job name, check if the stream name contains it
                                    elif job_name and job_name in stream_name:
                                        # This is likely related to our job
                                        pass
                                    # For multi-node jobs, check if the stream was created around the same time as other streams
                                    elif log_streams:
                                        # Get the creation time of this stream
                                        this_time = stream.get('creationTime', 0)

                                        # Check if any existing stream was created within 5 minutes
                                        related = False
                                        for existing_stream in log_streams:
                                            existing_time = existing_stream.get('creationTime', 0)
                                            if abs(this_time - existing_time) < 300000:  # 5 minutes in milliseconds
                                                related = True
                                                break

                                        if not related:
                                            continue
                                    else:
                                        # If we don't have any streams yet and can't determine if this one is related,
                                        # we'll skip it for now
                                        continue

                                # Extract attempt information from the stream name if possible
                                if '/attempt/' in stream_name:
                                    try:
                                        attempt_num = int(stream_name.split('/attempt/')[1].split('/')[0])
                                        stream['attemptNum'] = attempt_num
                                    except (ValueError, IndexError):
                                        stream['attemptNum'] = -1
                                else:
                                    stream['attemptNum'] = -1

                                # Add to list if not already there
                                if not any(s.get('logStreamName') == stream.get('logStreamName') and
                                          s.get('logGroupName') == stream.get('logGroupName') for s in log_streams):
                                    log_streams.append(stream)
                    except Exception as e:
                        # Just continue to the next pattern if this one fails
                        continue

                # Filter by attempt index if specified
                if attempt_index is not None:
                    log_streams = [stream for stream in log_streams if stream.get('attemptNum') == attempt_index]

                if log_streams:
                    break  # Stop if we found logs in this group
            except logs.exceptions.ResourceNotFoundException:
                # Log group doesn't exist, try the next one
                if debug:
                    print(f"Debug: Log group {log_group} not found")
                continue
            except Exception as e:
                print(f"Error retrieving log streams from group {log_group}: {str(e)}")
    except Exception as e:
        print(f"Error finding alternative log streams for job {job_id}: {str(e)}")

    return log_streams

def format_time_difference(timestamp, end_timestamp=None):
    """
    Format the time difference between timestamps in a human-readable format.

    Args:
        timestamp (float): The start timestamp in seconds
        end_timestamp (float, optional): The end timestamp in seconds.
            If None, the current time is used.

    Returns:
        str: A human-readable time difference string
    """
    if not timestamp:
        return "Unknown"

    # Convert timestamp to datetime
    try:
        # If timestamp is in milliseconds (AWS often uses milliseconds)
        if timestamp > 1000000000000:
            timestamp = timestamp / 1000

        timestamp_dt = datetime.fromtimestamp(timestamp)

        if end_timestamp:
            # If end_timestamp is in milliseconds
            if end_timestamp > 1000000000000:
                end_timestamp = end_timestamp / 1000

            end_dt = datetime.fromtimestamp(end_timestamp)
        else:
            end_dt = datetime.now()

        # Calculate the time difference
        diff = end_dt - timestamp_dt

        # Extract days, hours, minutes, seconds
        days = diff.days
        seconds = diff.seconds
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60

        # Format the time difference
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception as e:
        print(f"Error formatting time difference: {str(e)}")
        return "Unknown"

def get_log_events(log_group, log_stream, start_time=None, tail=False):
    """
    Get log events from a specific log stream.

    Args:
        log_group (str): The CloudWatch log group name
        log_stream (str): The CloudWatch log stream name
        start_time (int, optional): The start time in milliseconds since epoch
        tail (bool): If True, only return the most recent events

    Returns:
        list: A list of log events
    """
    logs = boto3.client('logs')

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
                from datetime import datetime, timedelta
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

def print_job_logs(job_id, attempt_index=None, tail=False, debug=False):
    """
    Print logs for a specific job.

    Args:
        job_id (str): The job ID to get logs for
        attempt_index (int, optional): The specific attempt index to get logs for
        tail (bool): Whether to continuously poll for new logs
        debug (bool): Whether to show debug information

    Returns:
        bool: True if logs were found and printed, False otherwise
    """
    # Get job details
    jobs = get_job_details(job_id=job_id)
    if not jobs:
        print(f"No job found with ID {job_id}")
        return False

    job = jobs[0]
    job_name = job.get('jobName', 'Unknown')
    status = job.get('status', 'Unknown')

    print(f"\nShowing logs for job {job_name} (ID: {job_id}):")

    # Check if this is a multi-node job
    is_multi_node = False
    if 'nodeProperties' in job:
        num_nodes = job.get('nodeProperties', {}).get('numNodes', 0)
        if num_nodes > 1:
            is_multi_node = True
            if debug:
                print(f"Debug: This is a multi-node job with {num_nodes} nodes")

    # If the job is in RUNNABLE or SUBMITTED state, logs might not be available yet
    if status in ['RUNNABLE', 'SUBMITTED', 'PENDING']:
        print(f"Job is in {status} state. Logs may not be available until the job starts running.")

    # Get job attempts
    attempts = get_job_attempts(job_id)

    if attempts:
        print(f"\nJob has {len(attempts)} attempt(s):")
        for i, attempt in enumerate(attempts):
            container = attempt.get('container', {})
            exit_code = container.get('exitCode', 'N/A')
            reason = container.get('reason', 'N/A')

            # Format timestamps
            started_at = attempt.get('startedAt')
            if started_at:
                started_at_str = datetime.fromtimestamp(started_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
            else:
                started_at_str = 'N/A'

            stopped_at = attempt.get('stoppedAt')
            if stopped_at:
                stopped_at_str = datetime.fromtimestamp(stopped_at / 1000).strftime('%Y-%m-%d %H:%M:%S')
                duration = format_time_difference(started_at / 1000, stopped_at / 1000) if started_at else 'N/A'
            else:
                stopped_at_str = 'N/A'
                duration = 'Running' if started_at else 'N/A'

            print(f"Attempt {i+1}: Started: {started_at_str}, Stopped: {stopped_at_str}, Duration: {duration}, Exit Code: {exit_code}, Reason: {reason}")

    # Get log streams for the job
    if debug:
        print(f"\nDebug: Searching for log streams for job {job_id}" +
              (f" attempt {attempt_index}" if attempt_index is not None else ""))

    log_streams = get_job_log_streams(job_id, attempt_index=attempt_index)

    if debug and log_streams:
        print(f"Debug: Found {len(log_streams)} log streams in primary search:")
        for i, stream in enumerate(log_streams):
            print(f"  {i+1}. {stream.get('logGroupName')}/{stream.get('logStreamName')}")

    if not log_streams:
        # If no logs found in default group, try alternative log groups
        if debug:
            print("Debug: No log streams found in primary search, trying alternative search...")

        log_streams = find_alternative_log_streams(job_id, attempt_index=attempt_index, debug=debug)

        if debug and log_streams:
            print(f"Debug: Found {len(log_streams)} log streams in alternative search:")
            for i, stream in enumerate(log_streams):
                print(f"  {i+1}. {stream.get('logGroupName')}/{stream.get('logStreamName')}")

    if not log_streams:
        print(f"No log streams found for job {job_id}" +
              (f" attempt {attempt_index}" if attempt_index is not None else "") + ".")

        # Provide more information based on job status
        if status == 'SUCCEEDED':
            print("The job has completed successfully, but no logs were found. This can happen if the job didn't produce any output.")
        elif status == 'FAILED':
            print("The job has failed, but no logs were found. This can happen if the job failed before producing any output.")
            print("You might want to check the job definition and container configuration.")
        elif status in ['RUNNING', 'STARTING']:
            print("The job is running, but logs might not be available yet. Try again in a few moments.")

        # In debug mode, suggest checking the CloudWatch console directly
        if debug:
            print("\nDebug: Try checking logs directly in the CloudWatch console:")
            print(f"https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#logsV2:log-groups/log-group/%2Faws%2Fbatch%2Fjob/log-events/metta-batch-dist-train%2Fdefault%2F{job_id}")

        return False

    # Check if this is likely a multi-node job based on the number of log streams
    if not is_multi_node and len(log_streams) > 3:  # Arbitrary threshold, adjust as needed
        is_multi_node = True

    if is_multi_node and debug:
        print(f"\nDebug: This appears to be a multi-node job with {len(log_streams)} log streams")

    # Sort log streams by name
    log_streams.sort(key=lambda x: x.get('logStreamName', ''))

    # For multi-node jobs, we'll collect all events first and then sort them by timestamp
    if is_multi_node and not tail:
        all_events = []
        empty_streams = []

        # Get events from all streams
        for stream in log_streams:
            stream_name = stream.get('logStreamName', '')
            log_group_name = stream.get('logGroupName', '/aws/batch/job')

            # Get log events
            events = get_log_events(log_group_name, stream_name, tail=False)

            # Add stream information to each event
            for event in events:
                event['streamName'] = stream_name
                event['logGroupName'] = log_group_name

            if events:
                all_events.extend(events)
            else:
                empty_streams.append(stream)

        # Sort all events by timestamp
        all_events.sort(key=lambda x: x.get('timestamp', 0))

        # Print all events in chronological order
        if all_events:
            print(f"\nShowing logs from all {len(log_streams) - len(empty_streams)} non-empty streams in chronological order:")
            print("-" * 80)

            for event in all_events:
                timestamp = datetime.fromtimestamp(event.get('timestamp', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                message = event.get('message', '')
                stream_name = event.get('streamName', '')

                # Extract a short stream identifier for display
                stream_id = stream_name.split('/')[-1][:8] if '/' in stream_name else stream_name[:8]

                print(f"[{timestamp}] [{stream_id}] {message}")
        else:
            print("No log events found in any stream.")

            if debug and empty_streams:
                print(f"\nDebug: Found {len(empty_streams)} empty log streams:")
                for i, stream in enumerate(empty_streams):
                    print(f"  {i+1}. {stream.get('logGroupName')}/{stream.get('logStreamName')}")

                # For multi-node jobs, it's common to have empty log streams
                if is_multi_node:
                    print("\nThis is a multi-node job, and it's common for some nodes to not produce logs.")
                    print("The job may have completed successfully even if some log streams are empty.")

        return True

    # For single-node jobs or when tailing, process streams individually
    for stream in log_streams:
        stream_name = stream.get('logStreamName', '')
        log_group_name = stream.get('logGroupName', '/aws/batch/job')
        attempt_num = stream.get('attemptNum', -1)

        attempt_info = f" (Attempt {attempt_num})" if attempt_num > 0 else ""
        print(f"\nLog stream: {log_group_name}/{stream_name}{attempt_info}")
        print("-" * 80)

        # Get log events
        events = get_log_events(log_group_name, stream_name, tail=tail)

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
            next_token = None

            try:
                while True:
                    import time
                    time.sleep(2)  # Poll every 2 seconds

                    params = {
                        'logGroupName': log_group_name,
                        'logStreamName': stream_name
                    }

                    if next_token:
                        params['nextToken'] = next_token

                    response = boto3.client('logs').get_log_events(**params)

                    events = response.get('events', [])
                    for event in events:
                        timestamp = datetime.fromtimestamp(event.get('timestamp', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')
                        message = event.get('message', '')
                        print(f"[{timestamp}] {message}")

                    next_token = response.get('nextForwardToken')
            except KeyboardInterrupt:
                print("\nStopped tailing logs.")

    return True
