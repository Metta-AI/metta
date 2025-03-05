"""
Examples of how to use the AWS Batch job utilities.
"""

import argparse
from datetime import datetime
from devops.aws.batch.job import (
    get_batch_job_queues,
    get_job_details,
    get_job_log_streams,
    find_alternative_log_streams,
    format_time_difference
)

def list_job_queues():
    """List all available AWS Batch job queues."""
    queues = get_batch_job_queues()

    print("\nAvailable Job Queues:")
    print("---------------------")
    for i, queue in enumerate(queues, 1):
        print(f"{i}. {queue}")

    return queues

def list_jobs_in_queue(queue_name, max_jobs=10):
    """List jobs in a specific queue."""
    jobs = get_job_details(job_queue=queue_name, max_jobs=max_jobs)

    # Sort jobs by creation time (newest first)
    jobs.sort(key=lambda x: x.get('createdAt', 0), reverse=True)

    if not jobs:
        print(f"No jobs found in queue '{queue_name}'.")
        return []

    print(f"\nJobs in Queue '{queue_name}' (showing up to {max_jobs}, newest first):")
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

    return jobs

def get_job_info(job_id_or_name):
    """Get information about a specific job by ID or name."""
    # Check if the input looks like a job ID
    is_job_id = job_id_or_name.startswith('job-') or job_id_or_name.startswith('aws:') or (len(job_id_or_name) >= 32 and '-' in job_id_or_name)

    if is_job_id:
        # Search by job ID
        jobs = get_job_details(job_id=job_id_or_name)
    else:
        # Search by job name
        jobs = get_job_details(job_prefix=job_id_or_name)

    if not jobs:
        print(f"No jobs found matching '{job_id_or_name}'.")
        return None

    # Sort jobs by creation time (newest first)
    jobs.sort(key=lambda x: x.get('createdAt', 0), reverse=True)

    print(f"\nJobs Matching '{job_id_or_name}':")
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

    # Print detailed information about the first job
    if jobs:
        job = jobs[0]
        print("\nDetailed Information for First Job:")
        print(f"Job Name: {job.get('jobName', 'Unknown')}")
        print(f"Job ID: {job.get('jobId', 'Unknown')}")
        print(f"Status: {job.get('status', 'Unknown')}")
        print(f"Created At: {datetime.fromtimestamp(job.get('createdAt', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Age: {format_time_difference(job.get('createdAt', 0) / 1000)}")

        # Print job definition
        print(f"Job Definition: {job.get('jobDefinition', 'Unknown')}")

        # Print container details if available
        container = job.get('container', {})
        if container:
            print("\nContainer Information:")
            print(f"Image: {container.get('image', 'Unknown')}")
            print(f"Command: {' '.join(container.get('command', []))}")
            print(f"Exit Code: {container.get('exitCode', 'N/A')}")
            print(f"Reason: {container.get('reason', 'N/A')}")

        # Print log streams
        log_streams = get_job_log_streams(job.get('jobId'))
        if log_streams:
            print("\nLog Streams:")
            for i, stream in enumerate(log_streams, 1):
                print(f"{i}. {stream.get('logGroupName', '/aws/batch/job')}/{stream.get('logStreamName', 'Unknown')}")
        else:
            print("\nNo log streams found.")

    return jobs

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='AWS Batch Job Utilities Examples')
    parser.add_argument('--list-queues', action='store_true', help='List all available job queues')
    parser.add_argument('--queue', help='Job queue to list jobs from')
    parser.add_argument('--job', help='Job ID or name to get information about')
    parser.add_argument('--max-jobs', type=int, default=10, help='Maximum number of jobs to list')
    args = parser.parse_args()

    if args.list_queues:
        list_job_queues()
    elif args.queue:
        list_jobs_in_queue(args.queue, args.max_jobs)
    elif args.job:
        get_job_info(args.job)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
