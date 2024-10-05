import boto3
import argparse
from devops.aws.cluster_info import get_batch_job_queues, get_batch_jobs

def stop_batch_jobs(job_prefix):
    batch = boto3.client('batch')
    job_queues = get_batch_job_queues()

    for queue in job_queues:
        jobs = get_batch_jobs(queue, max_jobs=1000)  # Increase max_jobs to get all jobs
        for job in jobs:
            if job['status'] == 'RUNNING' and job['name'].startswith(job_prefix):
                job_id = job['stop_command'].split()[-1]
                print(f"Stopping job: {job['name']} (ID: {job_id})")
                try:
                    batch.terminate_job(jobId=job_id, reason=f'Stopped by stop_jobs script (prefix: {job_prefix})')
                    print(f"Successfully stopped job: {job['name']}")
                except Exception as e:
                    print(f"Failed to stop job {job['name']}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stop running AWS Batch jobs with a specific prefix.')
    parser.add_argument('--job_prefix', type=str, required=True, help='Prefix of job names to stop')
    args = parser.parse_args()
    args.job_prefix = args.job_prefix.replace('.', '_')

    stop_batch_jobs(args.job_prefix)
    print(f"WARNING: This will stop all running AWS Batch jobs that start with '{args.job_prefix}'.")
