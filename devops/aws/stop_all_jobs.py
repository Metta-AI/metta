import boto3
import argparse
from devops.aws.cluster_info import get_batch_job_queues, get_batch_jobs

def stop_all_batch_jobs():
    batch = boto3.client('batch')
    job_queues = get_batch_job_queues()

    for queue in job_queues:
        jobs = get_batch_jobs(queue, max_jobs=1000)  # Increase max_jobs to get all jobs
        for job in jobs:
            if job['status'] == 'RUNNING':
                job_id = job['stop_command'].split()[-1]
                print(f"Stopping job: {job['name']} (ID: {job_id})")
                try:
                    batch.terminate_job(jobId=job_id, reason='Stopped by stop_all_jobs script')
                    print(f"Successfully stopped job: {job['name']}")
                except Exception as e:
                    print(f"Failed to stop job {job['name']}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stop all running AWS Batch jobs.')
    parser.add_argument('--confirm', action='store_true', help='Confirm that you want to stop all jobs')
    args = parser.parse_args()

    if args.confirm:
        stop_all_batch_jobs()
    else:
        print("Please run the script with --confirm to stop all jobs.")
        print("WARNING: This will stop all running AWS Batch jobs.")
