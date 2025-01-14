import boto3
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError
from devops.aws.cluster_info import get_batch_job_queues, get_batch_jobs

def stop_jobs_batch(batch, jobs, job_prefix, silent=False):
    jobs_to_stop = [job for job in jobs if job['status'] == 'RUNNING' and job['name'].startswith(job_prefix)]
    if not jobs_to_stop:
        return []

    if not silent:
        print("\nJobs that will be stopped:")
        print("-" * 80)
        for job in jobs_to_stop:
            print(f"Name: {job['name']}")
            print(f"ID: {job['stop_command'].split()[-1]}")
            print(f"Status: {job['status']}")
            print("-" * 80)

        confirm = input("\nDo you want to proceed with stopping these jobs? (y/N): ")
        if confirm.lower() != 'y':
            print("Operation cancelled")
            return []

    results = []
    job_ids = [job['stop_command'].split()[-1] for job in jobs_to_stop]

    max_retries = 5
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            # Process jobs in batches of 100
            for i in range(0, len(job_ids), 100):
                batch_job_ids = job_ids[i:i+100]
                response = batch.terminate_job(
                    jobId=batch_job_ids[0],
                    reason=f'Stopped by stop_jobs script (prefix: {job_prefix})'
                )
                for job_id in batch_job_ids[1:]:
                    batch.terminate_job(
                        jobId=job_id,
                        reason=f'Stopped by stop_jobs script (prefix: {job_prefix})'
                    )

                for job in jobs_to_stop[i:i+100]:
                    results.append(f"Successfully stopped job: {job['name']} (ID: {job['stop_command'].split()[-1]})")

            return results

        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return [f"Failed to stop jobs after {max_retries} attempts: Throttling"]
            else:
                return [f"Failed to stop jobs: {str(e)}"]
        except Exception as e:
            return [f"Failed to stop jobs: {str(e)}"]

def stop_batch_jobs(job_prefix, job_queue=None, silent=False):
    batch = boto3.client('batch')
    if job_queue:
        job_queues = [job_queue]
    else:
        job_queues = get_batch_job_queues()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for queue in job_queues:
            jobs = get_batch_jobs(queue, max_jobs=100)  # Changed max_jobs to 100
            futures.append(executor.submit(stop_jobs_batch, batch, jobs, job_prefix, silent))

        for future in as_completed(futures):
            results = future.result()
            for result in results:
                print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stop running AWS Batch jobs with a specific prefix.')
    parser.add_argument('--job_prefix', type=str, required=True, help='Prefix of job names to stop')
    parser.add_argument('--job_queue', type=str, default="metta-batch-jq-g6-8xlarge", help='Specific job queue to stop jobs from')
    parser.add_argument('--silent', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    args.job_prefix = args.job_prefix.replace('.', '_')

    stop_batch_jobs(args.job_prefix, args.job_queue, args.silent)
