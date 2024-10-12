import boto3
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError
from devops.aws.cluster_info import get_batch_job_queues, get_batch_jobs

def stop_jobs_batch(batch, jobs, job_prefix):
    jobs_to_stop = [job for job in jobs if job['status'] == 'RUNNING' and job['name'].startswith(job_prefix)]
    if not jobs_to_stop:
        return []

    results = []
    job_ids = [job['stop_command'].split()[-1] for job in jobs_to_stop]

    max_retries = 5
    retry_delay = 1
    for attempt in range(max_retries):
        try:
            response = batch.terminate_jobs(
                jobIds=job_ids,
                reason=f'Stopped by stop_jobs script (prefix: {job_prefix})'
            )

            for job, result in zip(jobs_to_stop, response['ResponseMetadata']['HTTPStatusCode']):
                if result == 200:
                    results.append(f"Successfully stopped job: {job['name']} (ID: {job['stop_command'].split()[-1]})")
                else:
                    results.append(f"Failed to stop job {job['name']} (ID: {job['stop_command'].split()[-1]}): HTTP {result}")

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

def stop_batch_jobs(job_prefix, job_queue=None):
    batch = boto3.client('batch')
    if job_queue:
        job_queues = [job_queue]
    else:
        job_queues = get_batch_job_queues()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for queue in job_queues:
            jobs = get_batch_jobs(queue, max_jobs=1000)
            futures.append(executor.submit(stop_jobs_batch, batch, jobs, job_prefix))

        for future in as_completed(futures):
            results = future.result()
            for result in results:
                print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stop running AWS Batch jobs with a specific prefix.')
    parser.add_argument('--job_prefix', type=str, required=True, help='Prefix of job names to stop')
    parser.add_argument('--job_queue', type=str, default="metta-batch-jq-g6-8xlarge", help='Specific job queue to stop jobs from')
    args = parser.parse_args()
    args.job_prefix = args.job_prefix.replace('.', '_')

    stop_batch_jobs(args.job_prefix, args.job_queue)
