import boto3
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError
from devops.aws.cluster_info import get_batch_job_queues, get_batch_jobs

def stop_job(batch, job, job_prefix):
    if job['status'] == 'RUNNING' and job['name'].startswith(job_prefix):
        job_id = job['stop_command'].split()[-1]
        max_retries = 5
        retry_delay = 1
        for attempt in range(max_retries):
            try:
                batch.terminate_job(jobId=job_id, reason=f'Stopped by stop_jobs script (prefix: {job_prefix})')
                return f"Successfully stopped job: {job['name']} (ID: {job_id})"
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        return f"Failed to stop job {job['name']} (ID: {job_id}) after {max_retries} attempts: Throttling"
                else:
                    return f"Failed to stop job {job['name']} (ID: {job_id}): {str(e)}"
            except Exception as e:
                return f"Failed to stop job {job['name']} (ID: {job_id}): {str(e)}"
    return None

def stop_batch_jobs(job_prefix):
    batch = boto3.client('batch')
    job_queues = get_batch_job_queues()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for queue in job_queues:
            jobs = get_batch_jobs(queue, max_jobs=1000)
            for job in jobs:
                futures.append(executor.submit(stop_job, batch, job, job_prefix))

        for future in as_completed(futures):
            result = future.result()
            if result:
                print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stop running AWS Batch jobs with a specific prefix.')
    parser.add_argument('--job_prefix', type=str, required=True, help='Prefix of job names to stop')
    args = parser.parse_args()
    args.job_prefix = args.job_prefix.replace('.', '_')

    stop_batch_jobs(args.job_prefix)
