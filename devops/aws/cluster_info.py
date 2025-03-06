import boto3
import argparse
from colorama import init, Fore, Style
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.config import Config

# Configure boto3 to use a higher max_pool_connections
config = Config(
    retries = {'max_attempts': 10, 'mode': 'standard'},
    max_pool_connections = 50
)

def get_batch_job_queues():
    batch = boto3.client('batch', config=config)
    response = batch.describe_job_queues()
    return [queue['jobQueueName'] for queue in response['jobQueues']]

def get_batch_jobs(job_queue, max_jobs):
    batch = boto3.client('batch', config=config)
    ecs = boto3.client('ecs', config=config)
    ec2 = boto3.client('ec2', config=config)

    def get_jobs_by_status(status):
        response = batch.list_jobs(jobQueue=job_queue, jobStatus=status, maxResults=min(max_jobs, 100))
        return response['jobSummaryList']

    with ThreadPoolExecutor() as executor:
        job_futures = [executor.submit(get_jobs_by_status, state) for state in ['RUNNING', 'SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'SUCCEEDED', 'FAILED']]
        all_jobs = [job for future in job_futures for job in future.result()]

    all_jobs = sorted(all_jobs, key=lambda job: job['createdAt'], reverse=True)[:max_jobs]

    job_ids = [job['jobId'] for job in all_jobs]
    job_descriptions = batch.describe_jobs(jobs=job_ids)['jobs']

    task_arns = []
    cluster_arns = []
    task_to_cluster = {}  # Map to track which cluster each task belongs to
    for job in job_descriptions:
        container = job.get('container', {})
        if container and container.get('taskArn'):
            task_arn = container.get('taskArn')
            cluster_arn = container.get('containerInstanceArn')
            if cluster_arn:  # Only add if we have cluster information
                cluster_name = cluster_arn.split('/')[1]
                task_arns.append(task_arn)
                cluster_arns.append(cluster_arn)
                task_to_cluster[task_arn] = cluster_name

    # Skip task processing if there are no tasks
    if not task_arns:
        return [get_job_details(job) for job in all_jobs]

    # Batch describe tasks by their respective clusters
    tasks_by_cluster = {}
    for task_arn in task_arns:
        cluster_name = task_to_cluster.get(task_arn)
        if cluster_name:
            tasks_by_cluster.setdefault(cluster_name, []).append(task_arn)

    task_descriptions = {}
    for cluster_name, tasks in tasks_by_cluster.items():
        for i in range(0, len(tasks), 100):
            chunk = tasks[i:i+100]
            try:
                response = ecs.describe_tasks(cluster=cluster_name, tasks=chunk)
                task_descriptions.update({task['taskArn']: task for task in response['tasks']})
            except Exception as e:
                print(f"Warning: Failed to describe tasks for cluster {cluster_name}: {str(e)}")
                continue

    container_instances_by_cluster = {}
    for task in task_descriptions.values():
        cluster_name = task['clusterArn'].split('/')[1]
        container_instances_by_cluster.setdefault(cluster_name, set()).add(task['containerInstanceArn'])

    container_instance_descriptions = {}
    for cluster_name, container_instances in container_instances_by_cluster.items():
        for i in range(0, len(container_instances), 100):
            chunk = list(container_instances)[i:i+100]
            response = ecs.describe_container_instances(cluster=cluster_name, containerInstances=chunk)
            container_instance_descriptions.update({instance['containerInstanceArn']: instance for instance in response['containerInstances']})

    # Batch describe EC2 instances
    ec2_instance_ids = [instance['ec2InstanceId'] for instance in container_instance_descriptions.values()]
    ec2_instances = {}
    for i in range(0, len(ec2_instance_ids), 100):
        chunk = ec2_instance_ids[i:i+100]
        response = ec2.describe_instances(InstanceIds=chunk)
        ec2_instances.update({instance['InstanceId']: instance for reservation in response['Reservations'] for instance in reservation['Instances']})

    def get_job_details(job):
        job_id = job['jobId']
        job_name = job['jobName']
        job_status = job['status']
        job_link = f"https://console.aws.amazon.com/batch/home?region=us-east-1#jobs/detail/{job_id}"

        job_desc = next((j for j in job_descriptions if j['jobId'] == job_id), {})
        container = job_desc.get('container', {})
        task_arn = container.get('taskArn')

        num_retries = len(job_desc.get('attempts', [])) - 1 if job_desc.get('attempts') else 0

        public_ip = ''
        if task_arn and task_arn in task_descriptions:
            task_desc = task_descriptions.get(task_arn, {})
            container_instance_arn = task_desc.get('containerInstanceArn')
            if container_instance_arn and container_instance_arn in container_instance_descriptions:
                container_instance_desc = container_instance_descriptions.get(container_instance_arn, {})
                ec2_instance_id = container_instance_desc.get('ec2InstanceId')
                if ec2_instance_id and ec2_instance_id in ec2_instances:
                    ec2_instance = ec2_instances.get(ec2_instance_id, {})
                    public_ip = ec2_instance.get('PublicIpAddress', '')

        stop_command = f"aws batch terminate-job --reason man_stop --job-id {job_id}" if job_status in ['RUNNING', 'RUNNABLE', 'STARTING'] else ''

        return {
            'name': job_name,
            'status': job_status,
            'retries': num_retries,
            'link': job_link,
            'public_ip': public_ip,
            'stop_command': stop_command
        }

    with ThreadPoolExecutor() as executor:
        job_details = list(executor.map(get_job_details, all_jobs))

    return job_details

def get_ecs_clusters():
    ecs = boto3.client('ecs', config=config)
    response = ecs.list_clusters()
    return response['clusterArns']

def get_ecs_tasks(clusters, max_tasks):
    ecs = boto3.client('ecs', config=config)
    ec2 = boto3.client('ec2', config=config)

    all_tasks = []
    for cluster in clusters:
        response = ecs.list_tasks(cluster=cluster, maxResults=min(max_tasks, 100))
        all_tasks.extend([(cluster, task_arn) for task_arn in response['taskArns']])

    all_tasks = all_tasks[:max_tasks]

    task_descriptions = {}
    for i in range(0, len(all_tasks), 100):
        chunk = all_tasks[i:i+100]
        cluster = chunk[0][0]  # Assuming all tasks in the chunk are from the same cluster
        task_arns = [task[1] for task in chunk]
        response = ecs.describe_tasks(cluster=cluster, tasks=task_arns)
        task_descriptions.update({task['taskArn']: task for task in response['tasks']})

    container_instances = set(task['containerInstanceArn'] for task in task_descriptions.values())
    container_instance_descriptions = {}
    for i in range(0, len(container_instances), 100):
        chunk = list(container_instances)[i:i+100]
        response = ecs.describe_container_instances(cluster=cluster, containerInstances=chunk)
        container_instance_descriptions.update({instance['containerInstanceArn']: instance for instance in response['containerInstances']})

    ec2_instance_ids = [instance['ec2InstanceId'] for instance in container_instance_descriptions.values()]
    ec2_instances = {}
    for i in range(0, len(ec2_instance_ids), 100):
        chunk = ec2_instance_ids[i:i+100]
        response = ec2.describe_instances(InstanceIds=chunk)
        ec2_instances.update({instance['InstanceId']: instance for reservation in response['Reservations'] for instance in reservation['Instances']})

    def get_task_details(cluster, task_arn):
        task_desc = task_descriptions[task_arn]
        task_id = task_arn.split('/')[-1]
        task_name = task_desc['overrides']['containerOverrides'][0]['name']
        task_status = task_desc['lastStatus']
        task_link = f"https://console.aws.amazon.com/ecs/home?region=us-east-1#/clusters/{cluster}/tasks/{task_id}/details"

        container_instance_arn = task_desc['containerInstanceArn']
        container_instance_desc = container_instance_descriptions[container_instance_arn]
        ec2_instance_id = container_instance_desc['ec2InstanceId']

        public_ip = ec2_instances[ec2_instance_id].get('PublicIpAddress', '')

        stop_command = f"aws ecs stop-task --cluster {cluster} --task {task_arn}" if task_status == 'RUNNING' else ''

        return {
            'name': task_name,
            'status': task_status,
            'retries': 0,
            'link': task_link,
            'public_ip': public_ip,
            'stop_command': stop_command
        }

    with ThreadPoolExecutor() as executor:
        task_details = list(executor.map(lambda x: get_task_details(*x), all_tasks))

    return task_details

def print_row(key, value, use_color):
    if use_color:
        print(f"  {Fore.BLUE}{key}:{Style.RESET_ALL} {value}")
    else:
        print(f"  {key}: {value}")

def print_status(jobs_by_queue, tasks, use_color):
    for job_queue, jobs in jobs_by_queue.items():
        if use_color:
            print(f"{Fore.CYAN}AWS Batch Jobs - Queue: {job_queue}{Style.RESET_ALL}")
        else:
            print(f"AWS Batch Jobs - Queue: {job_queue}")

        for job in jobs:
            status_color = {
                'SUBMITTED': Fore.YELLOW,
                'PENDING': Fore.YELLOW,
                'RUNNABLE': Fore.YELLOW,
                'STARTING': Fore.YELLOW,
                'RUNNING': Fore.GREEN,
                'SUCCEEDED': Fore.GREEN,
                'FAILED': Fore.RED
            }.get(job['status'], Fore.YELLOW)

            print_row("Name", job['name'], use_color)
            print_row("Status", f"{status_color}{job['status']}{Style.RESET_ALL} ({job['retries']})" if use_color else job['status'], use_color)
            print_row("Link", job['link'], use_color)
            print_row("Public IP", job['public_ip'], use_color)
            print_row("Stop Command", job['stop_command'], use_color)
            print()

    if tasks:
        if use_color:
            print(f"{Fore.CYAN}ECS Tasks{Style.RESET_ALL}")
        else:
            print("ECS Tasks")

        for task in tasks:
            status_color = Fore.GREEN if task['status'] == 'RUNNING' else Fore.RED
            print_row("Name", task['name'], use_color)
            print_row("Status", f"{status_color}{task['status']}{Style.RESET_ALL}" if use_color else task['status'], use_color)
            print_row("Link", task['link'], use_color)
            print_row("Public IP", task['public_ip'], use_color)
            print_row("Stop Command", task['stop_command'], use_color)
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get the status of AWS Batch jobs and ECS tasks.')
    parser.add_argument('--max-jobs', type=int, default=10, help='The maximum number of jobs to display.')
    parser.add_argument('--ecs', action='store_true', help='Include ECS tasks in the status dump.')
    parser.add_argument('--no-color', action='store_true', help='Disable color output.')
    args = parser.parse_args()

    init()  # Initialize colorama

    job_queues = get_batch_job_queues()
    args.queue = f"metta-batch-jq-2"

    selected_queues = [queue for queue in job_queues if args.queue in queue]

    with ThreadPoolExecutor() as executor:
        jobs_by_queue = {queue: executor.submit(get_batch_jobs, queue, args.max_jobs) for queue in selected_queues}
        jobs_by_queue = {queue: future.result() for queue, future in jobs_by_queue.items()}

    ecs_tasks = []
    if args.ecs:
        ecs_clusters = get_ecs_clusters()
        ecs_tasks = get_ecs_tasks(ecs_clusters, args.max_jobs)

    print_status(jobs_by_queue, ecs_tasks, not args.no_color)
