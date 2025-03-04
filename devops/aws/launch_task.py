import argparse
import netrc
import os
import random
import string
import sys
import subprocess

import boto3

def get_current_commit(repo_path=None):
    """Get the current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "rev-parse", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def submit_batch_job(args, task_args):
    session_kwargs = {'region_name': 'us-east-1'}
    if args.profile:
        session_kwargs['profile_name'] = args.profile

    # Create a new session with the specified profile
    session = boto3.Session(**session_kwargs)
    batch = session.client('batch')

    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    job_name = args.run.replace('.', '_') + "_" + random_id
    job_queue = args.job_queue

    # Always use the distributed job definition
    job_definition = "metta-batch-dist-train"

    # Submit multi-node job
    response = batch.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        nodeOverrides={
            'nodePropertyOverrides': [
                {
                    'targetNodes': '0:',
                    'containerOverrides': container_config(args, task_args, job_name)
                }
            ],
            'numNodes': args.num_nodes
        }
    )

    print(f"Submitted job {job_name} to queue {job_queue} with job ID {response['jobId']}")
    print(f"https://us-east-1.console.aws.amazon.com/batch/v2/home?region=us-east-1#/jobs/detail/{response['jobId']}")

def container_config(args, task_args, job_name):
    try:
        # Get the wandb key from the .netrc file
        netrc_info = netrc.netrc(os.path.expanduser('~/.netrc'))
        wandb_key = netrc_info.authenticators('api.wandb.ai')[2]
        if not wandb_key:
            raise ValueError('WANDB_API_KEY not found in .netrc file')
    except (FileNotFoundError, TypeError):
        print("Error: Could not find WANDB_API_KEY in ~/.netrc file")
        print("Please ensure you have a valid ~/.netrc file with api.wandb.ai credentials")
        sys.exit(1)

    # Calculate resource requirements
    vcpus_per_gpu = args.gpu_cpus * 2
    total_vcpus = vcpus_per_gpu * args.node_gpus

    # Memory in GB, convert to MB for AWS Batch API
    memory_gb = int(args.cpu_ram_gb)
    memory_mb = memory_gb * 1024

    # Calculate shared memory size as 90% of available memory
    shared_memory_mb = int(memory_mb * 0.9)

    # Set up environment variables for distributed training
    env_vars = [
        {
            'name': 'HYDRA_FULL_ERROR',
            'value': '1'
        },
        {
            'name': 'WANDB_API_KEY',
            'value': wandb_key
        },
        {
            'name': 'WANDB_SILENT',
            'value': 'true'
        },
        {
            'name': 'COLOR_LOGGING',
            'value': 'false'
        },
        {
            'name': 'WANDB_HOST',
            'value': job_name
        },
        {
            'name': 'METTA_HOST',
            'value': job_name
        },
        {
            'name': 'METTA_USER',
            'value': os.environ.get('USER', 'unknown')
        },
        {
            'name': 'NCCL_DEBUG',
            'value': 'INFO'
        },
        {
            'name': 'SHARED_MEMORY_SIZE',
            'value': str(shared_memory_mb)
        },
    ]

    # Calculate num_workers based on available vCPUs
    num_workers = min(6, vcpus_per_gpu // 2)  # 2 vCPU per worker

    # Add required environment variables for the entrypoint script
    env_vars.extend([
        {
            'name': 'RUN_ID',
            'value': args.run
        },
        {
            'name': 'CMD',
            'value': args.cmd
        },
        {
            'name': 'NUM_GPUS',
            'value': str(args.node_gpus)
        },
        {
            'name': 'NUM_WORKERS',
            'value': str(num_workers)
        }
    ])

    # Add git reference (branch or commit)
    if args.git_branch is not None:
        env_vars.append({
            'name': 'GIT_REF',
            'value': args.git_branch
        })
    elif args.git_commit is not None:
        env_vars.append({
            'name': 'GIT_REF',
            'value': args.git_commit
        })

    # Add mettagrid reference (branch or commit)
    if args.mettagrid_branch is not None:
        env_vars.append({
            'name': 'METTAGRID_REF',
            'value': args.mettagrid_branch
        })
    elif args.mettagrid_commit is not None:
        env_vars.append({
            'name': 'METTAGRID_REF',
            'value': args.mettagrid_commit
        })

    # Add task args if any
    if task_args:
        env_vars.append({
            'name': 'TASK_ARGS',
            'value': ' '.join(task_args)
        })

    # Build the command to run the entrypoint script
    entrypoint_cmd = []

    # Check out the git reference (branch or commit) before pulling
    entrypoint_cmd.append('git checkout fetch')
    if args.git_branch is not None:
        entrypoint_cmd.append(f'git checkout {args.git_branch}')
    elif args.git_commit is not None:
        entrypoint_cmd.append(f'git checkout {args.git_commit}')

    # Update the repository from the current branch
    entrypoint_cmd.append('git pull')

    # Run the entrypoint script
    entrypoint_cmd.append('./devops/aws/batch/train_entrypoint.sh')

    print("\n".join([
            "Setup:",
            "-"*10,
            "Using train_entrypoint.sh script with environment variables",
            "-"*10,
            "Command:",
            "-"*10,
            " ".join(entrypoint_cmd),
            "-"*10,
            f"Resources: {args.num_nodes} nodes, {args.node_gpus} GPUs, {total_vcpus} vCPUs ({vcpus_per_gpu} per GPU), {memory_gb}GB RAM, {shared_memory_mb}MB shared memory"
        ]))

    # Create resource requirements
    resource_requirements = [
        {
            'type': 'GPU',
            'value': str(args.node_gpus)
        }
    ]

    # Add vCPU and memory requirements
    resource_requirements.append({
        'type': 'VCPU',
        'value': str(total_vcpus)
    })

    resource_requirements.append({
        'type': 'MEMORY',
        'value': str(memory_mb)  # AWS Batch API expects MB
    })

    return {
        'command': ["bash", "-c", "; ".join(entrypoint_cmd)],
        'environment': env_vars,
        'resourceRequirements': resource_requirements
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch an AWS Batch task with a wandb key.')
    parser.add_argument('--cluster', default="metta", help='The name of the ECS cluster.')
    parser.add_argument('--run', required=True, help='The run id.')
    parser.add_argument('--cmd', required=True, choices=["train", "sweep", "evolve"], help='The command to run.')

    parser.add_argument('--git-branch', default=None, help='The git branch to use for the task. If not specified, will use the current commit.')
    parser.add_argument('--git-commit', default=None, help='The git commit to use for the task. If not specified, will use the current commit.')
    parser.add_argument('--mettagrid-branch', default=None, help='The mettagrid branch to use for the task. If not specified, will use the current commit.')
    parser.add_argument('--mettagrid-commit', default=None, help='The mettagrid commit to use for the task. If not specified, will use the current commit.')
    parser.add_argument('--gpus', type=int, default=4, help='Number of GPUs per node to use for the task.')
    parser.add_argument('--node-gpus', type=int, default=4, help='Number of GPUs per node to use for the task.')
    parser.add_argument('--gpu-cpus', type=int, default=6, help='Number of CPUs per GPU (vCPUs will be 2x this value).')
    parser.add_argument('--cpu-ram-gb', type=int, default=20, help='RAM per node in GB.')
    parser.add_argument('--copies', type=int, default=1, help='Number of job copies to submit.')
    parser.add_argument('--profile', default="stem", help='AWS profile to use. If not specified, uses the default profile.')
    parser.add_argument('--job-queue', default="metta-batch-jq-test", help='AWS Batch job queue to use.')
    args, task_args = parser.parse_known_args()

    args.num_nodes = max(1, args.gpus // args.node_gpus)

    # Set default commit values if not specified
    if args.git_branch is None and args.git_commit is None:
        args.git_commit = get_current_commit()
        print(f"Using current metta commit: {args.git_commit}")

    if args.mettagrid_branch is None and args.mettagrid_commit is None:
        mettagrid_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "deps", "mettagrid")
        args.mettagrid_commit = get_current_commit(mettagrid_path) if os.path.exists(mettagrid_path) else "main"
        print(f"Using current mettagrid commit: {args.mettagrid_commit}")


    for _ in range(args.copies):
        submit_batch_job(args, task_args)
