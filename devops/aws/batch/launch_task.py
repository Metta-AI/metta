import argparse
import netrc
import os
import random
import string
import sys
import subprocess
import json
from colorama import Fore, Style, init

import boto3

# Initialize colorama
init(autoreset=True)

specs = {
    1: {
        "node_gpus": 1,
        "node_ram_gb": 60,
        "gpu_cpus": 10,
    },
    4: {
        "node_gpus": 4,
        "node_ram_gb": 150,
        "gpu_cpus": 12,
    },
    8: {
        "node_gpus": 8,
        "node_ram_gb": 300,
        "gpu_cpus": 24,
    },
}

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

def is_commit_pushed(commit_hash, repo_path=None):
    """Check if a commit has been pushed to the remote repository."""
    try:
        cmd = ["git", "branch", "-r", "--contains", commit_hash]
        if repo_path:
            cmd = ["git", "-C", repo_path, "branch", "-r", "--contains", commit_hash]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # If there are any remote branches containing this commit, it has been pushed
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        # If the command fails, assume the commit hasn't been pushed
        return False

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

    request = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': 'metta-batch-train-jd',
        'containerOverrides': container_config(args, task_args, job_name)
    }

    # Choose job definition based on number of nodes
    if args.num_nodes > 1:
        request["jobDefinition"] = "metta-batch-dist-train"
        print(f"Using multi-node job definition: {request['jobDefinition']} for {args.num_nodes} nodes")
        request["nodeOverrides"] = {
            'nodePropertyOverrides': [
                {
                    'targetNodes': '0:',
                    'containerOverrides': container_config(args, task_args, job_name)
                }
            ],
            'numNodes': args.num_nodes
        }
        del request["containerOverrides"]

    # Check if no_color attribute exists and is True
    no_color = getattr(args, 'no_color', False)

    # Check if dry_run attribute exists and is True
    dry_run = getattr(args, 'dry_run', False)

    if dry_run:
        print(f"\n{'=' * 40}")
        print(f"DRY RUN - Job would be submitted with the following details:")
        print(f"{'=' * 40}")
        print(f"Job Name: {job_name}")
        print(f"Job Queue: {job_queue}")
        print(f"Job Definition: {request['jobDefinition']}")
        if args.num_nodes > 1:
            print(f"Number of Nodes: {args.num_nodes}")
        print(f"Number of GPUs per Node: {args.node_gpus}")
        print(f"Total GPUs: {args.gpus}")
        print(f"vCPUs per GPU: {args.gpu_cpus}")
        print(f"RAM per Node: {args.node_ram_gb} GB")
        print(f"Git Reference: {args.git_branch or args.git_commit}")
        print(f"Mettagrid Reference: {args.mettagrid_branch or args.mettagrid_commit}")
        print(f"{'-' * 40}")
        print(f"Command: {args.cmd}")
        if task_args:
            if no_color:
                print(f"\nTask Arguments:")
                for i, arg in enumerate(task_args):
                    print(f"  {i+1}. {arg}")
            else:
                print(f"\n{Fore.YELLOW}Task Arguments:{Style.RESET_ALL}")
                for i, arg in enumerate(task_args):
                    print(f"  {i+1}. {Fore.CYAN}{arg}{Style.RESET_ALL}")
        print(f"\n{'=' * 40}")
        print("DRY RUN - No job was actually submitted")
        print(f"{'=' * 40}")
        return

    response = batch.submit_job(**request)

    # Print job information with colors if not disabled
    job_id = response['jobId']
    job_url = f"https://us-east-1.console.aws.amazon.com/batch/v2/home?region=us-east-1#/jobs/detail/{job_id}"

    if no_color:
        print(f"Submitted job {job_name} to queue {job_queue} with job ID {job_id}")
        print(f"{job_url}")
    else:
        print(f"Submitted job {job_name} to queue {job_queue} with job ID {Fore.GREEN}{Style.BRIGHT}{job_id}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}{Style.BRIGHT}{job_url}{Style.RESET_ALL}")

    # Pretty print task arguments
    if task_args:
        if no_color:
            print(f"\nTask Arguments:")
            for i, arg in enumerate(task_args):
                print(f"  {i+1}. {arg}")
        else:
            print(f"\n{Fore.YELLOW}Task Arguments:{Style.RESET_ALL}")
            for i, arg in enumerate(task_args):
                print(f"  {i+1}. {Fore.CYAN}{arg}{Style.RESET_ALL}")

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
    vcpus_per_gpu = args.gpu_cpus
    total_vcpus = vcpus_per_gpu * args.node_gpus

    # Memory in GB, convert to MB for AWS Batch API
    memory_gb = int(args.node_ram_gb)
    memory_mb = memory_gb * 1024

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
            'value': 'ERROR'
        },
        {
            'name': 'NCCL_IGNORE_DISABLED_P2P',
            'value': '1'
        },
        {
            'name': 'NCCL_IB_DISABLE',
            'value': '1'
        },
        {
            'name': 'NCCL_P2P_DISABLE',
            'value': '1'
        },
    ]

    # Add required environment variables for the entrypoint script
    env_vars.extend([
        {
            'name': 'RUN_ID',
            'value': args.run
            },
        {
            'name': 'HARDWARE',
            'value': 'aws'
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
            'value': str(vcpus_per_gpu)
        },
        {
            'name': 'GIT_REF',
            'value': args.git_branch or args.git_commit
        },
        {
            'name': 'METTAGRID_REF',
            'value': args.mettagrid_branch or args.mettagrid_commit
        },
        {
            'name': 'TASK_ARGS',
            'value': ' '.join(task_args)
        }
    ])

    # Build the command to run the entrypoint script
    entrypoint_cmd = [
        'git fetch',
        f'git checkout {args.git_branch or args.git_commit}',
        './devops/aws/batch/entrypoint.sh'
    ]

    print("\n".join([
            f"Resources: {args.num_nodes} nodes, {args.node_gpus} GPUs, {total_vcpus} vCPUs ({vcpus_per_gpu} per GPU), {memory_gb}GB RAM"
        ]))

    # Create resource requirements
    resource_requirements = [
        {
            'type': 'GPU',
            'value': str(args.node_gpus)
        }, {
            'type': 'VCPU',
            'value': str(total_vcpus)
        }, {
            'type': 'MEMORY',
            'value': str(memory_mb)  # AWS Batch API expects MB
        }
    ]

    return {
        'command': ["; ".join(entrypoint_cmd)],
        'environment': env_vars,
        'resourceRequirements': resource_requirements
    }

def main():
    """Main entry point for the script."""
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
    parser.add_argument('--gpu-cpus', type=int, help='Number of CPUs per GPU (vCPUs will be 2x this value).')
    parser.add_argument('--node-ram-gb', type=int, help='RAM per node in GB.')
    parser.add_argument('--copies', type=int, default=1, help='Number of job copies to submit.')
    parser.add_argument('--profile', default="stem", help='AWS profile to use. If not specified, uses the default profile.')
    parser.add_argument('--job-queue', default="metta-jq", help='AWS Batch job queue to use.')
    parser.add_argument('--skip-push-check', action='store_true', help='Skip checking if commits have been pushed.')
    parser.add_argument('--no-color', action='store_true', help='Disable colored output.')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode, prints job details without submitting.')
    args, task_args = parser.parse_known_args()

    args.num_nodes = max(1, args.gpus // args.node_gpus)
    if args.node_ram_gb is None:
        args.node_ram_gb = specs[args.node_gpus]["node_ram_gb"]
    if args.gpu_cpus is None:
        args.gpu_cpus = specs[args.node_gpus]["gpu_cpus"]

    # Set default commit values if not specified
    if args.git_branch is None and args.git_commit is None:
        args.git_commit = get_current_commit()

    if args.mettagrid_branch is None and args.mettagrid_commit is None:
        args.mettagrid_commit = get_current_commit("deps/mettagrid")

    # Check if commits have been pushed
    if not args.skip_push_check:
        # Check if git commit has been pushed
        if args.git_commit and not is_commit_pushed(args.git_commit):
            print(f"Error: Git commit {args.git_commit} has not been pushed to the remote repository.")
            print("Please push your changes or use --skip-push-check to bypass this check.")
            sys.exit(1)

        # Check if mettagrid commit has been pushed
        if args.mettagrid_commit and not is_commit_pushed(args.mettagrid_commit, "deps/mettagrid"):
            print(f"Error: Mettagrid commit {args.mettagrid_commit} has not been pushed to the remote repository.")
            print("Please push your changes or use --skip-push-check to bypass this check.")
            sys.exit(1)

    # Submit the job
    for i in range(args.copies):
        submit_batch_job(args, task_args)

if __name__ == '__main__':
    main()
