import boto3
import argparse
import netrc
import os
import logging

machine_profiles = {
    "g5.2xlarge": {
        "vcpus": 8,
        "memory": 28,
    },
    "g5.4xlarge": {
        "vcpus": 16,
        "memory": 60,
    },
    "g5.8xlarge": {
        "vcpus": 32,
        "memory": 120,
    },
    "g5.16xlarge": {
        "vcpus": 64,
        "memory": 250,
    },
    "g6.8xlarge": {
        "vcpus": 32,
        "memory": 120,
    },}

def submit_batch_job(args, task_args):
    batch = boto3.client('batch')

    job_name = args.run.replace('.', '_')
    job_queue = "metta-batch-jq-" + args.instance_type.replace('.', '-')
    job_definition = "metta-batch-train-jd"

    response = batch.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides=container_config(args, task_args)
    )

    print(f"Submitted job {job_name} to queue {job_queue} with job ID {response['jobId']}")
    print(f"https://us-east-1.console.aws.amazon.com/batch/v2/home?region=us-east-1#/jobs/detail/{response['jobId']}")

def container_config(args, task_args):
    # Get the wandb key from the .netrc file
    netrc_info = netrc.netrc(os.path.expanduser('~/.netrc'))
    wandb_key = netrc_info.authenticators('api.wandb.ai')[2]
    if not wandb_key:
        raise ValueError('WANDB_API_KEY not found in .netrc file')

    # Get the hugging face key from the cache file
    hugging_face_key_file = os.path.expanduser("~/.cache/huggingface/token")
    with open(hugging_face_key_file, 'r') as file:
        hugging_face_key = file.read().strip()

    setup_cmds = [
        'git pull',
        'pip install -r requirements.txt',
        './devops/setup_build.sh',
        'ln -s /mnt/efs/train_dir train_dir',
    ]
    train_cmd = [
        f'./devops/{args.cmd}.sh',
        f'run={args.run}',
        'hardware=aws.' + args.instance_type,
        *task_args,
    ]
    if args.git_branch is not None:
        setup_cmds.append(f'git checkout {args.git_branch}')
        setup_cmds.append(f'pip uninstall termcolor')
        setup_cmds.append(f'pip install termcolor==2.4.0')

    print("\n".join([
            "Setup:",
            "-"*10,
            "\n".join(setup_cmds),
            "-"*10,
            "Command:",
            "-"*10,
            " ".join(train_cmd),
        ]))

    return {
        'command': ["bash", "-c", "; ".join([
            *setup_cmds,
            " ".join(train_cmd),
        ])],
        'environment': [
            {
                'name': 'HYDRA_FULL_ERROR',
                'value': '1'
            },
            {
                'name': 'WANDB_API_KEY',
                'value': wandb_key
            },
            {
                'name': 'TRANSFORMERS_TOKEN',
                'value': hugging_face_key
            },
            {
                'name': 'COLOR_LOGGING',
                'value': 'false'
            },
        ],
        'vcpus': machine_profiles[args.instance_type]['vcpus'],
        'memory': machine_profiles[args.instance_type]['memory'],
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch an AWS Batch task with a wandb key.')
    parser.add_argument('--cluster', default="metta", help='The name of the ECS cluster.')
    parser.add_argument('--run', required=True, help='The run id.')
    parser.add_argument('--cmd', required=True, choices=["train", "sweep", "evolve"], help='The command to run.')
    parser.add_argument('--git_branch', default=None, help='The git branch to use for the task.')
    parser.add_argument('--instance_type', default="g5.8xlarge", help='The instance type to use for the task.')
    args, task_args = parser.parse_known_args()

    submit_batch_job(args, task_args)
