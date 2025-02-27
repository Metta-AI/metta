import argparse
import netrc
import os
import random
import string

import boto3

machine_profiles = {
    "g6.8xlarge": {
        "vcpus": 32,
        "memory": 120,
        "gpus": 1,
    },
    "g6.12xlarge": {
        "vcpus": 48,
        "memory": 120,
        "gpus": 4,
    },
    "g6.48xlarge": {
        "vcpus": 192,
        "memory": 600,
        "gpus": 8,
    },
}

machine_for_gpu_count = {
    1: "g6.8xlarge",
    4: "g6.12xlarge",
    8: "g6.48xlarge",
}

def submit_batch_job(args, task_args):
    batch = boto3.client('batch')

    random_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
    job_name = args.run.replace('.', '_') + "_" + random_id
    job_queue = "metta-batch-jq-2"
    job_definition = "metta-batch-train-jd"

    response = batch.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides=container_config(args, task_args, job_name)
    )

    print(f"Submitted job {job_name} to queue {job_queue} with job ID {response['jobId']}")
    print(f"https://us-east-1.console.aws.amazon.com/batch/v2/home?region=us-east-1#/jobs/detail/{response['jobId']}")

def container_config(args, task_args, job_name):
    # Get the wandb key from the .netrc file
    netrc_info = netrc.netrc(os.path.expanduser('~/.netrc'))
    wandb_key = netrc_info.authenticators('api.wandb.ai')[2]
    if not wandb_key:
        raise ValueError('WANDB_API_KEY not found in .netrc file')

    machine_profile = machine_profiles[machine_for_gpu_count[args.gpus]]
    setup_cmds = [
        'git pull',
        'pip install -r requirements.txt',
        './devops/setup_build.sh',
        'ln -s /mnt/efs/train_dir train_dir',
    ]
    train_cmd = [
        f'NUM_GPUS={args.gpus} ',
        f'./devops/{args.cmd}.sh',
        f'run={args.run}',
        'hardware=aws',
        f'trainer.num_workers={min(6, machine_profile["vcpus"] // args.gpus // 2)}', # 2 vcpu per worker
        *task_args,
    ]
    if args.git_branch is not None:
        setup_cmds.append(f'git checkout {args.git_branch}')
        setup_cmds.append('pip uninstall termcolor')
        setup_cmds.append('pip install termcolor==2.4.0')

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
            }
        ],
        'vcpus': machine_profile['vcpus'],
        'memory': machine_profile['memory'],
        'resourceRequirements': [
            {
                'type': 'GPU',
                'value': str(machine_profile['gpus'])
            }
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch an AWS Batch task with a wandb key.')
    parser.add_argument('--cluster', default="metta", help='The name of the ECS cluster.')
    parser.add_argument('--run', required=True, help='The run id.')
    parser.add_argument('--cmd', required=True, choices=["train", "sweep", "evolve"], help='The command to run.')
    parser.add_argument('--git_branch', default=None, help='The git branch to use for the task.')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use for the task.')
    parser.add_argument('--copies', type=int, default=1, help='Number of job copies to submit.')
    args, task_args = parser.parse_known_args()

    for _ in range(args.copies):
        submit_batch_job(args, task_args)
