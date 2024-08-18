import boto3
import argparse
import netrc
import os
import logging

def submit_ecs_task(args, task_args):
    ecs = boto3.client('ecs')

    # Get the latest version of the task definition
    response = ecs.describe_task_definition(taskDefinition="metta-trainer")
    task_definition = response['taskDefinition']['taskDefinitionArn']

    print(f"launching task using {task_definition}")

    # Launch the task
    response = ecs.run_task(
        cluster=args.cluster,
        taskDefinition=task_definition,
        startedBy=args.experiment,
        launchType='EC2',
        overrides={
            'containerOverrides': [{
                "name": "metta",
                **container_config(args, task_args)
            }]
        }
    )

    if response['ResponseMetadata']['HTTPStatusCode'] == 200 and response['tasks']:
        task_id = response['tasks'][0]['taskArn']
        print(f'Task submitted: {task_id}')
        print(
            "https://us-east-1.console.aws.amazon.com/ecs/v2/clusters/" +
            args.cluster +
            "/tasks/" +
            task_id.split('/')[-1] +
            "/?selectedContainer=metta")
    else:
        logging.error('Failed to submit: %s', response)

def submit_batch_job(args, task_args):
    batch = boto3.client('batch')

    job_name = args.experiment.replace('.', '_')
    job_queue = "metta-batch-jq"
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
        'git submodule update',
        'pip install -e .',
        'ln -s /mnt/efs/train_dir train_dir',
    ]
    train_cmd = [
        './devops/train.sh',
        f'++experiment={args.experiment}',
        f'++framework={args.framework}',
        '++hardware=aws',
        *task_args,
    ]
    if args.git_branch is not None:
        setup_cmds.append(f'git checkout {args.git_branch}')
    if args.init_model is not None:
        setup_cmds.append(f'./devops/load_model.sh {args.init_model}',)
        train_cmd.append(f'+framework.init_checkpoint_path=train_dir/{args.init_model}/latest.pth')

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
        ]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch an ECS task with a wandb key.')
    parser.add_argument('--cluster', default="metta", help='The name of the ECS cluster.')
    parser.add_argument('--experiment', required=True, help='The experiment to run.')
    parser.add_argument('--framework', default="sample_factory", choices=["sample_factory", "pufferlib"], help='The framework to use.')
    parser.add_argument('--init_model', default=None, help='The experiment to run.')
    parser.add_argument('--git_branch', default=None, help='The git branch to use for the task.')
    parser.add_argument('--batch', default=True, help='Submit as a batch job.')
    args, task_args = parser.parse_known_args()

    if args.batch:
        submit_batch_job(args, task_args)
    else:
        submit_ecs_task(args, task_args)
