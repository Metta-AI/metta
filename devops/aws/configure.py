import argparse
import boto3
import logging

def register_task_definition(args):
    ecs = boto3.client('ecs')

    memory = int(args.memory * 1024)

    task_definition = {
        "family": "metta-trainer",
        "containerDefinitions": [
            {
                "name": "metta",
                "image": "daveey/metta:latest",
                "portMappings": [],
                "essential": True,
                "entryPoint": [
                    "bash",
                    "-c"
                ],
                "command": [
                    "/bin/sleep 10000"
                ],
                "linuxParameters": {
                    "sharedMemorySize": memory * 1024 * 1024
                },
                'mountPoints': [{
                    'sourceVolume': 'efs',
                    'containerPath': '/mnt/efs'
                }],
                "resourceRequirements": [
                    {
                        "type": "GPU",
                        "value": "1"
                    }
                ],
                "environment": [],
                "environmentFiles": [],
                "volumesFrom": [],
                "workingDirectory": "/workspace/metta",
                "ulimits": [
                    {
                        "name": "nproc",
                        "softLimit": 640000,
                        "hardLimit": 640000
                    },
                    {
                        "name": "nofile",
                        "softLimit": 64000,
                        "hardLimit": 64000
                    }
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-create-group": "true",
                        "awslogs-group": "/ecs/metta-trainer",
                        "awslogs-region": "us-east-1",
                        "awslogs-stream-prefix": "ecs"
                    },
                    "secretOptions": []
                },
                "systemControls": []
            }
        ],
        "executionRoleArn": "arn:aws:iam::767406518141:role/ecsTaskExecutionRole",
        'networkMode': 'host',
        "requiresCompatibilities": [
            "EC2"
        ],
        "memory": str(memory),
        "runtimePlatform": {
            "cpuArchitecture": "X86_64",
            "operatingSystemFamily": "LINUX"
        },
        'volumes': [{
            'name': 'efs',
            'host': {
                'sourcePath': '/mnt/efs',
            }
        }]
    }

    response = ecs.register_task_definition(**task_definition)

    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print('Task definition registered successfully.')
    else:
        logging.error('Failed to register task definition: %s', response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Register an ECS task definition.')
    parser.add_argument('--task-name', default="metta-train", help='The name of the task definition.')
    parser.add_argument('--memory', type=float, default=29.7, help='The amount of memory to use.')
    args = parser.parse_args()

    register_task_definition(args)
