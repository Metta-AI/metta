#!/usr/bin/env python3
import argparse
import json
import boto3
import os
from pprint import pprint

def get_role_arn(role_name):
    """Look up the ARN for a given IAM role name."""
    try:
        # Explicitly use the default profile
        iam = boto3.client('iam', region_name='us-east-1')
        response = iam.get_role(RoleName=role_name)
        return response['Role']['Arn']
    except Exception as e:
        print(f"Error: Could not retrieve ARN for role {role_name}: {e}")
        # Return a hardcoded ARN as fallback
        return f"arn:aws:iam::767406518141:role/{role_name}"

def get_efs_id(name_tag=None):
    """Look up the EFS file system ID, optionally filtering by name tag."""
    try:
        # Explicitly use the default profile
        efs = boto3.client('efs', region_name='us-east-1')
        response = efs.describe_file_systems()

        # If no name tag is provided, return the first file system
        if not name_tag:
            if response['FileSystems']:
                return response['FileSystems'][0]['FileSystemId']
            raise ValueError("No EFS file systems found")

        # Filter by name tag if provided
        for fs in response['FileSystems']:
            # Get tags for this file system
            tags_response = efs.describe_tags(FileSystemId=fs['FileSystemId'])
            for tag in tags_response['Tags']:
                if tag['Key'] == 'Name' and tag['Value'] == name_tag:
                    return fs['FileSystemId']

        # If we get here, no matching file system was found
        if response['FileSystems']:
            print(f"Warning: No EFS with name tag '{name_tag}' found. Using first available.")
            return response['FileSystems'][0]['FileSystemId']
        raise ValueError("No EFS file systems found")
    except Exception as e:
        print(f"Error: Could not retrieve EFS file system ID: {e}")
        # Return a hardcoded EFS ID as fallback
        return "fs-084c321137786b15c"

def create_job_definition(args):
    """Create a job definition dictionary without registering it."""
    # Look up role ARNs if not provided
    job_role_arn = args.job_role_arn
    if not job_role_arn:
        job_role_arn = get_role_arn(args.job_role_name)

    execution_role_arn = args.execution_role_arn
    if not execution_role_arn:
        execution_role_arn = get_role_arn(args.execution_role_name)

    # Look up EFS ID if not provided
    efs_id = args.efs_id
    if not efs_id:
        efs_id = get_efs_id(args.efs_name)

    # Prepare job definition parameters
    job_def = {
        "jobDefinitionName": args.job_definition_name,
        "type": "multinode",
        "nodeProperties": {
            "mainNode": 0,
            "nodeRangeProperties": [
                {
                    "targetNodes": "0:",
                    "container": {
                        "image": args.image,
                        "command": [
                            "/bin/bash",
                            "-c",
                            "/workspace/entrypoint.sh"
                        ],
                        "jobRoleArn": job_role_arn,
                        "executionRoleArn": execution_role_arn,
                        "volumes": [
                            {
                                "name": "efs",
                                "efsVolumeConfiguration": {
                                    "fileSystemId": efs_id
                                }
                            },
                            {
                                "name": "tmp",
                                "host": {
                                    "sourcePath": "/tmp"
                                }
                            }
                        ],
                        "environment": [
                            {
                                "name": "NCCL_DEBUG",
                                "value": "INFO"
                            },
                            {
                                "name": "NCCL_SOCKET_IFNAME",
                                "value": "eth0"
                            }
                        ],
                        "mountPoints": [
                            {
                                "containerPath": "/mnt/efs",
                                "readOnly": False,
                                "sourceVolume": "efs"
                            },
                            {
                                "containerPath": "/tmp",
                                "readOnly": False,
                                "sourceVolume": "tmp"
                            }
                        ],
                        "privileged": True,
                        "ulimits": [
                            {
                                "hardLimit": 640000,
                                "name": "nproc",
                                "softLimit": 640000
                            },
                            {
                                "hardLimit": 640000,
                                "name": "nofile",
                                "softLimit": 640000
                            }
                        ],
                        "user": "root",
                        "resourceRequirements": [
                            {
                                "value": str(args.vcpus),
                                "type": "VCPU"
                            },
                            {
                                "value": str(args.memory),
                                "type": "MEMORY"
                            },
                            {
                                "value": str(args.gpus),
                                "type": "GPU"
                            }
                        ],
                        "linuxParameters": {
                            "devices": [],
                            "sharedMemorySize": 230000,
                            "tmpfs": [],
                            "maxSwap": 0,
                            "swappiness": 0
                        },
                        "logConfiguration": {
                            "logDriver": "awslogs",
                            "options": {},
                            "secretOptions": []
                        }
                    }
                }
            ],
            "numNodes": args.num_nodes
        },
        "parameters": {},
        "platformCapabilities": [
            "EC2"
        ],
        "retryStrategy": {
            "attempts": 10,
            "evaluateOnExit": [
                {
                    "onExitCode": "1",
                    "action": "retry"
                },
                {
                    "onExitCode": "137",
                    "action": "retry"
                },
                {
                    "onExitCode": "139",
                    "action": "retry"
                }
            ]
        },
        "tags": {
            "Purpose": "DistributedTraining",
            "Framework": "PyTorch"
        },
        "propagateTags": True
    }

    return job_def

def register_job_definition(args):
    """Register a new job definition or a new version of an existing one."""
    # Explicitly use the default profile
    batch = boto3.client('batch', region_name='us-east-1')

    # Check if job definition already exists
    try:
        response = batch.describe_job_definitions(
            jobDefinitionName=args.job_definition_name,
            status='ACTIVE'
        )
        job_definitions = response.get('jobDefinitions', [])
        if job_definitions:
            print(f"Job definition '{args.job_definition_name}' already exists. Registering a new version.")
    except Exception as e:
        print(f"Error checking existing job definition: {e}")
        print("Proceeding with registration...")

    # Create the job definition
    job_def = create_job_definition(args)

    # Register the job definition
    try:
        response = batch.register_job_definition(**job_def)
        print(f"Successfully registered job definition:")
        pprint(response)
        print(f"\nARN: {response.get('jobDefinitionArn')}")
        print(f"Revision: {response.get('revision')}")
    except Exception as e:
        print(f"Error registering job definition: {e}")
        print("Job definition that failed validation:")
        pprint(job_def)
        raise

def main():
    # Unset AWS_PROFILE if it's set to avoid profile not found errors
    if 'AWS_PROFILE' in os.environ:
        print(f"Unsetting AWS_PROFILE environment variable (was set to: {os.environ['AWS_PROFILE']})")
        del os.environ['AWS_PROFILE']

    parser = argparse.ArgumentParser(description='Register a multi-node AWS Batch job definition')
    parser.add_argument('--job-definition-name', default='metta-batch-dist-train',
                        help='Name of the job definition')
    parser.add_argument('--image', default='mettaai/metta:latest',
                        help='Docker image to use')

    # Role ARN options
    role_group = parser.add_argument_group('IAM Role Options')
    role_group.add_argument('--job-role-arn', default=None,
                        help='Job role ARN (if not provided, will look up by name)')
    role_group.add_argument('--job-role-name', default='ecsTaskExecutionRole',
                        help='Job role name to look up ARN')
    role_group.add_argument('--execution-role-arn', default=None,
                        help='Execution role ARN (if not provided, will look up by name)')
    role_group.add_argument('--execution-role-name', default='ecsTaskExecutionRole',
                        help='Execution role name to look up ARN')

    # EFS options
    efs_group = parser.add_argument_group('EFS Options')
    efs_group.add_argument('--efs-id', default=None,
                        help='EFS file system ID (if not provided, will look up)')
    efs_group.add_argument('--efs-name', default=None,
                        help='EFS name tag to look up file system ID')

    # Resource options
    parser.add_argument('--num-nodes', type=int, default=2,
                        help='Number of nodes for distributed training')
    parser.add_argument('--vcpus', type=int, default=32,
                        help='Number of vCPUs per node')
    parser.add_argument('--memory', type=int, default=128000,
                        help='Memory in MB per node')
    parser.add_argument('--gpus', type=int, default=4,
                        help='Number of GPUs per node')
    parser.add_argument('--output-json', action='store_true',
                        help='Output the job definition as JSON')

    args = parser.parse_args()

    if args.output_json:
        # Create job definition but don't register it
        job_def = create_job_definition(args)
        print(json.dumps(job_def, indent=2))
    else:
        register_job_definition(args)

if __name__ == "__main__":
    main()
