#!/usr/bin/env python3
import argparse
import json
import os
from pprint import pprint

import boto3

def get_role_arn(role_name, profile=None):
    """Look up the ARN for a given IAM role name."""
    try:
        # Use the specified profile or default to stem-root
        session = boto3.Session(profile_name=profile or "stem-root", region_name="us-east-1")
        iam = session.client("iam")
        response = iam.get_role(RoleName=role_name)
        return response["Role"]["Arn"]
    except Exception as e:
        print(f"Error: Could not retrieve ARN for role {role_name}: {e}")
        # Return a hardcoded ARN as fallback
        return f"arn:aws:iam::767406518141:role/{role_name}"


def get_efs_id(name_tag=None):
    """Look up the EFS file system ID, optionally filtering by name tag."""
    try:
        # Explicitly use the default profile
        efs = boto3.client("efs", region_name="us-east-1")
        response = efs.describe_file_systems()

        # If no name tag is provided, return the first file system
        if not name_tag:
            if response["FileSystems"]:
                return response["FileSystems"][0]["FileSystemId"]
            raise ValueError("No EFS file systems found")

        # Filter by name tag if provided
        for fs in response["FileSystems"]:
            # Get tags for this file system
            tags_response = efs.describe_tags(FileSystemId=fs["FileSystemId"])
            for tag in tags_response["Tags"]:
                if tag["Key"] == "Name" and tag["Value"] == name_tag:
                    return fs["FileSystemId"]

        # If we get here, no matching file system was found
        if response["FileSystems"]:
            print(f"Warning: No EFS with name tag '{name_tag}' found. Using first available.")
            return response["FileSystems"][0]["FileSystemId"]
        raise ValueError("No EFS file systems found")
    except Exception as e:
        print(f"Error: Could not retrieve EFS file system ID: {e}")
        # Return a hardcoded EFS ID as fallback
        return "fs-084c321137786b15c"


def create_job_definition(args):
    """Create a job definition dictionary without registering it."""
    # Use the stem-root profile for AWS operations
    boto3.Session(profile_name="stem-root", region_name="us-east-1")

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
                        "command": ["/bin/bash", "-c"],
                        "jobRoleArn": job_role_arn,
                        "executionRoleArn": execution_role_arn,
                        "volumes": [
                            {"name": "efs", "efsVolumeConfiguration": {"fileSystemId": efs_id}},
                        ],
                        "environment": [
                            {"name": "NCCL_DEBUG", "value": "INFO"},
                            {"name": "NCCL_SOCKET_IFNAME", "value": "eth0"},
                        ],
                        "mountPoints": [{"containerPath": "/mnt/efs", "readOnly": False, "sourceVolume": "efs"}],
                        "ulimits": [
                            {"hardLimit": 640000, "name": "nproc", "softLimit": 640000},
                            {"hardLimit": 640000, "name": "nofile", "softLimit": 640000},
                        ],
                        "user": "root",
                        "resourceRequirements": [
                            {"value": str(args.vcpus), "type": "VCPU"},
                            {"value": str(args.memory), "type": "MEMORY"},
                            {"value": str(args.gpus), "type": "GPU"},
                        ],
                        "linuxParameters": {
                            "devices": [],
                            "sharedMemorySize": args.shared_memory,
                            "tmpfs": [],
                            "maxSwap": 0,
                            "swappiness": 0,
                        },
                        "logConfiguration": {
                            "logDriver": "awslogs",
                            # "options": {
                            #     "awslogs-group": f"/aws/batch/{args.job_definition_name}",
                            #     "awslogs-region": "us-east-1",
                            #     "awslogs-stream-prefix": "batch-job",
                            #     "awslogs-create-group": "true",
                            #     "mode": "non-blocking",
                            #     "max-buffer-size": "4m"
                            # }
                        },
                    },
                }
            ],
            "numNodes": args.num_nodes,
        },
        "parameters": {},
        "platformCapabilities": ["EC2"],
        "retryStrategy": {
            "attempts": 10,
            "evaluateOnExit": [
                {"onExitCode": "1", "action": "retry"},
                {"onExitCode": "137", "action": "retry"},
                {"onExitCode": "139", "action": "retry"},
                {"onExitCode": "127", "action": "exit"},
            ],
        },
        "tags": {"Purpose": "DistributedTraining", "Framework": "PyTorch"},
        "propagateTags": True,
    }

    return job_def


def register_job_definition(args):
    """Register a new job definition or a new version of an existing one."""
    # Use the stem-root profile for AWS operations
    session = boto3.Session(profile_name="stem-root", region_name="us-east-1")
    batch = session.client("batch")

    # Check if job definition already exists
    try:
        response = batch.describe_job_definitions(jobDefinitionName=args.job_definition_name, status="ACTIVE")
        job_definitions = response.get("jobDefinitions", [])
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
        print("Successfully registered job definition:")
        pprint(response)
        print(f"\nARN: {response.get('jobDefinitionArn')}")
        print(f"Revision: {response.get('revision')}")
    except Exception as e:
        print(f"Error registering job definition: {e}")
        print("Job definition that failed validation:")
        pprint(job_def)
        raise


def main():
    # Use stem-root profile instead of unsetting AWS_PROFILE
    if "AWS_PROFILE" in os.environ and os.environ["AWS_PROFILE"] != "stem-root":
        print(f"Setting AWS_PROFILE environment variable to stem-root (was: {os.environ['AWS_PROFILE']})")
        os.environ["AWS_PROFILE"] = "stem-root"
    elif "AWS_PROFILE" not in os.environ:
        print("Setting AWS_PROFILE environment variable to stem-root")
        os.environ["AWS_PROFILE"] = "stem-root"

    parser = argparse.ArgumentParser(description="Register a multi-node AWS Batch job definition")
    parser.add_argument("--job-definition-name", default="metta-batch-dist-train", help="Name of the job definition")
    parser.add_argument(
        "--image", default="767406518141.dkr.ecr.us-east-1.amazonaws.com/metta:latest", help="Docker image to use"
    )

    # Role ARN options
    role_group = parser.add_argument_group("IAM Role Options")
    role_group.add_argument("--job-role-arn", default=None, help="Job role ARN (if not provided, will look up by name)")
    role_group.add_argument("--job-role-name", default="ecsTaskExecutionRole", help="Job role name to look up ARN")
    role_group.add_argument(
        "--execution-role-arn", default=None, help="Execution role ARN (if not provided, will look up by name)"
    )
    role_group.add_argument(
        "--execution-role-name", default="ecsTaskExecutionRole", help="Execution role name to look up ARN"
    )

    # EFS options
    efs_group = parser.add_argument_group("EFS Options")
    efs_group.add_argument("--efs-id", default=None, help="EFS file system ID (if not provided, will look up)")
    efs_group.add_argument("--efs-name", default=None, help="EFS name tag to look up file system ID")

    parser.add_argument("--output-json", action="store_true", help="Output the job definition as JSON")

    args = parser.parse_args()

    # Hardcoded minimal resource values
    args.num_nodes = 1
    args.vcpus = 1
    args.memory = 1024  # 1GB
    args.gpus = 1
    args.shared_memory = 230000

    # Print resource configuration
    print("Using resource configuration:")
    print(f"  Job Definition: {args.job_definition_name}")
    print(f"  Nodes: {args.num_nodes}")
    print(f"  vCPUs: {args.vcpus}")
    print(f"  Memory: {args.memory}MB")
    print(f"  GPUs: {args.gpus}")
    print(f"  Shared Memory: {args.shared_memory}MB")

    if args.output_json:
        # Create job definition but don't register it
        job_def = create_job_definition(args)
        print(json.dumps(job_def, indent=2))
    else:
        register_job_definition(args)


if __name__ == "__main__":
    main()
