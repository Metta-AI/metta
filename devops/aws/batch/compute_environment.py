#!/usr/bin/env python3
"""
AWS Batch Compute Environment Utilities

This module provides functions for interacting with AWS Batch compute environments.
"""

import boto3
from botocore.config import Config
from tabulate import tabulate

def get_boto3_client(service_name="batch"):
    """Get a boto3 client with standard configuration."""
    config = Config(retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=50)
    return boto3.client(service_name, config=config)


def list_compute_environments():
    """List all available AWS Batch compute environments."""
    batch = get_boto3_client()

    try:
        response = batch.describe_compute_environments()
        compute_envs = response["computeEnvironments"]

        # Format the output
        table_data = []
        for ce in compute_envs:
            name = ce["computeEnvironmentName"]

            # Get compute resources if available
            compute_resources = ce.get("computeResources", {})
            instance_types = compute_resources.get("instanceTypes", [])
            instance_type_str = ", ".join(instance_types) if instance_types else "N/A"

            # Get ECS cluster and count instances
            num_instances = 0
            ecs_cluster_arn = ce.get("ecsClusterArn")
            if ecs_cluster_arn:
                ec2_instances = get_ec2_instances_for_cluster(ecs_cluster_arn)
                num_instances = len(ec2_instances)

            table_data.append([name, instance_type_str, num_instances])

        # Print the table
        headers = ["Name", "Instance Types", "Num Instances"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        return [ce["computeEnvironmentName"] for ce in compute_envs]
    except Exception as e:
        print(f"Error retrieving compute environments: {str(e)}")
        return []


def get_ec2_instances_for_cluster(cluster_arn):
    """Get EC2 instances for an ECS cluster."""
    ecs = get_boto3_client("ecs")
    ec2 = get_boto3_client("ec2")

    try:
        # Get container instances for the cluster
        container_instances = []
        paginator = ecs.get_paginator("list_container_instances")
        for page in paginator.paginate(cluster=cluster_arn):
            if page.get("containerInstanceArns"):
                container_instances.extend(page["containerInstanceArns"])

        if not container_instances:
            return []

        # Get container instance details
        container_instance_details = []
        # Process in batches of 100 (AWS API limit)
        for i in range(0, len(container_instances), 100):
            batch = container_instances[i : i + 100]
            response = ecs.describe_container_instances(cluster=cluster_arn, containerInstances=batch)
            container_instance_details.extend(response["containerInstances"])

        # Extract EC2 instance IDs
        ec2_instance_ids = [ci["ec2InstanceId"] for ci in container_instance_details]

        if not ec2_instance_ids:
            return []

        # Get EC2 instance details
        ec2_instances = []
        # Process in batches of 100 (AWS API limit)
        for i in range(0, len(ec2_instance_ids), 100):
            batch = ec2_instance_ids[i : i + 100]
            response = ec2.describe_instances(InstanceIds=batch)
            for reservation in response["Reservations"]:
                ec2_instances.extend(reservation["Instances"])

        return ec2_instances
    except Exception as e:
        print(f"Error retrieving EC2 instances for cluster: {str(e)}")
        return []


def get_gpu_count(instance_type):
    """Get the number of GPUs for an instance type."""
    # Extract instance family and size
    parts = instance_type.split(".")
    if len(parts) < 2:
        return 0

    family = parts[0]
    size = parts[1]

    # Known GPU instance types and their GPU counts
    gpu_instances = {
        # P2 instances
        "p2.xlarge": 1,
        "p2.8xlarge": 8,
        "p2.16xlarge": 16,
        # P3 instances
        "p3.2xlarge": 1,
        "p3.8xlarge": 4,
        "p3.16xlarge": 8,
        "p3dn.24xlarge": 8,
        # P4 instances
        "p4d.24xlarge": 8,
        "p4de.24xlarge": 8,
        # G3 instances
        "g3s.xlarge": 1,
        "g3.4xlarge": 1,
        "g3.8xlarge": 2,
        "g3.16xlarge": 4,
        # G4 instances
        "g4dn.xlarge": 1,
        "g4dn.2xlarge": 1,
        "g4dn.4xlarge": 1,
        "g4dn.8xlarge": 1,
        "g4dn.16xlarge": 1,
        "g4dn.12xlarge": 4,
        "g4dn.metal": 8,
        # G5 instances
        "g5.xlarge": 1,
        "g5.2xlarge": 1,
        "g5.4xlarge": 1,
        "g5.8xlarge": 1,
        "g5.16xlarge": 1,
        "g5.12xlarge": 4,
        "g5.24xlarge": 4,
        "g5.48xlarge": 8,
        # G6 instances
        "g6.xlarge": 1,
        "g6.2xlarge": 1,
        "g6.4xlarge": 1,
        "g6.8xlarge": 1,
        "g6.12xlarge": 4,
        "g6.16xlarge": 1,
        "g6.24xlarge": 4,
        "g6.48xlarge": 8,
    }

    # Check if the instance type is in our known list
    if instance_type in gpu_instances:
        return gpu_instances[instance_type]

    # For unknown instance types, try to infer based on family
    if family == "p2":
        if size == "xlarge":
            return 1
        elif size == "8xlarge":
            return 8
        elif size == "16xlarge":
            return 16
    elif family == "p3":
        if size == "2xlarge":
            return 1
        elif size == "8xlarge":
            return 4
        elif size == "16xlarge":
            return 8
        elif size == "24xlarge" or "dn.24xlarge" in instance_type:
            return 8
    elif family == "p4":
        if "24xlarge" in size:
            return 8
    elif family == "g3":
        if "s.xlarge" in instance_type:
            return 1
        elif size == "4xlarge":
            return 1
        elif size == "8xlarge":
            return 2
        elif size == "16xlarge":
            return 4
    elif family == "g4":
        if "dn.12xlarge" in instance_type:
            return 4
        elif "dn.metal" in instance_type:
            return 8
        elif "dn" in instance_type:
            return 1
    elif family == "g5":
        if size == "12xlarge":
            return 4
        elif size == "24xlarge":
            return 4
        elif size == "48xlarge":
            return 8
        else:
            return 1
    elif family == "g6":
        if size == "12xlarge":
            return 4
        elif size == "24xlarge":
            return 4
        elif size == "48xlarge":
            return 8
        else:
            return 1

    # Default to 0 if we can't determine
    return 0


def get_instance_specs(instance_type):
    """Get the vCPU and memory specifications for an instance type."""
    # Extract instance family and size
    parts = instance_type.split(".")
    if len(parts) < 2:
        return (0, 0)

    family = parts[0]
    size = parts[1]

    # Common instance types and their specs (vCPUs, Memory in GiB)
    instance_specs = {
        # General Purpose
        "t2.micro": (1, 1),
        "t2.small": (1, 2),
        "t2.medium": (2, 4),
        "t2.large": (2, 8),
        "t2.xlarge": (4, 16),
        "t2.2xlarge": (8, 32),
        "t3.micro": (2, 1),
        "t3.small": (2, 2),
        "t3.medium": (2, 4),
        "t3.large": (2, 8),
        "t3.xlarge": (4, 16),
        "t3.2xlarge": (8, 32),
        "t4g.micro": (2, 1),
        "t4g.small": (2, 2),
        "t4g.medium": (2, 4),
        "t4g.large": (2, 8),
        "t4g.xlarge": (4, 16),
        "t4g.2xlarge": (8, 32),
        "m4.large": (2, 8),
        "m4.xlarge": (4, 16),
        "m4.2xlarge": (8, 32),
        "m4.4xlarge": (16, 64),
        "m4.10xlarge": (40, 160),
        "m4.16xlarge": (64, 256),
        "m5.large": (2, 8),
        "m5.xlarge": (4, 16),
        "m5.2xlarge": (8, 32),
        "m5.4xlarge": (16, 64),
        "m5.8xlarge": (32, 128),
        "m5.12xlarge": (48, 192),
        "m5.16xlarge": (64, 256),
        "m5.24xlarge": (96, 384),
        "m6g.large": (2, 8),
        "m6g.xlarge": (4, 16),
        "m6g.2xlarge": (8, 32),
        "m6g.4xlarge": (16, 64),
        "m6g.8xlarge": (32, 128),
        "m6g.12xlarge": (48, 192),
        "m6g.16xlarge": (64, 256),
        # Compute Optimized
        "c4.large": (2, 3.75),
        "c4.xlarge": (4, 7.5),
        "c4.2xlarge": (8, 15),
        "c4.4xlarge": (16, 30),
        "c4.8xlarge": (36, 60),
        "c5.large": (2, 4),
        "c5.xlarge": (4, 8),
        "c5.2xlarge": (8, 16),
        "c5.4xlarge": (16, 32),
        "c5.9xlarge": (36, 72),
        "c5.18xlarge": (72, 144),
        "c6g.large": (2, 4),
        "c6g.xlarge": (4, 8),
        "c6g.2xlarge": (8, 16),
        "c6g.4xlarge": (16, 32),
        "c6g.8xlarge": (32, 64),
        "c6g.12xlarge": (48, 96),
        "c6g.16xlarge": (64, 128),
        # Memory Optimized
        "r4.large": (2, 15.25),
        "r4.xlarge": (4, 30.5),
        "r4.2xlarge": (8, 61),
        "r4.4xlarge": (16, 122),
        "r4.8xlarge": (32, 244),
        "r4.16xlarge": (64, 488),
        "r5.large": (2, 16),
        "r5.xlarge": (4, 32),
        "r5.2xlarge": (8, 64),
        "r5.4xlarge": (16, 128),
        "r5.8xlarge": (32, 256),
        "r5.12xlarge": (48, 384),
        "r5.16xlarge": (64, 512),
        "r5.24xlarge": (96, 768),
        "r6g.large": (2, 16),
        "r6g.xlarge": (4, 32),
        "r6g.2xlarge": (8, 64),
        "r6g.4xlarge": (16, 128),
        "r6g.8xlarge": (32, 256),
        "r6g.12xlarge": (48, 384),
        "r6g.16xlarge": (64, 512),
        # GPU Instances
        "p2.xlarge": (4, 61),
        "p2.8xlarge": (32, 488),
        "p2.16xlarge": (64, 732),
        "p3.2xlarge": (8, 61),
        "p3.8xlarge": (32, 244),
        "p3.16xlarge": (64, 488),
        "p3dn.24xlarge": (96, 768),
        "p4d.24xlarge": (96, 1152),
        "p4de.24xlarge": (96, 1152),
        "g3s.xlarge": (4, 30.5),
        "g3.4xlarge": (16, 122),
        "g3.8xlarge": (32, 244),
        "g3.16xlarge": (64, 488),
        "g4dn.xlarge": (4, 16),
        "g4dn.2xlarge": (8, 32),
        "g4dn.4xlarge": (16, 64),
        "g4dn.8xlarge": (32, 128),
        "g4dn.16xlarge": (64, 256),
        "g4dn.12xlarge": (48, 192),
        "g4dn.metal": (96, 384),
        "g5.xlarge": (4, 16),
        "g5.2xlarge": (8, 32),
        "g5.4xlarge": (16, 64),
        "g5.8xlarge": (32, 128),
        "g5.16xlarge": (64, 256),
        "g5.12xlarge": (48, 192),
        "g5.24xlarge": (96, 384),
        "g5.48xlarge": (192, 768),
        "g6.xlarge": (4, 16),
        "g6.2xlarge": (8, 32),
        "g6.4xlarge": (16, 64),
        "g6.8xlarge": (32, 128),
        "g6.12xlarge": (48, 192),
        "g6.16xlarge": (64, 256),
        "g6.24xlarge": (96, 384),
        "g6.48xlarge": (192, 768),
    }

    # Check if the instance type is in our known list
    if instance_type in instance_specs:
        return instance_specs[instance_type]

    # For unknown instance types, try to infer based on family and size
    vcpus = 0
    memory = 0

    # Estimate vCPUs based on instance size
    if "nano" in size:
        vcpus = 1
    elif "micro" in size:
        vcpus = 1
    elif "small" in size:
        vcpus = 1
    elif "medium" in size:
        vcpus = 2
    elif "large" in size and "xlarge" not in size:
        vcpus = 2
    elif "xlarge" in size:
        # Extract the multiplier if present (e.g., 2xlarge, 4xlarge)
        if size == "xlarge":
            vcpus = 4
        else:
            try:
                multiplier = int(size.split("xlarge")[0])
                vcpus = 4 * multiplier
            except ValueError:
                vcpus = 4  # Default if we can't parse

    # Estimate memory based on instance family and vCPUs
    if family in ["t2", "t3", "t4"]:
        memory = vcpus * 2  # Roughly 2 GiB per vCPU
    elif family in ["m4", "m5", "m6"]:
        memory = vcpus * 4  # Roughly 4 GiB per vCPU
    elif family in ["c4", "c5", "c6"]:
        memory = vcpus * 2  # Roughly 2 GiB per vCPU
    elif family in ["r4", "r5", "r6"]:
        memory = vcpus * 8  # Roughly 8 GiB per vCPU
    elif family in ["p2", "p3", "p4"]:
        memory = vcpus * 8  # Roughly 8 GiB per vCPU
    elif family in ["g3", "g4", "g5", "g6"]:
        memory = vcpus * 4  # Roughly 4 GiB per vCPU

    # If we couldn't estimate, try to get information from EC2
    if vcpus == 0 or memory == 0:
        try:
            ec2 = get_boto3_client("ec2")
            response = ec2.describe_instance_types(InstanceTypes=[instance_type])
            if response["InstanceTypes"]:
                instance_info = response["InstanceTypes"][0]
                vcpus = instance_info.get("VCpuInfo", {}).get("DefaultVCpus", 0)
                memory = instance_info.get("MemoryInfo", {}).get("SizeInMiB", 0) / 1024  # Convert MiB to GiB
        except Exception:
            # If API call fails, just use our estimates
            pass

    return (vcpus, memory)


def get_compute_environment_info(compute_env_name):
    """Get detailed information about a specific compute environment."""
    batch = get_boto3_client()

    try:
        response = batch.describe_compute_environments(computeEnvironments=[compute_env_name])

        if not response["computeEnvironments"]:
            print(f"Compute environment '{compute_env_name}' not found")
            return None

        ce = response["computeEnvironments"][0]

        # Print basic information
        print(f"\nCompute Environment: {ce['computeEnvironmentName']}")
        print(f"ARN: {ce['computeEnvironmentArn']}")
        print(f"State: {ce['state']}")
        print(f"Status: {ce['status']}")
        print(f"Status Reason: {ce.get('statusReason', 'N/A')}")
        print(f"Type: {ce['type']}")

        # Print compute resources if available
        if "computeResources" in ce:
            cr = ce["computeResources"]
            print("\nCompute Resources:")
            print(f"  Type: {cr.get('type', 'N/A')}")
            print(f"  Min vCPUs: {cr.get('minvCpus', 'N/A')}")
            print(f"  Max vCPUs: {cr.get('maxvCpus', 'N/A')}")
            print(f"  Desired vCPUs: {cr.get('desiredvCpus', 'N/A')}")

            # Print instance types
            instance_types = cr.get("instanceTypes", [])
            if instance_types:
                print("\n  Instance Types:")
                for it in instance_types:
                    vcpus, memory = get_instance_specs(it)
                    gpus = get_gpu_count(it)
                    print(f"    - {it} ({vcpus} vCPUs, {memory} GiB RAM, {gpus} GPUs)")

            # Print subnets
            subnets = cr.get("subnets", [])
            if subnets:
                print("\n  Subnets:")
                for subnet in subnets:
                    print(f"    - {subnet}")

            # Print security groups
            security_groups = cr.get("securityGroupIds", [])
            if security_groups:
                print("\n  Security Groups:")
                for sg in security_groups:
                    print(f"    - {sg}")

            # Print allocation strategy
            print(f"\n  Allocation Strategy: {cr.get('allocationStrategy', 'N/A')}")

            # Print EC2 key pair
            if "ec2KeyPair" in cr:
                print(f"  EC2 Key Pair: {cr['ec2KeyPair']}")

            # Print instance role
            if "instanceRole" in cr:
                print(f"  Instance Role: {cr['instanceRole']}")

            # Print tags
            if "tags" in cr:
                print("\n  Tags:")
                for key, value in cr["tags"].items():
                    print(f"    {key}: {value}")

        # Get ECS cluster
        ecs_cluster_arn = ce.get("ecsClusterArn")
        if ecs_cluster_arn:
            print(f"\nECS Cluster: {ecs_cluster_arn}")

            # Get EC2 instances for the cluster
            ec2_instances = get_ec2_instances_for_cluster(ecs_cluster_arn)

            if ec2_instances:
                print(f"\nInstances ({len(ec2_instances)}):")

                # Format the output
                table_data = []
                for instance in ec2_instances:
                    instance_id = instance["InstanceId"]
                    instance_type = instance["InstanceType"]
                    state = instance["State"]["Name"]

                    # Get instance specifications
                    vcpus, memory = get_instance_specs(instance_type)
                    gpus = get_gpu_count(instance_type)

                    # Get private and public IPs
                    private_ip = instance.get("PrivateIpAddress", "N/A")
                    public_ip = instance.get("PublicIpAddress", "N/A")

                    table_data.append(
                        [instance_id, instance_type, state, gpus, vcpus, f"{memory} GiB", private_ip, public_ip]
                    )

                # Print the table
                headers = ["Instance ID", "Type", "State", "GPUs", "vCPUs", "Memory", "Private IP", "Public IP"]
                print(tabulate(table_data, headers=headers, tablefmt="grid"))
            else:
                print("No instances found for this compute environment")

        # Get service role
        if "serviceRole" in ce:
            print(f"Service Role: {ce['serviceRole']}")

        return ce
    except Exception as e:
        print(f"Error retrieving compute environment information: {str(e)}")
        return None


def stop_compute_environment(compute_env_name):
    """Stop a compute environment by setting its state to DISABLED."""
    batch = get_boto3_client()

    try:
        # First, check if the compute environment exists
        response = batch.describe_compute_environments(computeEnvironments=[compute_env_name])

        if not response["computeEnvironments"]:
            print(f"Compute environment '{compute_env_name}' not found")
            return False

        # Update the compute environment state to DISABLED
        batch.update_compute_environment(computeEnvironment=compute_env_name, state="DISABLED")

        print(f"Compute environment '{compute_env_name}' has been disabled")
        return True
    except Exception as e:
        print(f"Error stopping compute environment: {str(e)}")
        return False
