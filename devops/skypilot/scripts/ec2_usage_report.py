#!/usr/bin/env python3
"""Generate a comprehensive EC2 usage report for the Softmax AWS account.

This script reports on:
- Instance counts by type and state
- vCPU usage by instance family (Standard vs GPU-accelerated)
- Service quota limits and available capacity
- Cost estimates for running instances

Usage:
    ./devops/skypilot/scripts/ec2_usage_report.py [--profile PROFILE] [--region REGION]
"""

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from typing import Any

# Instance type to vCPU mapping
VCPU_MAP = {
    # C family - Compute Optimized (Standard)
    "c6a.xlarge": 4,
    "c6a.2xlarge": 8,
    "c6a.4xlarge": 16,
    "c6a.8xlarge": 32,
    "c6a.12xlarge": 48,
    "c6a.16xlarge": 64,
    "c6a.24xlarge": 96,
    "c6i.large": 2,
    "c6i.xlarge": 4,
    "c6i.2xlarge": 8,
    "c6i.4xlarge": 16,
    "c6i.8xlarge": 32,
    "c6i.12xlarge": 48,
    "c6i.16xlarge": 64,
    "c6i.24xlarge": 96,
    # M family - General Purpose (Standard)
    "m5.xlarge": 4,
    "m5.2xlarge": 8,
    "m5.4xlarge": 16,
    "m6i.large": 2,
    "m6i.xlarge": 4,
    "m6i.2xlarge": 8,
    "m6i.4xlarge": 16,
    "m6i.8xlarge": 32,
    "m6i.12xlarge": 48,
    "m6i.16xlarge": 64,
    # R family - Memory Optimized (Standard)
    "r6i.large": 2,
    "r6i.xlarge": 4,
    "r6i.2xlarge": 8,
    "r6i.4xlarge": 16,
    "r6i.8xlarge": 32,
    "r6i.12xlarge": 48,
    "r6i.16xlarge": 64,
    # T family - Burstable (Standard)
    "t3.micro": 2,
    "t3.small": 2,
    "t3.medium": 2,
    "t3.large": 2,
    "t3.xlarge": 4,
    "t3.2xlarge": 8,
    # G family - GPU Accelerated
    "g4dn.xlarge": 4,
    "g4dn.2xlarge": 8,
    "g5.xlarge": 4,
    "g5.2xlarge": 8,
    "g5.4xlarge": 16,
    "g5.8xlarge": 32,
    "g6.xlarge": 4,
    "g6.2xlarge": 8,
    "g6.4xlarge": 16,
    "g6.8xlarge": 32,
    "g6.12xlarge": 48,
    "g6.16xlarge": 64,
    "g6.24xlarge": 96,
    "g6.48xlarge": 192,
}

# Instance type to RAM (GB) mapping
RAM_MAP = {
    # C family
    "c6a.xlarge": 8,
    "c6a.4xlarge": 32,
    "c6i.large": 4,
    "c6i.xlarge": 8,
    "c6i.2xlarge": 16,
    "c6i.4xlarge": 32,
    "c6i.8xlarge": 64,
    "c6i.12xlarge": 96,
    "c6i.16xlarge": 128,
    "c6i.24xlarge": 192,
    # M family
    "m5.xlarge": 16,
    "m6i.2xlarge": 32,
    # R family
    "r6i.8xlarge": 256,
    # T family
    "t3.micro": 1,
    # G family
    "g4dn.xlarge": 16,
    "g5.xlarge": 16,
    "g5.2xlarge": 32,
    "g6.xlarge": 16,
    "g6.2xlarge": 32,
    "g6.4xlarge": 64,
    "g6.8xlarge": 128,
    "g6.12xlarge": 192,
    "g6.16xlarge": 256,
    "g6.24xlarge": 384,
    "g6.48xlarge": 768,
}


def get_instance_family(instance_type: str) -> str:
    """Extract the instance family from instance type (e.g., c6i.xlarge -> c6i)."""
    return instance_type.split(".")[0]


def is_standard_instance(instance_type: str) -> bool:
    """Check if instance belongs to Standard family (C, M, R, T)."""
    family = get_instance_family(instance_type)
    return family[0] in ["c", "m", "r", "t"]


def run_aws_command(command: list[str], profile: str) -> Any:
    """Run AWS CLI command and return JSON output."""
    full_command = ["aws"] + command + ["--profile", profile, "--output", "json"]
    try:
        result = subprocess.run(full_command, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running AWS command: {e}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output: {e}", file=sys.stderr)
        sys.exit(1)


def get_service_quota(profile: str, region: str, quota_code: str) -> dict[str, Any]:
    """Get AWS service quota information."""
    command = [
        "service-quotas",
        "get-service-quota",
        "--service-code",
        "ec2",
        "--quota-code",
        quota_code,
        "--region",
        region,
    ]
    result = run_aws_command(command, profile)
    return result.get("Quota", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EC2 usage report for Softmax AWS account")
    parser.add_argument("--profile", default="softmax", help="AWS profile to use (default: softmax)")
    parser.add_argument("--region", default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument(
        "--show-stopped",
        action="store_true",
        help="Include stopped instances in the report",
    )
    args = parser.parse_args()

    # Get all instances
    filters = (
        "Name=instance-state-name,Values=running,pending,stopping,stopped"
        if args.show_stopped
        else "Name=instance-state-name,Values=running,pending"
    )

    command = [
        "ec2",
        "describe-instances",
        "--filters",
        filters,
        "--region",
        args.region,
    ]
    response = run_aws_command(command, args.profile)

    # Parse instances
    instances_by_type: dict[str, list[str]] = defaultdict(list)
    instances_by_state: dict[str, int] = defaultdict(int)
    standard_vcpus_by_state: dict[str, int] = defaultdict(int)
    gpu_vcpus_by_state: dict[str, int] = defaultdict(int)
    total_ram_by_state: dict[str, int] = defaultdict(int)

    for reservation in response.get("Reservations", []):
        for instance in reservation.get("Instances", []):
            instance_type = instance.get("InstanceType", "unknown")
            state = instance["State"]["Name"]

            instances_by_type[instance_type].append(state)
            instances_by_state[state] += 1

            # Count vCPUs
            vcpus = VCPU_MAP.get(instance_type, 0)
            if is_standard_instance(instance_type):
                standard_vcpus_by_state[state] += vcpus
            else:
                gpu_vcpus_by_state[state] += vcpus

            # Count RAM
            ram = RAM_MAP.get(instance_type, 0)
            total_ram_by_state[state] += ram

    # Get service quotas
    print("=" * 80)
    print("EC2 USAGE REPORT")
    print("=" * 80)
    print(f"AWS Profile: {args.profile}")
    print(f"Region: {args.region}")
    print()

    # Service Quotas
    print("📊 SERVICE QUOTAS")
    print("-" * 80)
    try:
        standard_quota = get_service_quota(args.profile, args.region, "L-1216C47A")
        standard_limit = int(standard_quota.get("Value", 0))
        print(f"Standard (C/M/R/T) vCPUs: {standard_limit}")
    except Exception:
        print("Standard (C/M/R/T) vCPUs: Unable to fetch")
        standard_limit = None

    try:
        gpu_quota = get_service_quota(args.profile, args.region, "L-DB2E81BA")
        gpu_limit = int(gpu_quota.get("Value", 0))
        print(f"GPU-accelerated (G) vCPUs: {gpu_limit}")
    except Exception:
        print("GPU-accelerated (G) vCPUs: Unable to fetch")
        gpu_limit = None

    print()

    # Instance summary by state
    print("🖥️  INSTANCE SUMMARY")
    print("-" * 80)
    for state in ["running", "pending", "stopping", "stopped"]:
        count = instances_by_state.get(state, 0)
        if count > 0 or state == "running":
            print(f"{state.capitalize():12} {count:4} instances")
    print()

    # vCPU usage by state
    print("⚙️  vCPU USAGE")
    print("-" * 80)
    for state in ["running", "pending", "stopping", "stopped"]:
        standard = standard_vcpus_by_state.get(state, 0)
        gpu = gpu_vcpus_by_state.get(state, 0)
        if standard > 0 or gpu > 0 or state == "running":
            print(f"{state.capitalize():12} Standard: {standard:4} vCPUs  |  GPU: {gpu:4} vCPUs")

    print()
    print("💡 CAPACITY ANALYSIS")
    print("-" * 80)

    # Standard instance analysis
    standard_running = standard_vcpus_by_state.get("running", 0)
    standard_pending = standard_vcpus_by_state.get("pending", 0)
    standard_total_used = standard_running + standard_pending

    if standard_limit is not None:
        standard_available = standard_limit - standard_total_used
        print("Standard Instances (C/M/R/T):")
        print(f"  Quota:     {standard_limit:4} vCPUs")
        print(f"  Running:   {standard_running:4} vCPUs")
        if standard_pending > 0:
            print(f"  Pending:   {standard_pending:4} vCPUs")
        print(f"  Available: {standard_available:4} vCPUs")
        print()

        # Show what can be launched
        print("  Can launch:")
        for cores in [96, 64, 48, 32, 16, 8, 4, 2]:
            if standard_available >= cores:
                instance_types = [it for it, v in VCPU_MAP.items() if v == cores and is_standard_instance(it)]
                if instance_types:
                    instance_type = instance_types[0]  # Show first matching type
                    count = standard_available // cores
                    ram = RAM_MAP.get(instance_type, 0)
                    print(f"    • {count}x {cores}-core instances (e.g., {instance_type}, {ram}GB RAM)")

    # GPU instance analysis
    gpu_running = gpu_vcpus_by_state.get("running", 0)
    gpu_pending = gpu_vcpus_by_state.get("pending", 0)
    gpu_total_used = gpu_running + gpu_pending

    if gpu_limit is not None:
        gpu_available = gpu_limit - gpu_total_used
        print()
        print("GPU-Accelerated Instances (G):")
        print(f"  Quota:     {gpu_limit:4} vCPUs")
        print(f"  Running:   {gpu_running:4} vCPUs")
        if gpu_pending > 0:
            print(f"  Pending:   {gpu_pending:4} vCPUs")
        print(f"  Available: {gpu_available:4} vCPUs")

    print()

    # Detailed instance breakdown
    print("📋 RUNNING INSTANCES BY TYPE")
    print("-" * 80)

    # Group by family
    families: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for instance_type, states in sorted(instances_by_type.items()):
        family = get_instance_family(instance_type)
        families[family].append((instance_type, states))

    for family in sorted(families.keys()):
        family_instances = families[family]
        is_standard = is_standard_instance(family_instances[0][0])
        family_label = f"{family.upper()} ({'Standard' if is_standard else 'GPU'})"

        print(f"\n{family_label}:")
        for instance_type, states in family_instances:
            running_count = states.count("running")
            stopped_count = states.count("stopped")
            pending_count = states.count("pending")

            if running_count > 0 or (args.show_stopped and stopped_count > 0) or pending_count > 0:
                vcpus = VCPU_MAP.get(instance_type, 0)
                ram = RAM_MAP.get(instance_type, 0)
                status_parts = []
                if running_count > 0:
                    status_parts.append(f"{running_count} running")
                if pending_count > 0:
                    status_parts.append(f"{pending_count} pending")
                if args.show_stopped and stopped_count > 0:
                    status_parts.append(f"{stopped_count} stopped")

                status = ", ".join(status_parts)
                total_count = running_count + pending_count + (stopped_count if args.show_stopped else 0)
                total_vcpus = vcpus * (running_count + pending_count)
                total_ram = ram * (running_count + pending_count)

                print(
                    f"  {instance_type:20} x{total_count:2}  ({status:20})  {total_vcpus:4} vCPUs, {total_ram:5} GB RAM"
                )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
