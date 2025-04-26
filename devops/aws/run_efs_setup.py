#!/usr/bin/env python3
import argparse
import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from setup_efs_with_tailscale import EFSTailscaleSetup


def parse_args():
    parser = argparse.ArgumentParser(description="Set up an AWS EFS volume with Tailscale subnet router")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", type=str, default=None, help="AWS profile name (optional)")
    parser.add_argument("--tailscale-auth-key", type=str, required=True, help="Tailscale authentication key")
    parser.add_argument("--instance-type", type=str, default="t4g.nano", help="EC2 instance type (default: t4g.nano)")
    parser.add_argument("--vpc-id", type=str, default=None, help="Existing VPC ID (optional)")
    parser.add_argument("--subnet-id", type=str, default=None, help="Existing subnet ID (optional)")
    parser.add_argument("--efs-id", type=str, default=None, help="Existing EFS ID (optional)")
    parser.add_argument("--key-name", type=str, default=None, help="EC2 key pair name for SSH access (optional)")
    return parser.parse_args()


def main():
    args = parse_args()
    setup = EFSTailscaleSetup(args.region, args.profile)
    result = setup.setup(
        tailscale_auth_key=args.tailscale_auth_key,
        instance_type=args.instance_type,
        vpc_id=args.vpc_id,
        subnet_id=args.subnet_id,
        efs_id=args.efs_id,
        key_name=args.key_name,
    )
    print("\nSetup completed successfully!")
    print("To mount the EFS on your Mac:")
    print("1. Install and log in to Tailscale on your Mac")
    print("2. Create a mount point: sudo mkdir -p /path/to/mount")
    print(f"3. Mount the EFS: {result['mount_command']}")


if __name__ == "__main__":
    main()
