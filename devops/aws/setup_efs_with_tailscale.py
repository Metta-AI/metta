#!/usr/bin/env python3
"""
AWS EFS with Tailscale Setup Script

This script sets up a Tailscale subnet router on a small EC2 instance to enable
mounting an AWS EFS volume from your local Mac without having to maintain a large instance.

Requirements:
- Python 3.6+
- boto3
- Tailscale account with an auth key
"""

import argparse
import base64
import json
import sys
import time
from typing import Any, Dict, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("Error: boto3 is not installed. Please install it using 'pip install boto3'")
    sys.exit(1)


class EFSTailscaleSetup:
    def __init__(self, region_name: str, profile_name: Optional[str] = None):
        """Initialize the EFS Tailscale Setup.

        Args:
            region_name: AWS region name where resources will be created
            profile_name: AWS profile name to use (optional)
        """
        self.region = region_name
        self.profile_name = profile_name

        # Set up AWS session
        if profile_name:
            self.session = boto3.Session(profile_name=profile_name, region_name=region_name)
        else:
            self.session = boto3.Session(region_name=region_name)

        # Initialize clients
        self.ec2 = self.session.client("ec2")
        self.efs = self.session.client("efs")
        self.iam = self.session.client("iam")
        self.ssm = self.session.client("ssm")

        # Initialize properties
        self.vpc_id = None
        self.subnet_id = None
        self.security_group_id = None
        self.instance_id = None
        self.efs_id = None
        self.efs_mount_target_id = None
        self.efs_mount_target_ip = None
        self.instance_profile_name = None
        self.role_name = None

    def setup(
        self,
        tailscale_auth_key: str,
        instance_type: str = "t4g.nano",
        vpc_id: Optional[str] = None,
        subnet_id: Optional[str] = None,
        efs_id: Optional[str] = None,
        key_name: Optional[str] = None,
    ) -> dict:
        """Set up the complete environment.

        Args:
            tailscale_auth_key: Tailscale authentication key
            instance_type: EC2 instance type (default: t4g.nano)
            vpc_id: Existing VPC ID (optional)
            subnet_id: Existing subnet ID (optional)
            efs_id: Existing EFS ID (optional)
            key_name: EC2 key pair name for SSH access (optional)

        Returns:
            dict: Setup information including IDs and connection details
        """
        # Store parameters
        self.vpc_id = vpc_id
        self.subnet_id = subnet_id
        self.efs_id = efs_id

        # Setup steps
        print("Starting EFS-Tailscale setup...")

        # Step 1: Set up networking if needed
        if not self.vpc_id or not self.subnet_id:
            self.setup_networking()
            print(f"Created VPC: {self.vpc_id} and Subnet: {self.subnet_id}")

        # Step 2: Create security group
        self.setup_security_group()
        print(f"Created Security Group: {self.security_group_id}")

        # Step 3: Create IAM role and instance profile
        self.setup_iam_role()
        print(f"Created IAM Role: {self.role_name} and Instance Profile: {self.instance_profile_name}")

        # Step 4: Set up EFS if needed
        if not self.efs_id:
            self.setup_efs()
            print(f"Created EFS: {self.efs_id} with Mount Target: {self.efs_mount_target_id}")
        else:
            self.get_efs_info()
            print(f"Using existing EFS: {self.efs_id}")

        # Step 5: Launch EC2 instance
        self.launch_instance(tailscale_auth_key, instance_type, key_name)
        print(f"Launched EC2 Instance: {self.instance_id}")

        # Wait for instance to initialize
        print("Waiting for instance to initialize (this may take a few minutes)...")
        self.wait_for_instance()

        # Get instance details
        instance_details = self.describe_instance()
        public_ip = instance_details.get("PublicIpAddress", "N/A")

        result = {
            "vpc_id": self.vpc_id,
            "subnet_id": self.subnet_id,
            "security_group_id": self.security_group_id,
            "efs_id": self.efs_id,
            "efs_mount_target_id": self.efs_mount_target_id,
            "efs_mount_target_ip": self.efs_mount_target_ip,
            "instance_id": self.instance_id,
            "instance_public_ip": public_ip,
            "mount_command": f"sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576 {self.efs_id}.efs.{self.region}.amazonaws.com:/ /path/to/mount",
        }

        print("\n=== Setup Complete ===")
        print(f"EFS ID: {self.efs_id}")
        print(f"EC2 Instance ID: {self.instance_id} (Tailscale Subnet Router)")
        print(f"EC2 Public IP: {public_ip}")
        print("\nTo mount the EFS on your Mac:")
        print("1. Install and log in to Tailscale on your Mac")
        print("2. Create a mount point: sudo mkdir -p /path/to/mount")
        print(
            f"3. Mount the EFS: sudo mount -t nfs -o nfsvers=4.1,rsize=1048576,wsize=1048576 {self.efs_id}.efs.{self.region}.amazonaws.com:/ /path/to/mount"
        )

        return result

    def setup_networking(self) -> None:
        """Set up VPC and subnet if not provided."""
        try:
            # Create VPC
            response = self.ec2.create_vpc(
                CidrBlock="10.0.0.0/16",
                TagSpecifications=[{"ResourceType": "vpc", "Tags": [{"Key": "Name", "Value": "efs-tailscale-vpc"}]}],
            )
            self.vpc_id = response["Vpc"]["VpcId"]

            # Enable DNS support
            self.ec2.modify_vpc_attribute(VpcId=self.vpc_id, EnableDnsSupport={"Value": True})
            self.ec2.modify_vpc_attribute(VpcId=self.vpc_id, EnableDnsHostnames={"Value": True})

            # Create internet gateway
            igw_response = self.ec2.create_internet_gateway(
                TagSpecifications=[
                    {"ResourceType": "internet-gateway", "Tags": [{"Key": "Name", "Value": "efs-tailscale-igw"}]}
                ]
            )
            igw_id = igw_response["InternetGateway"]["InternetGatewayId"]

            # Attach gateway to VPC
            self.ec2.attach_internet_gateway(InternetGatewayId=igw_id, VpcId=self.vpc_id)

            # Create subnet
            subnet_response = self.ec2.create_subnet(
                VpcId=self.vpc_id,
                CidrBlock="10.0.0.0/24",
                TagSpecifications=[
                    {"ResourceType": "subnet", "Tags": [{"Key": "Name", "Value": "efs-tailscale-subnet"}]}
                ],
            )
            self.subnet_id = subnet_response["Subnet"]["SubnetId"]

            # Enable auto-assign public IP
            self.ec2.modify_subnet_attribute(SubnetId=self.subnet_id, MapPublicIpOnLaunch={"Value": True})

            # Create route table
            route_table_response = self.ec2.create_route_table(
                VpcId=self.vpc_id,
                TagSpecifications=[
                    {"ResourceType": "route-table", "Tags": [{"Key": "Name", "Value": "efs-tailscale-rt"}]}
                ],
            )
            route_table_id = route_table_response["RouteTable"]["RouteTableId"]

            # Create route to Internet Gateway
            self.ec2.create_route(RouteTableId=route_table_id, DestinationCidrBlock="0.0.0.0/0", GatewayId=igw_id)

            # Associate route table with subnet
            self.ec2.associate_route_table(RouteTableId=route_table_id, SubnetId=self.subnet_id)

            # Wait for VPC to be available
            waiter = self.ec2.get_waiter("vpc_available")
            waiter.wait(VpcIds=[self.vpc_id])

        except ClientError as e:
            print(f"Error setting up networking: {e}")
            sys.exit(1)

    def setup_security_group(self) -> None:
        """Set up security group for EFS and EC2."""
        try:
            # Create security group
            response = self.ec2.create_security_group(
                GroupName="efs-tailscale-sg",
                Description="Security group for EFS with Tailscale",
                VpcId=self.vpc_id,
                TagSpecifications=[
                    {"ResourceType": "security-group", "Tags": [{"Key": "Name", "Value": "efs-tailscale-sg"}]}
                ],
            )
            self.security_group_id = response["GroupId"]

            # Add inbound rules
            self.ec2.authorize_security_group_ingress(
                GroupId=self.security_group_id,
                IpPermissions=[
                    # SSH access
                    {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                    # NFS access
                    {"IpProtocol": "tcp", "FromPort": 2049, "ToPort": 2049, "IpRanges": [{"CidrIp": "10.0.0.0/16"}]},
                    # Tailscale local network (UDP)
                    {"IpProtocol": "udp", "FromPort": 41641, "ToPort": 41641, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]},
                ],
            )

            # Add outbound rule (allow all)
            self.ec2.authorize_security_group_egress(
                GroupId=self.security_group_id,
                IpPermissions=[
                    {"IpProtocol": "-1", "FromPort": -1, "ToPort": -1, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}
                ],
            )

        except ClientError as e:
            if "InvalidPermission.Duplicate" in str(e):
                print("Security group rules already exist, continuing...")
            else:
                print(f"Error setting up security group: {e}")
                sys.exit(1)

    def setup_iam_role(self) -> None:
        """Set up IAM role for EC2 instance."""
        try:
            self.role_name = "efs-tailscale-role"
            self.instance_profile_name = "efs-tailscale-instance-profile"

            # Create IAM role
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}
                ],
            }

            try:
                self.iam.create_role(
                    RoleName=self.role_name,
                    AssumeRolePolicyDocument=json.dumps(assume_role_policy),
                    Description="Role for EFS-Tailscale EC2 instance",
                )
            except ClientError as e:
                if "EntityAlreadyExists" in str(e):
                    print(f"IAM role {self.role_name} already exists, reusing it.")
                else:
                    raise

            # Attach policy for SSM
            self.iam.attach_role_policy(
                RoleName=self.role_name, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
            )

            # Create instance profile
            try:
                self.iam.create_instance_profile(InstanceProfileName=self.instance_profile_name)
            except ClientError as e:
                if "EntityAlreadyExists" in str(e):
                    print(f"Instance profile {self.instance_profile_name} already exists, reusing it.")
                else:
                    raise

            # Check if role is already attached to instance profile
            try:
                response = self.iam.get_instance_profile(InstanceProfileName=self.instance_profile_name)
                roles = [role["RoleName"] for role in response["InstanceProfile"]["Roles"]]
                if self.role_name not in roles:
                    # Add role to instance profile
                    self.iam.add_role_to_instance_profile(
                        InstanceProfileName=self.instance_profile_name, RoleName=self.role_name
                    )
            except ClientError as e:
                print(f"Error checking instance profile: {e}")
                sys.exit(1)

            # Wait for role to propagate
            print("Waiting for IAM role to propagate...")
            time.sleep(10)

        except ClientError as e:
            print(f"Error setting up IAM role: {e}")
            sys.exit(1)

    def setup_efs(self) -> None:
        """Set up EFS and mount target."""
        try:
            # Create EFS file system
            response = self.efs.create_file_system(
                PerformanceMode="generalPurpose", Encrypted=True, Tags=[{"Key": "Name", "Value": "efs-tailscale-fs"}]
            )
            self.efs_id = response["FileSystemId"]

            # Wait for EFS to become available
            print("Waiting for EFS to become available...")
            waiter = None
            while True:
                response = self.efs.describe_file_systems(FileSystemId=self.efs_id)
                status = response["FileSystems"][0]["LifeCycleState"]
                if status == "available":
                    break
                print(f"EFS status: {status}, waiting...")
                time.sleep(5)

            # Create mount target
            response = self.efs.create_mount_target(
                FileSystemId=self.efs_id, SubnetId=self.subnet_id, SecurityGroups=[self.security_group_id]
            )
            self.efs_mount_target_id = response["MountTargetId"]
            self.efs_mount_target_ip = response["IpAddress"]

            # Wait for mount target to become available
            print("Waiting for EFS mount target to become available...")
            while True:
                response = self.efs.describe_mount_targets(MountTargetId=self.efs_mount_target_id)
                status = response["MountTargets"][0]["LifeCycleState"]
                if status == "available":
                    break
                print(f"Mount target status: {status}, waiting...")
                time.sleep(5)

        except ClientError as e:
            print(f"Error setting up EFS: {e}")
            sys.exit(1)

    def get_efs_info(self) -> None:
        """Get information about existing EFS file system."""
        try:
            # Get EFS information
            response = self.efs.describe_file_systems(FileSystemId=self.efs_id)

            # Get mount target information
            mt_response = self.efs.describe_mount_targets(FileSystemId=self.efs_id)

            if mt_response["MountTargets"]:
                self.efs_mount_target_id = mt_response["MountTargets"][0]["MountTargetId"]
                self.efs_mount_target_ip = mt_response["MountTargets"][0]["IpAddress"]
            else:
                # Create a mount target if none exists
                print("Creating mount target for existing EFS...")
                response = self.efs.create_mount_target(
                    FileSystemId=self.efs_id, SubnetId=self.subnet_id, SecurityGroups=[self.security_group_id]
                )
                self.efs_mount_target_id = response["MountTargetId"]
                self.efs_mount_target_ip = response["IpAddress"]

                # Wait for mount target to become available
                print("Waiting for EFS mount target to become available...")
                while True:
                    response = self.efs.describe_mount_targets(MountTargetId=self.efs_mount_target_id)
                    status = response["MountTargets"][0]["LifeCycleState"]
                    if status == "available":
                        break
                    print(f"Mount target status: {status}, waiting...")
                    time.sleep(5)

        except ClientError as e:
            print(f"Error getting EFS information: {e}")
            sys.exit(1)

    def launch_instance(self, tailscale_auth_key: str, instance_type: str, key_name: Optional[str]) -> None:
        """Launch EC2 instance with Tailscale subnet router.

        Args:
            tailscale_auth_key: Tailscale authentication key
            instance_type: EC2 instance type
            key_name: EC2 key pair name for SSH access (optional)
        """
        try:
            # Get latest Amazon Linux 2023 AMI ID (ARM-based for t4g instances)
            ami_id = self.get_latest_ami()

            # Prepare user data script
            user_data = self.generate_user_data_script(tailscale_auth_key)
            user_data_b64 = base64.b64encode(user_data.encode()).decode()

            # Prepare launch parameters
            launch_params = {
                "ImageId": ami_id,
                "InstanceType": instance_type,
                "MaxCount": 1,
                "MinCount": 1,
                "SecurityGroupIds": [self.security_group_id],
                "SubnetId": self.subnet_id,
                "UserData": user_data_b64,
                "IamInstanceProfile": {"Name": self.instance_profile_name},
                "TagSpecifications": [
                    {"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": "efs-tailscale-router"}]}
                ],
                "InstanceInitiatedShutdownBehavior": "stop",
            }

            # Add key pair if provided
            if key_name:
                launch_params["KeyName"] = key_name

            # Launch instance
            response = self.ec2.run_instances(**launch_params)
            self.instance_id = response["Instances"][0]["InstanceId"]

        except ClientError as e:
            print(f"Error launching EC2 instance: {e}")
            sys.exit(1)

    def get_latest_ami(self) -> str:
        """Get the latest Amazon Linux 2023 AMI ID for ARM architecture."""
        try:
            response = self.ec2.describe_images(
                Owners=["amazon"],
                Filters=[
                    {"Name": "name", "Values": ["al2023-ami-2023*-arm64"]},
                    {"Name": "state", "Values": ["available"]},
                    {"Name": "architecture", "Values": ["arm64"]},
                ],
            )

            if not response["Images"]:
                # Fallback to fixed AMI ID if no images found
                print("No ARM AMIs found, falling back to fixed AMI ID...")
                return self.get_fallback_ami()

            # Sort by creation date (newest first)
            images = sorted(response["Images"], key=lambda x: x["CreationDate"], reverse=True)
            return images[0]["ImageId"]

        except ClientError as e:
            print(f"Error getting AMI: {e}, falling back to fixed AMI ID...")
            return self.get_fallback_ami()

    def get_fallback_ami(self) -> str:
        """Get a fallback AMI ID based on the region."""
        # These AMI IDs can become outdated, but provide a fallback
        ami_mapping = {
            "us-east-1": "ami-0a0c8eebcdd6dcbd0",  # Amazon Linux 2023 ARM64
            "us-east-2": "ami-0e0bf53f6def86294",
            "us-west-1": "ami-0a1b92c776846fbd0",
            "us-west-2": "ami-0cf29c27d6f5ae495",
            "eu-west-1": "ami-0e2d0aefd3fc769fe",
            "eu-central-1": "ami-0c2d5a72f9e863874",
            "ap-northeast-1": "ami-0f5d8cdf2eb047b3c",
            "ap-southeast-1": "ami-02045ebddb047018b",
            "ap-southeast-2": "ami-0e7a98cb8329c2960",
        }

        return ami_mapping.get(self.region, "ami-0a0c8eebcdd6dcbd0")  # Default to us-east-1

    def generate_user_data_script(self, tailscale_auth_key: str) -> str:
        """Generate user data script for EC2 instance."""
        return f"""#!/bin/bash
# Update system
yum update -y

# Install required packages
yum install -y amazon-efs-utils nfs-utils

# Install Tailscale
yum install -y yum-utils
yum-config-manager --add-repo https://pkgs.tailscale.com/stable/amazon-linux/2/tailscale.repo
yum install -y tailscale

# Start Tailscale
systemctl enable --now tailscaled
tailscale up --authkey={tailscale_auth_key} --advertise-routes=10.0.0.0/16 --accept-routes --hostname=efs-subnet-router

# Enable IP forwarding for subnet router functionality
echo 'net.ipv4.ip_forward = 1' | tee -a /etc/sysctl.conf
sysctl -p /etc/sysctl.conf

# Create directory for EFS mount
mkdir -p /mnt/efs

# Mount EFS
mount -t efs {self.efs_id}:/ /mnt/efs

# Add to fstab for persistence
echo "{self.efs_id}:/ /mnt/efs efs defaults,_netdev 0 0" >> /etc/fstab

# Set permissions on EFS mount
chmod 777 /mnt/efs

# Send signal that setup is complete
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 create-tags --resources $INSTANCE_ID --tags Key=Status,Value=SetupComplete --region {self.region}
"""

    def wait_for_instance(self) -> None:
        """Wait for instance to initialize."""
        try:
            # Wait for instance to be running
            waiter = self.ec2.get_waiter("instance_running")
            waiter.wait(InstanceIds=[self.instance_id])

            # Wait for status checks to pass
            waiter = self.ec2.get_waiter("instance_status_ok")
            waiter.wait(InstanceIds=[self.instance_id])

            # Wait for Tailscale setup to complete (polling for tag)
            max_wait_time = 300  # 5 minutes
            start_time = time.time()
            while time.time() - start_time < max_wait_time:
                response = self.ec2.describe_tags(
                    Filters=[
                        {"Name": "resource-id", "Values": [self.instance_id]},
                        {"Name": "key", "Values": ["Status"]},
                        {"Name": "value", "Values": ["SetupComplete"]},
                    ]
                )
                if response["Tags"]:
                    print("Tailscale setup completed successfully!")
                    return

                print("Waiting for Tailscale setup to complete...")
                time.sleep(15)

            print("Timeout waiting for Tailscale setup to complete, but instance is running.")
            print("Setup might still be in progress.")

        except ClientError as e:
            print(f"Error waiting for instance: {e}")

    def describe_instance(self) -> Dict[str, Any]:
        """Get details about the EC2 instance."""
        try:
            response = self.ec2.describe_instances(InstanceIds=[self.instance_id])
            return response["Reservations"][0]["Instances"][0]
        except ClientError as e:
            print(f"Error describing instance: {e}")
            return {}

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.instance_id:
                print(f"Terminating EC2 instance {self.instance_id}...")
                self.ec2.terminate_instances(InstanceIds=[self.instance_id])

            if self.efs_mount_target_id:
                print(f"Deleting EFS mount target {self.efs_mount_target_id}...")
                self.efs.delete_mount_target(MountTargetId=self.efs_mount_target_id)

                # Wait for mount target to be deleted
                while True:
                    try:
                        self.efs.describe_mount_targets(MountTargetId=self.efs_mount_target_id)
                        print("Waiting for mount target to be deleted...")
                        time.sleep(5)
                    except:
                        break

            if self.efs_id:
                print(f"Deleting EFS file system {self.efs_id}...")
                self.efs.delete_file_system(FileSystemId=self.efs_id)

            if self.security_group_id:
                print(f"Deleting security group {self.security_group_id}...")
                self.ec2.delete_security_group(GroupId=self.security_group_id)

            if self.instance_profile_name:
                print(f"Detaching role from instance profile {self.instance_profile_name}...")
                try:
                    self.iam.remove_role_from_instance_profile(
                        InstanceProfileName=self.instance_profile_name, RoleName=self.role_name
                    )
                except:
                    pass

                print(f"Deleting instance profile {self.instance_profile_name}...")
                self.iam.delete_instance_profile(InstanceProfileName=self.instance_profile_name)

            if self.role_name:
                print(f"Detaching policies from role {self.role_name}...")
                for policy in self.iam.list_attached_role_policies(RoleName=self.role_name)["AttachedPolicies"]:
                    self.iam.detach_role_policy(RoleName=self.role_name, PolicyArn=policy["PolicyArn"])

                print(f"Deleting role {self.role_name}...")
                self.iam.delete_role(RoleName=self.role_name)

            # Note: VPC and subnet cleanup omitted for safety
            # These often have dependencies and require specific order for cleanup

            print("Cleanup completed.")

        except ClientError as e:
            print(f"Error during cleanup: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up an AWS EFS volume with Tailscale subnet router")

    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region (default: us-east-1)")
    parser.add_argument("--profile", type=str, default=None, help="AWS profile name (optional)")
    parser.add_argument("--tailscale-auth-key", type=str, required=True, help="Tailscale authentication key")
    parser.add_argument("--instance-type", type=str, default="t4g.nano", help="EC2 instance type (default: t4g.nano)")
    parser.add_argument("--vpc-id", type=str, default=None, help="Existing VPC ID (optional)")
    parser.add_argument("--subnet-id", type=str, default=None, help="Existing subnet ID (optional)")
    parser.add_argument
