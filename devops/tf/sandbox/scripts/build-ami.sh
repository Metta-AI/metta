#!/bin/bash
# Automated AMI Builder for Researcher Sandbox
#
# This script automates the process of:
# 1. Launching a temporary GPU instance
# 2. Installing all required software (via setup-ami.sh)
# 3. Creating an AMI from the configured instance
# 4. Cleaning up the temporary instance
#
# Prerequisites:
# - AWS CLI configured with appropriate credentials
# - SSH key pair in AWS (or create one with: aws ec2 create-key-pair --key-name ami-builder)
#
# Usage:
#   cd devops/tf/sandbox
#   ./scripts/build-ami.sh --ssh-key KEY_NAME [--instance-type TYPE]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="$(dirname "$SCRIPT_DIR")"
REGION="us-east-1"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.12xlarge}"
SSH_KEY_NAME="${SSH_KEY_NAME:-}"
AMI_NAME="researcher-sandbox-$(date +%Y%m%d-%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --ssh-key)
      SSH_KEY_NAME="$2"
      shift 2
      ;;
    --instance-type)
      INSTANCE_TYPE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --ssh-key KEY_NAME         SSH key name for accessing instance (required)"
      echo "  --instance-type TYPE       Instance type (default: g5.12xlarge)"
      echo "  --help                     Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run with --help for usage information"
      exit 1
      ;;
  esac
done

# Validate SSH key
if [ -z "$SSH_KEY_NAME" ]; then
  echo -e "${RED}Error: SSH key name is required${NC}"
  echo "Usage: $0 --ssh-key YOUR_KEY_NAME"
  echo ""
  echo "To create a new SSH key:"
  echo "  aws ec2 create-key-pair --key-name ami-builder --query 'KeyMaterial' --output text > ~/.ssh/ami-builder.pem"
  echo "  chmod 400 ~/.ssh/ami-builder.pem"
  exit 1
fi

echo -e "${GREEN}=========================================="
echo "Automated AMI Builder"
echo "==========================================${NC}"
echo ""
echo "Configuration:"
echo "  Region:         $REGION"
echo "  Instance Type:  $INSTANCE_TYPE"
echo "  SSH Key:        $SSH_KEY_NAME"
echo "  AMI Name:       $AMI_NAME"
echo ""

# Step 1: Setup infrastructure for AMI build
echo -e "${YELLOW}[1/6] Setting up infrastructure for AMI build...${NC}"

# Find default VPC
VPC_ID=$(aws ec2 describe-vpcs \
  --region "$REGION" \
  --filters "Name=is-default,Values=true" \
  --query 'Vpcs[0].VpcId' \
  --output text)

if [ "$VPC_ID" == "None" ] || [ -z "$VPC_ID" ]; then
  echo -e "${RED}Error: Default VPC not found in $REGION${NC}"
  exit 1
fi

# Find a public subnet in the default VPC
SUBNET_ID=$(aws ec2 describe-subnets \
  --region "$REGION" \
  --filters "Name=vpc-id,Values=$VPC_ID" "Name=default-for-az,Values=true" \
  --query 'Subnets[0].SubnetId' \
  --output text)

if [ "$SUBNET_ID" == "None" ] || [ -z "$SUBNET_ID" ]; then
  echo -e "${RED}Error: No subnet found in default VPC${NC}"
  exit 1
fi

# Create temporary security group for AMI build
SG_NAME="ami-builder-temp-$(date +%s)"
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
  --region "$REGION" \
  --group-name "$SG_NAME" \
  --description "Temporary security group for AMI builder" \
  --vpc-id "$VPC_ID" \
  --query 'GroupId' \
  --output text)

# Allow SSH from anywhere (temporary, only for AMI build)
aws ec2 authorize-security-group-ingress \
  --region "$REGION" \
  --group-id "$SECURITY_GROUP_ID" \
  --protocol tcp \
  --port 22 \
  --cidr 0.0.0.0/0 \
  > /dev/null

echo "  VPC ID:           $VPC_ID (default)"
echo "  Subnet ID:        $SUBNET_ID"
echo "  Security Group:   $SECURITY_GROUP_ID (temporary)"

# Step 2: Find latest Ubuntu 22.04 AMI with GPU support
echo -e "${YELLOW}[2/6] Finding latest Ubuntu 22.04 AMI...${NC}"
BASE_AMI=$(aws ec2 describe-images \
  --region "$REGION" \
  --owners 099720109477 \
  --filters \
  "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
  "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

echo "  Base AMI: $BASE_AMI"

# Step 3: Launch temporary instance
echo -e "${YELLOW}[3/6] Launching temporary instance...${NC}"
INSTANCE_ID=$(aws ec2 run-instances \
  --region "$REGION" \
  --image-id "$BASE_AMI" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$SSH_KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SECURITY_GROUP_ID" \
  --metadata-options "HttpTokens=required,HttpPutResponseHopLimit=1" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=ami-builder-temp},{Key=Purpose,Value=ami-creation}]" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "  Instance ID: $INSTANCE_ID"

# Cleanup function
cleanup() {
  echo -e "${YELLOW}Cleaning up temporary resources...${NC}"

  # Terminate instance
  if [ -n "${INSTANCE_ID:-}" ]; then
    echo "  Terminating instance..."
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" > /dev/null 2>&1 || true
    aws ec2 wait instance-terminated --region "$REGION" --instance-ids "$INSTANCE_ID" 2>&1 || true
  fi

  # Delete temporary security group
  if [ -n "${SECURITY_GROUP_ID:-}" ]; then
    echo "  Deleting security group..."
    aws ec2 delete-security-group --region "$REGION" --group-id "$SECURITY_GROUP_ID" > /dev/null 2>&1 || true
  fi

  echo -e "${GREEN}Cleanup complete${NC}"
}
trap cleanup EXIT

# Step 4: Wait for instance to be running and get public IP
echo -e "${YELLOW}[4/6] Waiting for instance to be running...${NC}"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
  --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "  Public IP: $PUBLIC_IP"
echo "  Waiting 30 seconds for SSH to be ready..."
sleep 30

# Step 5: Copy setup script and run it
echo -e "${YELLOW}[5/6] Running setup script on instance...${NC}"
echo "  Copying setup script..."
scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$SCRIPT_DIR/setup-ami.sh" ubuntu@"$PUBLIC_IP":/tmp/setup-ami.sh

echo "  Running setup script (this will take 10-15 minutes)..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ubuntu@"$PUBLIC_IP" \
  "bash /tmp/setup-ami.sh"

echo "  Rebooting instance to activate NVIDIA drivers..."
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ubuntu@"$PUBLIC_IP" \
  "sudo reboot" || true

echo "  Waiting 60 seconds for reboot..."
sleep 60

# Wait for instance to be running again
aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"
sleep 30

# Verify GPU is working
echo "  Verifying GPU access..."
if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  ubuntu@"$PUBLIC_IP" \
  "nvidia-smi" > /dev/null 2>&1; then
  echo -e "${GREEN}  ✓ GPU verified${NC}"
else
  echo -e "${RED}  ✗ GPU verification failed${NC}"
  echo "  Continuing anyway, but you may need to debug..."
fi

# Step 6: Stop instance and create AMI
echo -e "${YELLOW}[6/6] Creating AMI...${NC}"
echo "  Stopping instance..."
aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE_ID" > /dev/null
aws ec2 wait instance-stopped --region "$REGION" --instance-ids "$INSTANCE_ID"

echo "  Creating AMI (this may take 5-10 minutes)..."
AMI_ID=$(aws ec2 create-image \
  --region "$REGION" \
  --instance-id "$INSTANCE_ID" \
  --name "$AMI_NAME" \
  --description "Researcher sandbox AMI with Docker, NVIDIA drivers, puffer, cogames" \
  --tag-specifications "ResourceType=image,Tags=[{Key=Name,Value=$AMI_NAME},{Key=Purpose,Value=researcher-sandbox}]" \
  --query 'ImageId' \
  --output text)

echo "  AMI ID: $AMI_ID"
echo "  Waiting for AMI to be available..."
aws ec2 wait image-available --region "$REGION" --image-ids "$AMI_ID"

echo ""
echo -e "${GREEN}=========================================="
echo "AMI Creation Complete!"
echo "==========================================${NC}"
echo ""
echo "AMI Details:"
echo "  ID:     $AMI_ID"
echo "  Name:   $AMI_NAME"
echo "  Region: $REGION"
echo ""
echo "Next steps:"
echo "  1. Update FastAPI configuration to use AMI: $AMI_ID"
echo "  2. Test the AMI by launching a sandbox instance"
echo ""
