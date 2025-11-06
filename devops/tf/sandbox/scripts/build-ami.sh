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
# - Terraform has been applied (creates VPC, security groups, etc.)
# - AWS CLI configured with appropriate credentials
# - SSH key for accessing instances
#
# Usage:
#   cd devops/tf/sandbox
#   ./scripts/build-ami.sh [--ssh-key KEY_NAME] [--instance-type TYPE]

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

# Step 1: Get Terraform outputs
echo -e "${YELLOW}[1/7] Reading Terraform outputs...${NC}"
cd "$TF_DIR"

if [ ! -f "terraform.tfstate" ]; then
    echo -e "${RED}Error: terraform.tfstate not found. Please run 'terraform apply' first.${NC}"
    exit 1
fi

SUBNET_ID=$(terraform output -json public_subnet_ids | jq -r '.[0]')
SECURITY_GROUP_ID=$(terraform output -raw security_group_id)
VPC_ID=$(terraform output -raw vpc_id)

echo "  VPC ID:           $VPC_ID"
echo "  Subnet ID:        $SUBNET_ID"
echo "  Security Group:   $SECURITY_GROUP_ID"

# Step 2: Find latest Ubuntu 22.04 AMI with GPU support
echo -e "${YELLOW}[2/7] Finding latest Ubuntu 22.04 AMI...${NC}"
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
echo -e "${YELLOW}[3/7] Launching temporary instance...${NC}"
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
    echo -e "${YELLOW}Cleaning up temporary instance...${NC}"
    aws ec2 terminate-instances --region "$REGION" --instance-ids "$INSTANCE_ID" > /dev/null || true
    echo -e "${GREEN}Cleanup complete${NC}"
}
trap cleanup EXIT

# Step 4: Wait for instance to be running and get public IP
echo -e "${YELLOW}[4/7] Waiting for instance to be running...${NC}"
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
echo -e "${YELLOW}[5/7] Running setup script on instance...${NC}"
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
echo -e "${YELLOW}[6/7] Creating AMI...${NC}"
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

# Step 7: Update Terraform config
echo -e "${YELLOW}[7/7] Updating Terraform configuration...${NC}"
if [ -f "$TF_DIR/terraform.tfvars" ]; then
    # Update existing file
    if grep -q "sandbox_ami_id" "$TF_DIR/terraform.tfvars"; then
        sed -i.bak "s|sandbox_ami_id.*|sandbox_ami_id = \"$AMI_ID\"|" "$TF_DIR/terraform.tfvars"
        rm "$TF_DIR/terraform.tfvars.bak"
        echo "  Updated terraform.tfvars"
    else
        echo "" >> "$TF_DIR/terraform.tfvars"
        echo "sandbox_ami_id = \"$AMI_ID\"" >> "$TF_DIR/terraform.tfvars"
        echo "  Added AMI ID to terraform.tfvars"
    fi
else
    # Create new file
    cat > "$TF_DIR/terraform.tfvars" << EOF
# Researcher Sandbox Configuration
# Generated by build-ami.sh on $(date)

region      = "$REGION"
environment = "production"

# AMI Configuration
sandbox_ami_id = "$AMI_ID"

# Instance Configuration
default_instance_type = "g5.12xlarge"
EOF
    echo "  Created terraform.tfvars with AMI ID"
fi

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
echo "  1. Apply Terraform to use new AMI:"
echo "     cd $TF_DIR && terraform apply"
echo ""
echo "  2. Test the AMI by launching a sandbox instance"
echo ""
echo "The AMI ID has been saved to terraform.tfvars"
echo ""
