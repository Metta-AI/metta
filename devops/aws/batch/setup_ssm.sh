#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to display messages
print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    command -v $1 >/dev/null 2>&1 || {
        print_error "Required command '$1' not found. Please install it first.";
        exit 1;
    }
}

# Check for AWS CLI
check_command aws
check_command jq

# Introduction
echo "======================================================"
echo "AWS SSM Session Manager Setup for AWS Batch Instances"
echo "======================================================"
echo ""
print_message "This script will help you configure AWS Systems Manager Session Manager"
print_message "to access your AWS Batch instances without using SSH keys."
print_message "You will be using your existing ecsTaskExecution role."
echo ""
print_message "The script will:"
print_message "1. Check and configure AWS credentials"
print_message "2. Verify your AWS Batch compute environment configuration"
print_message "3. Create or update necessary IAM policies"
print_message "4. Set up VPC endpoints if needed"
print_message "5. Verify SSM agent is running on instances"
echo ""
read -p "Press Enter to continue..."
echo ""

# AWS Credentials setup
print_message "Setting up AWS credentials..."

read -p "Enter your AWS Access Key ID: " aws_access_key_id
read -p "Enter your AWS Secret Access Key: " aws_secret_access_key
read -p "Enter your AWS Session Token (press Enter if using long-term credentials): " aws_session_token
read -p "Enter your preferred AWS region (e.g., us-west-2): " aws_region

export AWS_ACCESS_KEY_ID=$aws_access_key_id
export AWS_SECRET_ACCESS_KEY=$aws_secret_access_key
export AWS_DEFAULT_REGION=$aws_region

# If session token was provided, export it as well
if [[ ! -z "$aws_session_token" ]]; then
    export AWS_SESSION_TOKEN=$aws_session_token
    print_message "Using temporary credentials with session token."
fi

# Verify credentials
print_message "Verifying AWS credentials..."
aws_account_id=$(aws sts get-caller-identity --query "Account" --output text 2>/dev/null)

if [ $? -ne 0 ]; then
    print_error "Failed to authenticate with AWS. Please check your credentials and try again."
    exit 1
fi

print_success "AWS credentials verified successfully. Account ID: $aws_account_id"
echo ""

# Check ecsTaskExecution role
print_message "Checking ecsTaskExecution role..."
role_exists=$(aws iam get-role --role-name ecsTaskExecutionRole 2>/dev/null || echo "not_found")

if [[ $role_exists == "not_found" ]]; then
    print_warning "ecsTaskExecutionRole not found in your account."
    print_message "This is the default AWS managed role. You will need to:"
    print_message "1. Go to IAM console: https://console.aws.amazon.com/iam/"
    print_message "2. Create a role named 'ecsTaskExecutionRole' with the AmazonECSTaskExecutionRolePolicy"
    echo ""
    read -p "Press Enter to continue once you've created the role, or Ctrl+C to exit..."
fi

# Check if SSM managed policy is attached to role
print_message "Checking if SSM managed policy is attached to ecsTaskExecutionRole..."
ssm_policy_attached=$(aws iam list-attached-role-policies --role-name ecsTaskExecutionRole --query "AttachedPolicies[?PolicyName=='AmazonSSMManagedInstanceCore'].PolicyName" --output text)

if [[ -z $ssm_policy_attached ]]; then
    print_message "Attaching AmazonSSMManagedInstanceCore policy to ecsTaskExecutionRole..."
    aws iam attach-role-policy --role-name ecsTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

    if [ $? -ne 0 ]; then
        print_error "Failed to attach AmazonSSMManagedInstanceCore policy."
        print_message "Please manually attach this policy to ecsTaskExecutionRole:"
        print_message "1. Go to IAM console: https://console.aws.amazon.com/iam/"
        print_message "2. Find and select the 'ecsTaskExecutionRole'"
        print_message "3. Click 'Attach policies'"
        print_message "4. Search for and attach 'AmazonSSMManagedInstanceCore'"
        echo ""
        read -p "Press Enter to continue once you've attached the policy..."
    else
        print_success "AmazonSSMManagedInstanceCore policy attached successfully."
    fi
else
    print_success "AmazonSSMManagedInstanceCore policy is already attached."
fi

# Check AWS Batch compute environment
print_message "Checking AWS Batch compute environments..."
compute_envs=$(aws batch describe-compute-environments --query "computeEnvironments[].computeEnvironmentName" --output json 2>/dev/null)

if [ $? -ne 0 ]; then
    print_error "Failed to retrieve AWS Batch compute environments."
    print_message "Please make sure you have the necessary permissions and AWS Batch is set up."
    exit 1
fi

if [[ $compute_envs == "[]" ]]; then
    print_warning "No AWS Batch compute environments found."
    print_message "You need to create a compute environment before continuing:"
    print_message "1. Go to AWS Batch console: https://console.aws.amazon.com/batch/"
    print_message "2. Create a compute environment with the ecsTaskExecutionRole"
    echo ""
    read -p "Press Enter to continue once you've created a compute environment, or Ctrl+C to exit..."
else
    print_success "Found AWS Batch compute environments: $compute_envs"

    # Select compute environment
    echo ""
    print_message "Which compute environment would you like to update? (Enter the name)"
    read compute_env_name

    # Verify compute environment exists
    compute_env_exists=$(aws batch describe-compute-environments --compute-environments $compute_env_name --query "computeEnvironments[0].computeEnvironmentName" --output text 2>/dev/null)

    if [[ -z $compute_env_exists || $compute_env_exists == "None" ]]; then
        print_error "Compute environment '$compute_env_name' not found."
        exit 1
    fi

    # Get compute environment details
    print_message "Getting details for compute environment '$compute_env_name'..."
    compute_env_details=$(aws batch describe-compute-environments --compute-environments $compute_env_name --output json)

    # Extract instance role
    instance_role=$(echo $compute_env_details | jq -r '.computeEnvironments[0].computeResources.instanceRole')

    if [[ -z $instance_role || $instance_role == "null" ]]; then
        print_warning "Could not determine instance role for compute environment."
        print_message "Please make sure your compute environment is configured with an instance role."
        print_message "You will need to manually update your compute environment to use ecsTaskExecutionRole."
    else
        print_success "Current instance role: $instance_role"

        # Check if instance role has SSM policy
        instance_role_name=$(echo $instance_role | sed 's/.*\///')
        ssm_policy_attached_to_instance=$(aws iam list-attached-role-policies --role-name $instance_role_name --query "AttachedPolicies[?PolicyName=='AmazonSSMManagedInstanceCore'].PolicyName" --output text 2>/dev/null)

        if [[ $? -ne 0 ]]; then
            print_warning "Could not check policies for instance role. You may need to attach the SSM policy manually."
        elif [[ -z $ssm_policy_attached_to_instance ]]; then
            print_message "Attaching AmazonSSMManagedInstanceCore policy to instance role..."
            aws iam attach-role-policy --role-name $instance_role_name --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

            if [ $? -ne 0 ]; then
                print_error "Failed to attach AmazonSSMManagedInstanceCore policy to instance role."
                print_message "Please manually attach this policy to $instance_role_name:"
                print_message "1. Go to IAM console: https://console.aws.amazon.com/iam/"
                print_message "2. Find and select the role '$instance_role_name'"
                print_message "3. Click 'Attach policies'"
                print_message "4. Search for and attach 'AmazonSSMManagedInstanceCore'"
                echo ""
                read -p "Press Enter to continue once you've attached the policy..."
            else
                print_success "AmazonSSMManagedInstanceCore policy attached successfully to instance role."
            fi
        else
            print_success "AmazonSSMManagedInstanceCore policy is already attached to instance role."
        fi
    fi
fi

# Check/Create VPC endpoints for SSM
print_message "Checking VPC configuration for SSM endpoints..."

# Get VPC ID from compute environment
vpcs=$(aws ec2 describe-vpcs --query "Vpcs[].VpcId" --output json)
if [[ $vpcs == "[]" ]]; then
    print_error "No VPCs found in your account."
    exit 1
fi

echo "Available VPCs:"
aws ec2 describe-vpcs --query "Vpcs[].[VpcId,Tags[?Key=='Name'].Value|[0]]" --output table

read -p "Enter the VPC ID for your AWS Batch compute environment: " vpc_id

# Get subnet information
subnets=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$vpc_id" --query "Subnets[].SubnetId" --output json)
if [[ $subnets == "[]" ]]; then
    print_error "No subnets found in VPC $vpc_id."
    exit 1
fi

print_success "Found subnets in VPC $vpc_id: $subnets"

# Check for existing SSM endpoints
print_message "Checking for existing SSM VPC endpoints..."

ssm_endpoint=$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=$vpc_id" "Name=service-name,Values=com.amazonaws.$aws_region.ssm" --query "VpcEndpoints[0].VpcEndpointId" --output text 2>/dev/null)
ec2messages_endpoint=$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=$vpc_id" "Name=service-name,Values=com.amazonaws.$aws_region.ec2messages" --query "VpcEndpoints[0].VpcEndpointId" --output text 2>/dev/null)
ssmmessages_endpoint=$(aws ec2 describe-vpc-endpoints --filters "Name=vpc-id,Values=$vpc_id" "Name=service-name,Values=com.amazonaws.$aws_region.ssmmessages" --query "VpcEndpoints[0].VpcEndpointId" --output text 2>/dev/null)

endpoints_to_create=()

if [[ -z $ssm_endpoint || $ssm_endpoint == "None" ]]; then
    print_message "SSM endpoint not found. Will create."
    endpoints_to_create+=("com.amazonaws.$aws_region.ssm")
else
    print_success "SSM endpoint exists: $ssm_endpoint"
fi

if [[ -z $ec2messages_endpoint || $ec2messages_endpoint == "None" ]]; then
    print_message "EC2Messages endpoint not found. Will create."
    endpoints_to_create+=("com.amazonaws.$aws_region.ec2messages")
else
    print_success "EC2Messages endpoint exists: $ec2messages_endpoint"
fi

if [[ -z $ssmmessages_endpoint || $ssmmessages_endpoint == "None" ]]; then
    print_message "SSMMessages endpoint not found. Will create."
    endpoints_to_create+=("com.amazonaws.$aws_region.ssmmessages")
else
    print_success "SSMMessages endpoint exists: $ssmmessages_endpoint"
fi

# Create missing endpoints
if [ ${#endpoints_to_create[@]} -gt 0 ]; then
    print_message "Creating missing VPC endpoints for SSM..."

    # Get the default security group
    security_group=$(aws ec2 describe-security-groups --filters "Name=vpc-id,Values=$vpc_id" "Name=group-name,Values=default" --query "SecurityGroups[0].GroupId" --output text)

    # Parse subnets to array
    subnet_ids=$(echo $subnets | jq -r '.[]')
    subnet_array=($subnet_ids)

    for endpoint in "${endpoints_to_create[@]}"; do
        print_message "Creating VPC endpoint for $endpoint..."
        aws ec2 create-vpc-endpoint --vpc-id $vpc_id --service-name $endpoint --subnet-ids ${subnet_array[@]} --vpc-endpoint-type Interface --security-group-ids $security_group

        if [ $? -ne 0 ]; then
            print_error "Failed to create VPC endpoint for $endpoint."
            print_message "Please create this endpoint manually:"
            print_message "1. Go to VPC console: https://console.aws.amazon.com/vpc/"
            print_message "2. Navigate to 'Endpoints'"
            print_message "3. Create endpoint for $endpoint"
            print_message "4. Select VPC $vpc_id and appropriate subnets"
            echo ""
            read -p "Press Enter to continue once you've created the endpoint..."
        else
            print_success "VPC endpoint for $endpoint created successfully."
        fi
    done
else
    print_success "All required VPC endpoints for SSM already exist."
fi

# Create IAM policy for users
print_message "Creating IAM policy for users to use Session Manager..."

policy_name="SSMSessionManagerAccess"
policy_document='{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ssm:StartSession",
                "ssm:TerminateSession",
                "ssm:ResumeSession",
                "ssm:DescribeSessions",
                "ssm:GetConnectionStatus",
                "ec2:DescribeInstances"
            ],
            "Resource": "*"
        }
    ]
}'

# Check if policy already exists
existing_policy=$(aws iam get-policy --policy-arn arn:aws:iam::$aws_account_id:policy/$policy_name 2>/dev/null || echo "not_found")

if [[ $existing_policy == "not_found" ]]; then
    print_message "Creating new IAM policy for Session Manager access..."
    aws iam create-policy --policy-name $policy_name --policy-document "$policy_document"

    if [ $? -ne 0 ]; then
        print_error "Failed to create IAM policy for Session Manager access."
        print_message "Please create this policy manually:"
        print_message "1. Go to IAM console: https://console.aws.amazon.com/iam/"
        print_message "2. Navigate to 'Policies' and create a new policy"
        print_message "3. Use the JSON policy editor and paste the following policy:"
        echo "$policy_document"
        echo ""
        read -p "Press Enter to continue once you've created the policy..."
    else
        print_success "IAM policy for Session Manager access created successfully."
        print_message "Policy ARN: arn:aws:iam::$aws_account_id:policy/$policy_name"
    fi
else
    print_success "IAM policy for Session Manager access already exists."
    print_message "Policy ARN: arn:aws:iam::$aws_account_id:policy/$policy_name"
fi

# Final instructions
echo ""
echo "==================================================="
echo "               Setup Complete!                     "
echo "==================================================="
echo ""
print_message "To allow users to access AWS Batch instances with Session Manager:"
print_message "1. Attach the 'SSMSessionManagerAccess' policy to their IAM user or group"
print_message "2. Ensure your AWS Batch instances are using the ecsTaskExecutionRole"
print_message "   or another role with the AmazonSSMManagedInstanceCore policy attached"
echo ""
print_message "Users can now connect to instances using Session Manager via:"
print_message "- AWS Management Console: Systems Manager > Session Manager > Start a session"
print_message "- AWS CLI: aws ssm start-session --target i-1234567890abcdef0"
echo ""
print_success "Setup process completed."
