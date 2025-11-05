# Researcher Sandbox Terraform Module

This Terraform module creates the AWS infrastructure for the Alignment League researcher sandbox system.

## What it creates

- **Isolated VPC** (`10.100.0.0/16`) - Separate from main infrastructure for security
- **Public subnets** - For sandbox instances with public IPs (researchers SSH in)
- **Security group** - Allows SSH (port 22) inbound, all outbound
- **IAM user** (`sandbox-manager`) - For the FastAPI service to orchestrate EC2 instances
- **IAM role** (`sandbox-instance-role`) - Attached to EC2 instances for minimal permissions
- **IAM instance profile** - Links the role to EC2 instances

## Quick Start

### 1. Initialize and Apply Terraform

```bash
cd devops/tf/sandbox
terraform init
terraform plan
terraform apply
```

### 2. Build Base AMI

The AMI contains Ubuntu 22.04 + Docker + NVIDIA drivers + puffer + cogames.

**Option A: Automated (recommended)**

```bash
# Build AMI automatically
./scripts/build-ami.sh --ssh-key YOUR_AWS_SSH_KEY_NAME

# This will:
# 1. Launch a temporary g5.12xlarge instance
# 2. Install all software
# 3. Create AMI
# 4. Update terraform.tfvars with AMI ID
# 5. Clean up temporary instance
```

**Option B: Manual**

```bash
# 1. Launch temporary instance
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type g5.12xlarge \
  --key-name YOUR_KEY \
  --subnet-id $(terraform output -raw public_subnet_ids | jq -r '.[0]') \
  --security-group-ids $(terraform output -raw security_group_id)

# 2. SSH into instance
ssh ubuntu@<INSTANCE_IP>

# 3. Run setup script
curl -fsSL https://raw.githubusercontent.com/Metta-AI/metta/main/devops/tf/sandbox/scripts/setup-ami.sh | bash

# 4. Reboot and verify
sudo reboot
# (wait 60 seconds, then SSH back in)
nvidia-smi

# 5. Create AMI
aws ec2 create-image \
  --instance-id <INSTANCE_ID> \
  --name "researcher-sandbox-$(date +%Y%m%d)" \
  --description "Researcher sandbox with Docker, NVIDIA, puffer, cogames"

# 6. Update terraform.tfvars
echo 'sandbox_ami_id = "ami-xxxxx"' >> terraform.tfvars
terraform apply
```

### 3. Store Credentials in Kubernetes

```bash
# Create k8s secret for sandbox-manager service
kubectl create secret generic sandbox-manager-aws \
  --from-literal=AWS_ACCESS_KEY_ID=$(terraform output -raw sandbox_manager_access_key_id) \
  --from-literal=AWS_SECRET_ACCESS_KEY=$(terraform output -raw sandbox_manager_secret_access_key) \
  --namespace=default
```

## AMI Details

### What's Installed

The base AMI includes:

- **OS**: Ubuntu 22.04 LTS
- **Docker**: Latest version with NVIDIA Container Toolkit
- **NVIDIA Drivers**: Version 535 (supports A10G GPUs)
- **Python**: 3.12 with uv package manager
- **Repositories**: Metta/puffer pre-cloned with dependencies installed

### Building AMI

The `scripts/build-ami.sh` script automates AMI creation:

```bash
Usage: ./scripts/build-ami.sh [OPTIONS]

Options:
  --ssh-key KEY_NAME         SSH key name for accessing instance (required)
  --instance-type TYPE       Instance type (default: g5.12xlarge)
  --help                     Show help message
```

**What it does:**

1. Reads Terraform outputs (VPC, subnet, security group)
2. Launches temporary GPU instance with latest Ubuntu 22.04
3. Copies and runs `setup-ami.sh` to install software
4. Reboots instance to activate NVIDIA drivers
5. Verifies GPU access with `nvidia-smi`
6. Creates AMI from configured instance
7. Updates `terraform.tfvars` with new AMI ID
8. Terminates temporary instance

**Time:** ~20-25 minutes

### Updating AMI

When you need to update software versions:

```bash
# 1. Edit scripts/setup-ami.sh with new versions

# 2. Build new AMI
./scripts/build-ami.sh --ssh-key YOUR_KEY

# 3. Apply Terraform (uses new AMI ID from terraform.tfvars)
terraform apply
```

## Configuration

### Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `region` | AWS region | `us-east-1` |
| `vpc_cidr` | CIDR for sandbox VPC | `10.100.0.0/16` |
| `environment` | Environment name | `production` |
| `allowed_ssh_cidrs` | CIDRs allowed to SSH | `["0.0.0.0/0"]` |
| `default_instance_type` | Default EC2 type | `g5.12xlarge` |
| `sandbox_ami_id` | AMI for instances | `""` (required) |

### Customize via terraform.tfvars

```hcl
region      = "us-east-1"
environment = "production"
vpc_cidr    = "10.100.0.0/16"

# Set after building AMI
sandbox_ami_id = "ami-xxxxx"
```

## Outputs

Key outputs for integration with FastAPI service:

- `sandbox_config` - Complete configuration object
- `vpc_id` - VPC ID
- `public_subnet_ids` - List of subnet IDs
- `security_group_id` - Security group ID for instances
- `instance_profile_name` - IAM instance profile name
- `sandbox_manager_access_key_id` - IAM user access key (sensitive)
- `sandbox_manager_secret_access_key` - IAM user secret key (sensitive)

## Security Considerations

### Isolation

- **Separate VPC**: Sandbox VPC (`10.100.0.0/16`) is completely isolated from main EKS VPC (`10.0.0.0/16`)
- **No VPC peering**: No network connectivity between sandbox and production
- **Dedicated IAM roles**: Sandbox instances cannot access production resources

### IAM Permissions

**Sandbox manager service** can:

- Launch/stop/terminate EC2 instances in sandbox VPC
- Tag instances for cost tracking
- Read Cost Explorer data
- Pass `sandbox-instance-role` to EC2 instances

**Sandbox EC2 instances** can:

- Read from S3 buckets (`softmax-*`)
- Write to S3 outputs bucket (`softmax-sandbox-outputs`)
- Pull Docker images from ECR
- Write CloudWatch logs

**Sandbox instances CANNOT**:

- Launch other EC2 instances
- Modify IAM roles/policies
- Access other AWS services (RDS, EKS, etc.)

### Cost Controls

- All instances tagged with `user_id` for per-user cost tracking
- FastAPI service checks spending limits via Cost Explorer API before launching
- Security group restricts unnecessary network access

## Troubleshooting

### AMI Build Fails

**Issue**: `nvidia-smi` not working after reboot

```bash
# SSH into instance and check driver status
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

**Issue**: Docker can't access GPU

```bash
# Verify NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### Terraform Issues

**Error: VPC CIDR conflict**

If `10.100.0.0/16` conflicts with existing VPCs, change `vpc_cidr` in `terraform.tfvars`:

```hcl
vpc_cidr = "10.200.0.0/16"
```

**Error: IAM policy too large**

If IAM policies exceed size limits, split into multiple policies and attach separately.

## Next Steps

After applying this Terraform module and building the AMI:

1. Deploy FastAPI sandbox-manager service (see `services/sandbox-manager/`)
2. Configure FastAPI with Terraform outputs
3. Extend cogames CLI with sandbox commands
4. Test with pilot researchers

## Related Documentation

- Implementation plan: `.claude/tasks/sandbox-cli-plan.md`
- FastAPI service: `services/sandbox-manager/` (to be created)
- Asana task: "Allow researchers to run sandboxes on our account"
