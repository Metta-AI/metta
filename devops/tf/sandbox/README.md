# Researcher Sandbox Terraform Module

This Terraform module creates the AWS infrastructure for the Alignment League researcher sandbox system.

## What it creates

- **Isolated VPC** (`10.100.0.0/16`) - Separate from main infrastructure for security
- **Public subnets** - For sandbox instances with public IPs (researchers SSH in)
- **Security group** - Allows SSH (port 22) inbound, all outbound
- **IAM role** (`sandbox-instance-role`) - Attached to EC2 instances for minimal permissions
- **IAM instance profile** - Links the role to EC2 instances

**Note:** The IRSA role for the sandbox-manager service is defined in `devops/tf/eks/sandbox-manager.tf` (part of the EKS stack).

## Quick Start

### 1. Create Spacelift Stack

This module is deployed via [Spacelift](https://spacelift.io/). See `../README.md` for general Spacelift workflow.

**Create stack in Spacelift UI:**
1. Go to https://metta-ai.app.spacelift.io/
2. Create new stack:
   - **Name**: `sandbox`
   - **Project root**: `devops/tf/sandbox`
   - **OpenTofu version**: Latest
   - **Integrations**: Attach `softmax-aws` cloud integration
3. (Optional) Enable **Local Preview** for faster iteration
4. (Optional) Enable **Autodeploy** to apply on merge to main

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

### 2. Deploy EKS IRSA Role

The sandbox-manager service uses IRSA (IAM Roles for Service Accounts) for AWS access.

**IRSA role is in `devops/tf/eks/sandbox-manager.tf`** and will be deployed as part of the EKS stack.

The FastAPI deployment will reference this role via service account annotation:
```yaml
serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::751442549699:role/sandbox-manager
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
- `instance_role_arn` - ARN of the instance IAM role

**Note:** The IRSA role ARN for the sandbox-manager service is output from the EKS stack (`sandbox_manager_irsa_role_arn`).

## Security Considerations

### Isolation

- **Separate VPC**: Sandbox VPC (`10.100.0.0/16`) is completely isolated from main EKS VPC (`10.0.0.0/16`)
- **No VPC peering**: No network connectivity between sandbox and production
- **Dedicated IAM roles**: Sandbox instances cannot access production resources

### IAM Permissions

**Sandbox manager service** (via IRSA role in `devops/tf/eks/sandbox-manager.tf`):

- Launch/stop/terminate EC2 instances
- Tag instances for cost tracking
- Read Cost Explorer data
- Pass `sandbox-instance-role` to EC2 instances
- Get instance profile information

**Sandbox EC2 instances** (via instance role):

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
