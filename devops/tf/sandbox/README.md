# Researcher Sandbox Infrastructure

AWS infrastructure for ALife 2024 researcher sandboxes - isolated environment for external researchers to run training workloads.

## What This Creates

- Isolated VPC (`10.100.0.0/16`) separate from production
- S3 bucket (`softmax-sandbox-outputs`) for training outputs with lifecycle policies
- IAM roles with minimal permissions for EC2 instances
- Security groups (SSH access only)

**Note:** IRSA role for the FastAPI orchestration service is in `devops/tf/eks/sandbox-manager.tf`.

## Quick Start

### 1. Deploy via Spacelift

Create stack in [Spacelift UI](https://metta-ai.app.spacelift.io/):
- Name: `sandbox`
- Project root: `devops/tf/sandbox`
- Integration: `softmax-aws`

Review plan and apply.

### 2. Build AMI

```bash
cd devops/tf/sandbox
./scripts/build-ami.sh --ssh-key YOUR_AWS_KEY_NAME
```

This takes ~20 minutes and:
- Launches temp g5.12xlarge instance
- Installs Docker, NVIDIA drivers, Python 3.12, metta repo
- Creates AMI and updates `terraform.tfvars`
- Cleans up automatically

### 3. Apply Updated Config

```bash
git add terraform.tfvars
git commit -m "Update sandbox AMI"
# Spacelift applies automatically
```


## Configuration

Set variables in `terraform.tfvars`:

```hcl
region             = "us-east-1"
vpc_cidr           = "10.100.0.0/16"
sandbox_ami_id     = "ami-xxxxx"  # Set by build-ami.sh
```

## Outputs

Key outputs for FastAPI service:
- `vpc_id`, `public_subnet_ids`, `security_group_id`
- `instance_profile_name` - for EC2 instance launches
- IRSA role ARN from EKS stack: `sandbox_manager_irsa_role_arn`

## Troubleshooting

**GPU not working after AMI build:**
```bash
# SSH into instance
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

**Docker can't access GPU:**
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```


