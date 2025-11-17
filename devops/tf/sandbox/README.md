# Researcher Sandbox Infrastructure

AWS infrastructure for ALife 2024 researcher sandboxes - isolated environment for external researchers to run training
workloads.

## What This Creates

- Isolated VPC (`10.100.0.0/16`) separate from production
- IAM roles with minimal permissions for EC2 instances
- Pod Identity role for FastAPI orchestration service
- Security groups (SSH access only)

## Quick Start

### 1. Build AMI

**Note:** The AMI can be built independently of the Terraform infrastructure deployment.

```bash
cd devops/tf/sandbox
./scripts/build-ami.sh
```

This takes ~20 minutes and:

- Uses default VPC and creates temporary resources (security group, IAM role)
- Launches temp g5.12xlarge instance with SSM access (no SSH key needed!)
- Installs Docker, NVIDIA drivers, uv, cogames CLI, mettagrid package
- Creates AMI
- Cleans up automatically (instance, security group, IAM resources)

**Prerequisites:**

- AWS CLI configured with appropriate credentials
- Session Manager plugin: `brew install --cask session-manager-plugin` (macOS) or [install guide](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html)

### 2. Deploy Infrastructure via Spacelift

Create stack in [Spacelift UI](https://metta-ai.app.spacelift.io/):

- Name: `sandbox`
- Project root: `devops/tf/sandbox`
- Integration: `softmax-aws`

Review plan and apply.

## Configuration

Set variables in Spacelift:

```hcl
region    = "us-east-1"
vpc_cidr  = "10.100.0.0/16"
```

## Key Security Features

Isolated VPC (no connection to production) No S3 access (researchers use `cogames submit`) Region-scoped to us-east-1
IMDSv2 enforced Instances cannot launch instances or modify IAM Pod Identity for FastAPI (no access keys)

## Outputs

Key outputs for FastAPI service:

- `vpc_id`, `public_subnet_ids`, `security_group_id`
- `instance_profile_name` - for EC2 instance launches
- `sandbox_manager_role_arn` - for Pod Identity association

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
