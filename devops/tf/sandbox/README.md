# Researcher Sandbox Infrastructure

AWS infrastructure for ALife 2024 researcher sandboxes - isolated environment for external researchers to run training workloads.

## What This Creates

- Isolated VPC (`10.100.0.0/16`) separate from production
- IAM roles with minimal permissions for EC2 instances
- Pod Identity role for FastAPI orchestration service
- Security groups (SSH access only)

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
- Gets VPC/subnet/security group from Spacelift outputs
- Launches temp g5.12xlarge instance
- Installs Docker, NVIDIA drivers, Python 3.12, cogames CLI, mettagrid package
- Creates AMI
- Cleans up automatically


## Configuration

Set variables in Spacelift:

```hcl
region    = "us-east-1"
vpc_cidr  = "10.100.0.0/16"
```

## Key Security Features

Isolated VPC (no connection to production)
No S3 access (researchers use `cogames submit`)
Region-scoped to us-east-1
IMDSv2 enforced
Instances cannot launch instances or modify IAM
Pod Identity for FastAPI (no access keys)

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
