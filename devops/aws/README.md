# AWS Setup and Job Management Guide

## Initial Setup

1. Configure AWS SSO and credentials:

```bash
# Run the setup script to configure AWS profiles
./devops/aws/setup_aws_profiles.sh

# Login to AWS SSO
aws sso login --profile softmax

# Load new environment settings
source ~/.bashrc

# Verify access
aws s3 ls metta-ai
```

## Job Management with cmd.sh

The `cmd.sh` script provides a convenient command-line interface for interacting with AWS Batch resources.

### Command Syntax

```bash
# Simplified syntax
cmd.sh [command] [id] [options]

# Traditional syntax
cmd.sh [resource_type] [id] [command] [options]
```

### Resource Types

- `job-queue` or `jq`: AWS Batch job queues
- `compute-environment` or `ce`: AWS Batch compute environments
- `job` or `j`: AWS Batch jobs
- `jobs`: List jobs in the default queue (metta-jq)
- `compute`: Alias for compute-environment

### Common Commands

- `list` or `l`: List resources (default if not specified)
- `info` or `d`: Get detailed information about a resource
- `logs` or `ls`: Get logs for a job
- `stop` or `s`: Stop a job or compute environment
- `ssh`: Connect to the instance running a job via SSH
- `launch` or `st`: Launch a job

### Monitoring Jobs

```bash
# Get information about a specific job
cmd.sh info <job_id>

# View logs for a specific job
cmd.sh logs <job_id>

# Tail logs for a job
cmd.sh logs <job_id> --tail

# List jobs in a queue (default: metta-jq, max: 10)
cmd.sh jobs [<job_queue>="metta-jq"] [--max=10]

# List compute environments (default: all)
cmd.sh compute [<compute_environment>="all"]

# Get detailed information about a compute environment
cmd.sh compute <compute_env_name>

# Connect to the instance running a job via SSH
cmd.sh ssh <job_id>
```

### Launching Jobs

```bash
# Launch a new job
cmd.sh launch --run RUN_ID --cmd COMMAND [options]

# Launch command options
# --run RUN             The run id (required)
# --cmd {train,sweep,evolve}  The command to run (required)
# --git-branch BRANCH   The git branch to use (default: current commit)
# --git-commit COMMIT   The git commit to use (default: current commit)
# --mettagrid-branch BRANCH  The mettagrid branch to use (default: current commit)
# --mettagrid-commit COMMIT  The mettagrid commit to use (default: current commit)
# --gpus GPUS           Total number of GPUs to use (default: 4)
# --node-gpus NODE_GPUS GPUs per node (default: 4)
# --copies COPIES       Number of job copies to submit (default: 1)
# --job-queue QUEUE     AWS Batch job queue to use (default: metta-jq)
# --skip-push-check     Skip checking if commits have been pushed to remote repositories
# --dry-run             Dry run mode, prints job details without actually submitting the job
```

#### Launch Examples

```bash
# Launch a training job with 4 GPUs on a single node
cmd.sh launch --run my_run --cmd train

# Launch a training job with 8 GPUs across 2 nodes
cmd.sh launch --run my_run --cmd train --gpus 8 --node-gpus 4

# Launch a training job with a specific git branch
cmd.sh launch --run my_run --cmd train --git-branch my-branch

# Launch a training job without checking if commits have been pushed
cmd.sh launch --run my_run --cmd train --skip-push-check

# Preview a training job without actually submitting it
cmd.sh launch --run my_run --cmd train --dry-run
```

You can also use the Python launcher:

```bash
# Launch a training job
python -m devops.aws.batch.launch_task \
    --cmd=train \
    --run=b.$USER.run_name

# Specify a custom job queue
python -m devops.aws.batch.launch_task \
    --cmd=train \
    --run=b.$USER.run_name \
    --job-queue=metta-batch-jq-custom

# Specify a custom mettagrid branch
python -m devops.aws.batch.launch_task \
    --cmd=train \
    --run=b.$USER.run_name \
    --mettagrid-branch=feature-branch

# Use specific git and mettagrid commits
python -m devops.aws.batch.launch_task \
    --cmd=train \
    --run=b.$USER.run_name \
    --git-commit=abc123 \
    --mettagrid-commit=def456
```

### Stopping Jobs

```bash
# Stop a specific job
cmd.sh stop <job_id>

# Stop a compute environment
cmd.sh ce <compute_env_name> stop
```

## Monitoring Jobs

1. Monitor jobs through command line:

```bash
# Using cmd.sh
cmd.sh jobs [<job_queue>="metta-jq"] [--max=10]

# Using Python module
python -m devops.aws.cluster_info
```

2. Monitor jobs through AWS Batch Console:
   - URL: [AWS Batch Console](https://us-east-1.console.aws.amazon.com/batch/home?region=us-east-1#jobs)
   - Queue: `metta-batch-jq-g6-8xlarge`

## Important Notes

- Jobs are processed on the `metta-jq` queue by default
- You can specify a different queue using the `--job-queue` parameter with both cmd.sh and the Python launcher
- You can specify a different mettagrid branch using the `--mettagrid-branch` parameter with both cmd.sh and the Python launcher
- If no branch is specified, the current commit hash will be used by default
- While jobs may take time to initialize, multiple jobs can run in parallel
- Use your username in the run name (e.g., `b.$USER.run_name`) to track your jobs

## Troubleshooting

If you encounter issues:

1. Verify your AWS SSO session is active (`aws s3 ls`)
2. Check the AWS Batch console for job status and logs
3. Ensure your run name follows the correct format (`b.$USER.your_run_name`)
4. Use `cmd.sh info <job_id>` to get detailed information about a job
5. Use `cmd.sh logs <job_id>` to view job logs

## AWS Batch Setup for Metta

This section covers setting up and managing AWS Batch resources for Metta.

### Scripts

- `batch_setup.py`: Sets up a complete AWS Batch environment (compute environment, job queue, and job definition)
- `batch_register_job_definition.py`: Registers a job definition for AWS Batch

### Setting up AWS Batch Environment

The `batch_setup.py` script creates all the necessary AWS Batch resources:

1. Compute Environment: Configures the EC2 instances that will run your batch jobs
2. Job Queue: Creates a queue where jobs will be submitted
3. Job Definition: Defines the container, resources, and configuration for your jobs

#### Prerequisites

- AWS CLI installed and configured with the `softmax-root` profile
- Required IAM roles already created:
  - `AWSBatchServiceRole`: Service role for AWS Batch
  - `ecsInstanceRole`: Instance role for EC2 instances
  - `ecsTaskExecutionRole`: Task execution role for containers

#### Basic Usage

```bash
# Set up a complete AWS Batch environment with default settings
python batch_setup.py

# Set up with custom names
python batch_setup.py --compute-env-name metta-ce-test --job-queue-name metta-jq-test --job-definition-name metta-jd-test
```

#### Advanced Options

##### Compute Environment Options

```bash
# Customize compute environment
python batch_setup.py \
  --compute-env-name metta-ce-custom \
  --min-vcpus 0 \
  --max-vcpus 512 \
  --instance-types g5.8xlarge g5.12xlarge \
  --service-role-name AWSBatchServiceRole \
  --instance-role-name ecsInstanceRole \
  --key-pair your-key-pair \
  --placement-group your-placement-group \
  --security-group-name your-security-group
```

##### Job Definition Options

```bash
# Customize job definition
python batch_setup.py \
  --job-definition-name metta-custom-job \
  --image mettaai/metta:latest \
  --job-role-name ecsTaskExecutionRole \
  --execution-role-name ecsTaskExecutionRole \
  --efs-name metta \
  --num-nodes 2 \
  --vcpus 32 \
  --memory 128000 \
  --gpus 4 \
  --shared-memory 230000
```

### Submitting Jobs

After setting up the AWS Batch environment, you can submit jobs using the AWS CLI:

```bash
# Submit a simple job
aws batch submit-job \
  --job-name test-job \
  --job-queue metta-jq-prod \
  --job-definition metta-batch-dist-train \
  --profile softmax-root
```

### Advanced Troubleshooting

If jobs get stuck in the RUNNABLE state, check:

1. Security group configurations - ensure EFS mount targets allow NFS traffic from the ECS instance security groups
2. EFS mount target availability in the same AZ as the EC2 instance
3. IAM role permissions - ensure the task execution role has necessary permissions
4. Container instance status - check if it's in ACTIVE state

```bash
# Check job status
aws batch describe-jobs --jobs YOUR_JOB_ID --profile softmax-root

# Check compute environment status
aws batch describe-compute-environments --compute-environments YOUR_COMPUTE_ENV --profile softmax-root

# Check ECS container instances
aws ecs list-container-instances --cluster YOUR_ECS_CLUSTER --profile softmax-root
```
