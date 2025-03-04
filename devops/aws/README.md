# AWS Setup and Job Management Guide

## Initial Setup

1. Configure AWS SSO and credentials:
```bash
# Run the setup script
./devops/aws/setup_sso.sh

# Load new environment settings
source ~/.bashrc

# Verify access
aws s3 ls metta-ai
```

## Launching Jobs

1. Launch a training job using the Python launcher:
```bash
python -m devops.aws.launch_task \
    --cmd=train \
    --run=b.$USER.run_name
```

2. Specify a custom job queue:
```bash
python -m devops.aws.launch_task \
    --cmd=train \
    --run=b.$USER.run_name \
    --job_queue=metta-batch-jq-custom
```

3. Specify a custom mettagrid branch:
```bash
python -m devops.aws.launch_task \
    --cmd=train \
    --run=b.$USER.run_name \
    --mettagrid_branch=feature-branch
```

4. Use specific git and mettagrid commits:
```bash
python -m devops.aws.launch_task \
    --cmd=train \
    --run=b.$USER.run_name \
    --git_commit=abc123 \
    --mettagrid_commit=def456
```

## Stopping Jobs

1. Stop a running job:
```bash
python -m devops.aws.stop_jobs --job_prefix=b.$USER.run_name
```

## Monitoring Jobs

1. Monitor jobs through command line:
```bash
python -m devops.aws.cluster_info
```

2. Monitor jobs through AWS Batch Console:
- URL: [AWS Batch Console](https://us-east-1.console.aws.amazon.com/batch/home?region=us-east-1#jobs)
- Queue: `metta-batch-jq-g6-8xlarge`

## Important Notes

- Jobs are processed on the `metta-batch-jq-g6-8xlarge` queue by default
- You can specify a different queue using the `--job_queue` parameter
- You can specify a different mettagrid branch using the `--mettagrid_branch` parameter
- If no branch is specified, the current commit hash will be used by default
- While jobs may take time to initialize, multiple jobs can run in parallel
- Use your username in the run name (e.g., `b.$USER.run_name`) to track your jobs

## Troubleshooting

If you encounter issues:
1. Verify your AWS SSO session is active (`aws s3 ls`)
2. Check the AWS Batch console for job status and logs
3. Ensure your run name follows the correct format (`b.$USER.your_run_name`)

# AWS Batch Setup for Metta

This directory contains scripts for setting up and managing AWS Batch resources for Metta.

## Scripts

- `batch_setup.py`: Sets up a complete AWS Batch environment (compute environment, job queue, and job definition)
- `batch_register_job_definition.py`: Registers a job definition for AWS Batch

## Setting up AWS Batch Environment

The `batch_setup.py` script creates all the necessary AWS Batch resources:

1. Compute Environment: Configures the EC2 instances that will run your batch jobs
2. Job Queue: Creates a queue where jobs will be submitted
3. Job Definition: Defines the container, resources, and configuration for your jobs

### Prerequisites

- AWS CLI installed and configured with the `stem-root` profile
- Required IAM roles already created:
  - `AWSBatchServiceRole`: Service role for AWS Batch
  - `ecsInstanceRole`: Instance role for EC2 instances
  - `ecsTaskExecutionRole`: Task execution role for containers

### Basic Usage

```bash
# Set up a complete AWS Batch environment with default settings
python batch_setup.py

# Set up with custom names
python batch_setup.py --compute-env-name metta-ce-test --job-queue-name metta-jq-test --job-definition-name metta-jd-test
```

### Advanced Options

#### Compute Environment Options

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

#### Job Definition Options

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

## Submitting Jobs

After setting up the AWS Batch environment, you can submit jobs using the AWS CLI:

```bash
# Submit a simple job
aws batch submit-job \
  --job-name test-job \
  --job-queue metta-jq-prod \
  --job-definition metta-batch-dist-train \
  --profile stem-root
```

## Troubleshooting

If jobs get stuck in the RUNNABLE state, check:

1. Security group configurations - ensure EFS mount targets allow NFS traffic from the ECS instance security groups
2. EFS mount target availability in the same AZ as the EC2 instance
3. IAM role permissions - ensure the task execution role has necessary permissions
4. Container instance status - check if it's in ACTIVE state

To check job status:
```bash
aws batch describe-jobs --jobs YOUR_JOB_ID --profile stem-root
```

To check compute environment status:
```bash
aws batch describe-compute-environments --compute-environments YOUR_COMPUTE_ENV --profile stem-root
```

To check ECS container instances:
```bash
aws ecs list-container-instances --cluster YOUR_ECS_CLUSTER --profile stem-root
```

