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
    --run=b.$USER.run_name \
    env.game.objects.agent.energy_reward=1
```

## Stopping Jobs

1. Stop a running job:
```bash
python -m devops.aws.stop_jobs --job_prefix=b.$USER.run_name
```

## Monitoring Jobs

1. Monitor jobs through AWS Batch Console:
- URL: [AWS Batch Console](https://us-east-1.console.aws.amazon.com/batch/home?region=us-east-1#jobs)
- Queue: `metta-batch-jq-g6-8xlarge`

## Important Notes

- Jobs are processed on the `metta-batch-jq-g6-8xlarge` queue
- While jobs may take time to initialize, multiple jobs can run in parallel
- Use your username in the run name (e.g., `b.$USER.run_name`) to track your jobs

## Troubleshooting

If you encounter issues:
1. Verify your AWS SSO session is active (`aws s3 ls`)
2. Check the AWS Batch console for job status and logs
3. Ensure your run name follows the correct format (`b.$USER.your_run_name`)

