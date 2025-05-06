# SkyPilot Launch Script

This script provides a convenient way to launch training jobs on AWS using SkyPilot.

## Prerequisites

- AWS credentials configured with `softmax-db-admin` profile
- Access to AWS ECR in us-east-1 region
- SkyPilot CLI installed and configured

## Usage

```bash
./launch.sh <COMMAND> <RUN_ID> [COMMAND_ARGS...]
```

### Parameters

- `COMMAND`: The main command to execute
- `RUN_ID`: Unique identifier for the run
- `COMMAND_ARGS`: Additional arguments to pass to the command

## Examples

1. Launch a training run with default parameters:
```bash
./launch.sh train my_experiment_001
```

2. Launch a training run with specific arguments:
```bash
./launch.sh train my_experiment_002 trainer.learning_rate=0.001 trainer.batch_size=32
```

## Job Management

### Viewing Jobs

List all jobs:
```bash
sky jobs queue
```

View job logs:
```bash
sky jobs logs <JOB_ID>
sky jobs logs <JOB_ID> --controller
```

### Canceling Jobs

Cancel a specific job:
```bash
sky jobs cancel <JOB_ID>
```

Cancel all jobs:
```bash
sky jobs cancel --all
```

## Notes

- The script automatically:
  - Gets ECR login credentials
  - Sets up environment variables
  - Uses the current git commit hash
  - Launches jobs in detached mode
  - Uses the configuration from `./devops/skypilot/config/train.yaml`

- Jobs are launched asynchronously and will run in the background
- Monitor job status using SkyPilot CLI or AWS console
