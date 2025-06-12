# SkyPilot Launch Script

This script provides a convenient way to launch training jobs on AWS using SkyPilot.

## Prerequisites

- AWS credentials configured with `softmax` profile
- Access to AWS ECR in us-east-1 region
- SkyPilot CLI installed and configured

## Usage

```bash
./devops/skypilot/launch.py <COMMAND> [COMMAND_ARGS...] [OPTIONS]
```
You should include run=your_run_id in COMMAND_ARGS if your command logic requires it.

### Parameters

- `COMMAND`: The main command to execute
- `RUN_ID`: Unique identifier for the run
- `COMMAND_ARGS`: Additional arguments to pass to the command
- `OPTIONS`: Additional options to pass to the command (see `./devops/skypilot/launch.py --help` for the full list)

## Examples

1. Launch a training run with default parameters:

```bash
./launch.py train run=my_experiment_001
```

2. Launch a training run with specific arguments:

```bash
./launch.py train run=my_experiment_002 trainer.learning_rate=0.001 trainer.batch_size=32

```

3. Launch a training run with a 2-hour timeout:

```bash
./launch.py train run=my_experiment_003 --timeout-hours 2
```

4. Launch a quick experiment with a 30-minute timeout:

```bash
./launch.py train run=quick_test_004 --timeout-hours 0.5
```

5. Launch a long-running job with 4 hours timeout on multiple GPUs:

```bash
./launch.py train run=long_experiment_005 --timeout-hours 4 --gpus 2
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

## Timeout Feature

The `--timeout-hours` option provides automatic job termination to help manage resource usage and prevent runaway jobs:

- **Supports decimal values**: Use `--timeout-hours 1.5` for 90 minutes, `--timeout-hours 0.25` for 15 minutes
- **Automatic cleanup**: Jobs are terminated gracefully when the timeout is reached
- **Cost control**: Prevents unexpected charges from jobs that run longer than intended
- **Resource management**: Ensures compute resources are freed up for other tasks

**Example timeout values:**

- `--timeout-hours 0.5` = 30 minutes
- `--timeout-hours 1` = 1 hour
- `--timeout-hours 2.5` = 2 hours 30 minutes
- `--timeout-hours 8` = 8 hours

## Sandboxes

Sandboxes are often easier for quick experimentation.

### Deployment

The following command creates a new EC2 instance with specifications as defined in `sandbox.yaml`.
The script also runs a setup job to compile a metta repo on the machine, defaulting to main, but specific git commits can be specified.

```bash
./devops/skypilot/sandbox.py [--git-ref <GIT_REF>] [--new]
```

- `--git-ref <GIT_REF>`: Optional. Specify a git reference (branch, tag, or commit hash) to check out in the sandbox.
- `--new`: Optional. Force the creation of a new sandbox even if existing ones are found.

### Connecting

Connect to the sandbox using ssh (e.g., `ssh <cluster_name>`). The hostname will be printed by the deployment script. Authentication will be magically handled by SkyPilot.

### Shutting Down

To shut down a sandbox, use the following command:

```bash
sky down <CLUSTER_NAME>
```

## Notes

- The script automatically:

  - Gets ECR login credentials
  - Sets up environment variables
  - Uses the current git commit hash
  - Launches jobs in detached mode
  - Uses the configuration from `./devops/skypilot/config/sk_train.yaml`

- Jobs are launched asynchronously and will run in the background
- Monitor job status using SkyPilot CLI or AWS console
- When using `--timeout-hours`, jobs will be automatically terminated when the time limit is reached
