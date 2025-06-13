# SkyPilot Launch Script

This script provides a convenient way to launch training jobs on AWS using SkyPilot.

## Prerequisites

- AWS credentials configured with `softmax` profile
- Access to AWS ECR in us-east-1 region
- SkyPilot CLI installed and configured
- Git repository with pushed commits (unless using `--skip-git-check`)

## Usage

```bash
./devops/skypilot/launch.py <COMMAND> run=<RUN_ID> [COMMAND_ARGS...] [OPTIONS]
```

You should include run=your_run_id in COMMAND_ARGS if your command logic requires it.

### Required Parameters

- `COMMAND`: The main command to execute (e.g., `train`, `eval`)
- `run=<RUN_ID>`: Unique identifier for the run (required parameter)

### Optional Parameters

- `COMMAND_ARGS`: Additional arguments to pass to the command
- `--git-ref <REF>`: Specify a git reference (branch, tag, or commit hash) instead of current HEAD
- `--gpus <N>`: Number of GPUs per node (overrides config)
- `--nodes <N>`: Number of nodes to use (overrides config)
- `--cpus <N>`: Number of CPUs per node (overrides config)
- `--no-spot`: Disable spot instances (use on-demand instead)
- `--copies <N>`: Launch N identical copies of the job (default: 1)
- `--timeout-hours <HOURS>`: Auto-terminate job after specified hours (supports decimals)
- `--skip-git-check`: Skip validation that commits are pushed
- `-c, --confirm`: Show detailed job summary and confirmation prompt before launching
- `--dry-run`: Show what would be executed without actually launching

## Examples

### Basic Usage

1. **Launch a training run with default parameters:**
   ```bash
   ./launch.py train run=my_experiment_001
   ```

2. **Launch with custom hyperparameters:**
   ```bash
   ./launch.py train run=my_experiment_002 trainer.learning_rate=0.001 trainer.batch_size=32
   ```

### Resource Configuration

3. **Use multiple GPUs:**
   ```bash
   ./launch.py train run=gpu_experiment --gpus 4
   ```

4. **Multi-node training:**
   ```bash
   ./launch.py train run=distributed_training --nodes 2 --gpus 8
   ```

5. **Use on-demand instances (more reliable but costlier):**
   ```bash
   ./launch.py train run=critical_experiment --no-spot
   ```

### Time Management

6. **Quick 30-minute experiment:**
   ```bash
   ./launch.py train run=quick_test --timeout-hours 0.5
   ```

7. **Long-running job with 8-hour limit:**
   ```bash
   ./launch.py train run=long_experiment --timeout-hours 8 --gpus 2
   ```

### Advanced Usage

8. **Launch multiple identical experiments:**
   ```bash
   ./launch.py train run=ablation_study --copies 5 --timeout-hours 2
   ```

9. **Use specific git commit:**
   ```bash
   ./launch.py train run=reproducible_exp --git-ref abc123def
   ```

10. **Preview configuration before launching:**
    ```bash
    ./launch.py train run=test_config --confirm
    ```

The `--confirm` flag displays a detailed job summary before launching:

```
============================================================
Job details:
============================================================
Name: my_experiment_001
GPUs: 1x A10G
CPUs: 8+
Spot Instances: Yes
Auto-termination: 2h
Git Reference: 56e04aa725000f186ec1bb2de84b359b4f273947
------------------------------------------------------------
Command: train
Task Arguments:
  1. trainer.curriculum=env/mettagrid/curriculum/navigation
  2. trainer.learning_rate=0.001
============================================================
Should we launch this task? (Y/n):
```

11. **Dry run with confirmation:**
    ```bash
    ./launch.py train run=test_config --dry-run --confirm
    ```

## Job Management

### Viewing Jobs

```bash
# List all jobs with status
sky jobs queue

# View job logs
sky jobs logs <JOB_ID>

# View controller logs (for debugging)
sky jobs logs <JOB_ID> --controller

# Stream logs in real-time
sky jobs logs <JOB_ID> --follow
```

### Canceling Jobs

```bash
# Cancel a specific job
sky jobs cancel <JOB_ID>

# Cancel all jobs
sky jobs cancel --all

# Cancel jobs by name pattern
sky jobs cancel -n "experiment_*"
```

### Job Status

Jobs can have the following statuses:
- `PENDING`: Waiting for resources
- `RUNNING`: Currently executing
- `SUCCEEDED`: Completed successfully
- `FAILED`: Terminated with error
- `CANCELLED`: Manually cancelled


## Sandboxes

Sandboxes provide persistent development environments for experimentation.

### Creating a Sandbox

```bash
# Create sandbox with main branch
./devops/skypilot/sandbox.py

# Create sandbox with specific commit/branch
./devops/skypilot/sandbox.py --git-ref feature/my-branch

# Force create new sandbox (even if one exists)
./devops/skypilot/sandbox.py --new
```

### Connecting to Sandbox

```bash
# SSH into sandbox (cluster name shown after creation)
ssh <cluster_name>

# Or use SkyPilot SSH
sky ssh <cluster_name>
```

### Managing Sandboxes

```bash
# List all clusters (including sandboxes)
sky status

# Stop sandbox (keeps data, saves costs)
sky stop <cluster_name>

# Restart stopped sandbox
sky start <cluster_name>

# Delete sandbox completely
sky down <cluster_name>
```

## Configuration

The script uses `./devops/skypilot/config/sk_train.yaml` as the base configuration. This file defines:
- Default resource requirements (CPU, GPU, memory)
- Docker image settings
- Environment variables
- Setup and run scripts



### Environment Variables

The following environment variables are automatically set:
- `METTA_RUN_ID`: The run identifier
- `METTA_CMD`: The command being executed
- `METTA_CMD_ARGS`: Additional command arguments
- `METTA_GIT_REF`: Git commit hash being used

### Debug Commands

```bash
# Check cluster status
sky status

# View detailed resource availability
sky show-gpus

# Check SkyPilot configuration
sky check

# View job details
sky jobs queue -a
```

## Best Practices

1. **Use descriptive run IDs**: Include date, user name, experiment type, and key parameters
   ```bash
   ./launch.py train run=2024_01_15_bert_lr_sweep_001
   ```

2. **Set appropriate timeouts**: Always use `--timeout-hours` to prevent runaway costs

3. **Use confirmation for important jobs**: Add `--confirm` to review the job details including GPU allocation, spot instance usage, and all command arguments

4. **Monitor actively**: Check job status regularly, especially for long-running jobs

5. **Clean up resources**: Cancel failed jobs and shut down unused sandboxes

## Additional Resources

- [SkyPilot Documentation](https://skypilot.readthedocs.io/)
- [AWS Instance Types](https://aws.amazon.com/ec2/instance-types/)
