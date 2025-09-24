# SkyPilot Launch Script

This script provides a convenient way to launch training jobs on AWS using SkyPilot.

## Installation

- AWS credentials configured with `softmax` profile
- SkyPilot CLI installed and configured. This results in a ~/.sky/config.yaml

If you have successfully run `./devops/skypilot/install.sh` or `metta install`, these should be handled.

You can run this command to confirm your connectivity to the Softmax skypilot server, its health, and if you are
authenticated.

```bash
sky api info
```

## Usage

```bash
./devops/skypilot/launch.py <COMMAND> run=<RUN_ID> [COMMAND_ARGS...] [OPTIONS]
```

### Required Parameters

- `COMMAND`: The main command to execute (e.g., `train`, `eval`)
- `run=<RUN_ID>`: Unique identifier for the run (required parameter)

### Optional Parameters

For a complete list of optional parameters and their descriptions, use:

```bash
./devops/skypilot/launch.py --help
```

Note that launching jobs requires a repo with pushed commits (unless using `--skip-git-check`)

## Accessing the Dashboard

There's a [web dashboard](https://skypilot-api.softmax-research.net/) that displays the status of all clusters and jobs.

## Examples

### Basic Usage

1. **Launch a training run with default parameters:**

   ```bash
   devops/skypilot/launch.py train run=my_experiment_001
   ```

2. **Launch with custom hyperparameters:**
   ```bash
   devops/skypilot/launch.py train run=my_experiment_002 trainer.optimizer.learning_rate=0.001 trainer.rollout.batch_size=32
   ```

### Resource Configuration

3. **Use multiple GPUs:**

   ```bash
   devops/skypilot/launch.py train run=gpu_experiment --gpus 4
   ```

4. **Multi-node training:**

   ```bash
   devops/skypilot/launch.py train run=distributed_training --nodes 2 --gpus 8
   ```

5. **Use on-demand instances (more reliable but costlier):**
   ```bash
   devops/skypilot/launch.py train run=critical_experiment --no-spot
   ```

### Time Management

6. **Quick 30-minute experiment:**

   ```bash
   devops/skypilot/launch.py train run=quick_test --max-runtime-hours 0.5
   ```

7. **Long-running job with 8-hour limit:**
   ```bash
   devops/skypilot/launch.py train run=long_experiment ---max-runtime-hours 8 --gpus 2
   ```

### Advanced Usage

8. **Launch multiple identical experiments:**

   ```bash
   devops/skypilot/launch.py train run=ablation_study --copies 5 ---max-runtime-hours 2
   ```

9. **Use specific git commit:**

   ```bash
   devops/skypilot/launch.py train run=reproducible_exp --git-ref abc123def
   ```

10. **Preview configuration before launching:**
    ```bash
    devops/skypilot/launch.py train run=test_config --confirm
    ```

The `--confirm` flag displays a detailed job summary before launching:

```sh
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
1. training_env.curriculum=env/mettagrid/curriculum/navigation
2. trainer.optimizer.learning_rate=0.001
============================================================
Should we launch this task? (Y/n):
```

11. **Dry run:**
    ```bash
    devops/skypilot/launch.py train run=test_config --dry-run
    ```

The `--dry-run` flag allows you to preview the configuration that will be used before launching.

It will output the complete YAML configuration that would be used for the deployment, including:

- Resource specifications (cloud provider, instance types, GPUs)
- Docker configurations
- Environment variables
- File mounts
- Setup and run commands

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

## Shell Aliases

To streamline your workflow, we provide a script to set shell aliases for common SkyPilot operations. It also sets
`AWS_PROFILE=softmax` automatically.

To add these aliases temporarily:

```bash
source ./devops/skypilot/setup_shell.sh
```

To add them permanently, add the source command to your shell profile:

```bash
# For bash users:
echo "source /path/to/your/project/devops/skypilot/setup_shell.sh" >> ~/.bashrc

# For zsh users:
echo "source /path/to/your/project/devops/skypilot/setup_shell.sh" >> ~/.zshrc

# For fish users:
echo "source /path/to/your/project/devops/skypilot/setup_shell.sh" >> ~/.config/fish/config.fish
```

### Available Aliases

#### Job Queue Management

- `jj` - List active jobs (skips finished jobs)
- `jqa` - List all jobs including finished ones

#### Job Control

- `jk <JOB_ID>` - Cancel a job
- `jka` - Cancel all jobs

#### Logs

- `jl <JOB_ID>` - View job logs
- `jlc <JOB_ID>` - View controller logs (useful for debugging)
- `jll` - View logs for the most recent job
- `jllc` - View controller logs for the most recent job

#### Launching

- `lt run=<NAME>` - Quick launch training jobs
  ```bash
  lt run=my_experiment_001  # Equivalent to: ./devops/skypilot/launch.py train run=my_experiment_001
  ```

## Sandboxes

Sandboxes provide persistent GPU development environments for experimentation and debugging. Unlike training jobs that
terminate after completion, sandboxes remain running until you stop them.

### Quick Start

```bash
# Create sandbox with main branch
./devops/skypilot/sandbox.py

# Check if you have any existing sandboxes
./devops/skypilot/sandbox.py --check

# Force create new sandbox (even if one exists)
./devops/skypilot/sandbox.py --new

# Connect to your sandbox
ssh <sandbox-name>
```

### Creating a Sandbox

```bash
# Show existing sandboxes and management commands
./devops/skypilot/sandbox.py

# Launch a new sandbox with 1 GPU (default)
./devops/skypilot/sandbox.py --new

# Launch with multiple GPUs
./devops/skypilot/sandbox.py --new --gpus 4

# Launch with specific git branch
./devops/skypilot/sandbox.py --new --git-ref feature/my-branch

# Increase wait timeout for cluster initialization
./devops/skypilot/sandbox.py --new --wait-timeout 600
```

### Checking Sandbox Status

The `--check` mode provides a quick overview of your sandboxes without creating a new sandbox if none exist:

```bash
./devops/skypilot/sandbox.py --check
```

Example output:

```
Found 2 sandbox(es) for user alice:
  • alice-sandbox-1 (running) [L4:1]
  • alice-sandbox-2 (stopped) [L4:4]

Summary:
  1 running
  1 stopped

📦 Manage sandboxes:
  Launch new:     ./devops/skypilot/sandbox.py --new
  Connect:        ssh alice-sandbox-1
  Restart:        sky start alice-sandbox-2
  Stop:           sky stop alice-sandbox-1
  Delete:         sky down alice-sandbox-1
```

### Managing Sandboxes

```bash
# Connect to a running sandbox
ssh <sandbox-name>

# Stop sandbox (preserves data, saves costs)
sky stop <sandbox-name>

# Restart a stopped sandbox
sky start <sandbox-name>

# Delete sandbox completely
sky down <sandbox-name>

# Check logs if sandbox is stuck in INIT
sky logs <sandbox-name>

# Retry launch for stuck clusters
sky launch -c <sandbox-name> --no-setup
```

### Cost Management

- Sandboxes automatically stop after **48 hours** to prevent runaway costs
- L4 GPU instances cost approximately:
  - 1 GPU: ~$0.70-0.90/hour
  - 2 GPUs: ~$1.40-1.80/hour
  - 4 GPUs: ~$2.80-3.60/hour
  - 8 GPUs: ~$5.60-7.20/hour

To disable auto-stop:

```bash
sky autostop --cancel <sandbox-name>
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
   lt run=2024_01_15_bert_lr_sweep_001
   ```

2. **Set appropriate timeouts**: Always use `---max-runtime-hours` to prevent runaway costs

3. **Use confirmation for important jobs**: Add `--confirm` to review the job details including GPU allocation, spot
   instance usage, and all command arguments

4. **Monitor actively**: Check job status regularly, especially for long-running jobs

5. **Clean up resources**: Cancel failed jobs and shut down unused sandboxes

6. **Use sandboxes wisely**: Stop sandboxes when not in use to save costs, and delete old sandboxes you no longer need

## Additional Resources

- [SkyPilot Documentation](https://skypilot.readthedocs.io/)
- [AWS Instance Types](https://aws.amazon.com/ec2/instance-types/)
