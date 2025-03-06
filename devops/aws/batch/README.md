# AWS Batch Command Line Interface

A command-line interface for interacting with AWS Batch resources.

## Usage

There are two ways to use this command-line interface:

### Simplified Syntax
```
../cmd.sh [command] [id] [options]
```

### Traditional Syntax
```
../cmd.sh [resource_type] [id] [command] [options]
```

## Resource Types

- `job-queue` or `jq`: AWS Batch job queues
- `compute-environment` or `ce`: AWS Batch compute environments
- `job` or `j`: AWS Batch jobs
- `jobs`: List jobs in the default queue (metta-jq)
- `compute`: Alias for compute-environment

## Commands

- `list` or `l`: List resources (default if not specified)
- `info` or `d`: Get detailed information about a resource
- `logs` or `ls`: Get logs for a job
- `stop` or `s`: Stop a job or compute environment
- `ssh`: Connect to the instance running a job via SSH
- `launch` or `st`: Launch a job

## Options

- `--queue` or `-q`: Job queue name (default: metta-jq for job commands)
- `--max` or `-m`: Maximum number of items to return (default: 5)
- `--tail` or `-t`: Tail logs
- `--attempt` or `-a`: Job attempt index
- `--node` or `-n`: Node index for multi-node jobs
- `--instance` or `-i`: Connect directly to the instance without attempting to connect to the container (for ssh command)
- `--debug`: Enable debug output
- `--no-color`: Disable colored output

## Features

- **Simplified Syntax**: Commands like `../cmd.sh info <job_id>` are supported for common operations.
- **Default Commands**: When no command is specified, `list` is assumed.
- **Default Job Queue**: For job commands, if no queue is specified, `metta-jq` is used.
- **Job Name Support**: You can use either job IDs or job names for job commands.
- **Resource Auto-Detection**: When a job ID is not found, the system automatically checks if it's a compute environment or job queue.
- **Compute Environment Details**: The `compute` command displays Status, InstanceTypes, and Num Instances for compute environments.

## Examples

### Simplified Syntax Examples

Get information about a job:
```
../cmd.sh info <job_id>
```

Get logs for a job:
```
../cmd.sh logs <job_id>
```

Stop a job:
```
../cmd.sh stop <job_id>
```

List jobs in the default queue (metta-jq):
```
../cmd.sh jobs
```

List jobs in a specific queue:
```
../cmd.sh jobs <queue_name>
```

List jobs with a higher limit:
```
../cmd.sh jobs --max=10
```

List compute environments with status, instance types, and number of instances:
```
../cmd.sh compute
```

Get detailed information about a specific compute environment, including its instances:
```
../cmd.sh compute <compute_env_name>
```

### Traditional Syntax Examples

List all job queues:
```
../cmd.sh job-queue list
```

Or simply:
```
../cmd.sh job-queue
```

Get information about a specific job queue:
```
../cmd.sh job-queue my-queue info
```

Get information about a specific job queue with more jobs:
```
../cmd.sh job-queue my-queue info --max 10
```

List all compute environments:
```
../cmd.sh ce list
```

Or simply:
```
../cmd.sh ce
```

Get information about a specific compute environment (includes instance details):
```
../cmd.sh ce my-compute-env info
```

Stop a compute environment:
```
../cmd.sh ce my-compute-env stop
```

List jobs in a specific job queue:
```
../cmd.sh job list --queue my-queue
```

Or simply:
```
../cmd.sh job --queue my-queue
```

List jobs in the default queue (metta-jq):
```
../cmd.sh job
```

List more jobs in a job queue:
```
../cmd.sh job list --queue my-queue --max 20
```

Get information about a specific job (can use either job ID or job name):
```
../cmd.sh job my-job-id info
```

Get logs for a job:
```
../cmd.sh job my-job-id logs
```

Tail logs for a job:
```
../cmd.sh job my-job-id logs --tail
```

Get logs for a specific attempt:
```
../cmd.sh job my-job-id logs --attempt 1
```

Get logs for a specific node in a multi-node job:
```
../cmd.sh job my-job-id logs --node 0
```

Stop a job:
```
../cmd.sh job my-job-id stop
```

Connect to the instance running a job via SSH:
```
../cmd.sh ssh <job_id>
```

Connect directly to the instance without attempting to connect to the container:
```
../cmd.sh ssh <job_id> --instance
```

## Launch Command

Launch a new job:
```
../cmd.sh launch --run RUN_ID --cmd COMMAND [options]
```

### Launch Command Options

- `--run RUN`: The run id (required)
- `--cmd {train,sweep,evolve}`: The command to run (required)
- `--git-branch BRANCH`: The git branch to use (default: current commit)
- `--git-commit COMMIT`: The git commit to use (default: current commit)
- `--mettagrid-branch BRANCH`: The mettagrid branch to use (default: current commit)
- `--mettagrid-commit COMMIT`: The mettagrid commit to use (default: current commit)
- `--gpus GPUS`: Total number of GPUs to use (default: 4)
- `--node-gpus NODE_GPUS`: GPUs per node (default: 4)
- `--copies COPIES`: Number of job copies to submit (default: 1)
- `--job-queue QUEUE`: AWS Batch job queue to use (default: metta-jq)
- `--skip-push-check`: Skip checking if commits have been pushed to remote repositories
- `--no-color`: Disable colored output
- `--dry-run`: Dry run mode, prints job details without actually submitting the job

### Launch Command Examples

Launch a training job with 4 GPUs on a single node:
```
../cmd.sh launch --run my_run --cmd train
```

Launch a training job with 8 GPUs across 2 nodes:
```
../cmd.sh launch --run my_run --cmd train --gpus 8 --node-gpus 4
```

Launch a training job with a specific git branch:
```
../cmd.sh launch --run my_run --cmd train --git-branch my-branch
```

Launch a training job without checking if commits have been pushed:
```
../cmd.sh launch --run my_run --cmd train --skip-push-check
```

Preview a training job without actually submitting it:
```
../cmd.sh launch --run my_run --cmd train --dry-run
```


