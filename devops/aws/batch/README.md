# AWS Batch Command Line Interface

A command-line interface for interacting with AWS Batch resources.

## Usage

There are two ways to use this command-line interface:

### Simplified Syntax
```
cmd.sh [command] [id] [options]
```

### Traditional Syntax
```
cmd.sh [resource_type] [id] [command] [options]
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
- `launch` or `st`: Launch a job

## Options

- `--queue` or `-q`: Job queue name (default: metta-jq for job commands)
- `--max` or `-m`: Maximum number of items to return (default: 5)
- `--tail` or `-t`: Tail logs
- `--attempt` or `-a`: Job attempt index
- `--node` or `-n`: Node index for multi-node jobs
- `--debug`: Enable debug output

## Features

- **Simplified Syntax**: Commands like `cmd.sh info <job_id>` are supported for common operations.
- **Default Commands**: When no command is specified, `list` is assumed.
- **Default Job Queue**: For job commands, if no queue is specified, `metta-jq` is used.
- **Job Name Support**: You can use either job IDs or job names for job commands.
- **Resource Auto-Detection**: When a job ID is not found, the system automatically checks if it's a compute environment or job queue.
- **Compute Environment Details**: The `compute` command displays Status, InstanceTypes, and Num Instances for compute environments.

## Examples

### Simplified Syntax Examples

Get information about a job:
```
cmd.sh info <job_id>
```

Get logs for a job:
```
cmd.sh logs <job_id>
```

Stop a job:
```
cmd.sh stop <job_id>
```

List jobs in the default queue (metta-jq):
```
cmd.sh jobs
```

List jobs in a specific queue:
```
cmd.sh jobs <queue_name>
```

List jobs with a higher limit:
```
cmd.sh jobs --max=10
```

List compute environments with status, instance types, and number of instances:
```
cmd.sh compute
```

Get detailed information about a specific compute environment, including its instances:
```
cmd.sh compute <compute_env_name>
```

### Traditional Syntax Examples

List all job queues:
```
cmd.sh job-queue list
```

Or simply:
```
cmd.sh job-queue
```

Get information about a specific job queue:
```
cmd.sh job-queue my-queue info
```

Get information about a specific job queue with more jobs:
```
cmd.sh job-queue my-queue info --max 10
```

List all compute environments:
```
cmd.sh ce list
```

Or simply:
```
cmd.sh ce
```

Get information about a specific compute environment (includes instance details):
```
cmd.sh ce my-compute-env info
```

Stop a compute environment:
```
cmd.sh ce my-compute-env stop
```

List jobs in a specific job queue:
```
cmd.sh job list --queue my-queue
```

Or simply:
```
cmd.sh job --queue my-queue
```

List jobs in the default queue (metta-jq):
```
cmd.sh job
```

List more jobs in a job queue:
```
cmd.sh job list --queue my-queue --max 20
```

Get information about a specific job (can use either job ID or job name):
```
cmd.sh job my-job-id info
```

Get logs for a job:
```
cmd.sh job my-job-id logs
```

Tail logs for a job:
```
cmd.sh job my-job-id logs --tail
```

Get logs for a specific attempt:
```
cmd.sh job my-job-id logs --attempt 1
```

Get logs for a specific node in a multi-node job:
```
cmd.sh job my-job-id logs --node 0
```

Stop a job:
```
cmd.sh job my-job-id stop
```


