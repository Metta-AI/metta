# Logging Guide

This guide explains how to use the logging helpers for distributed computing contexts like PyTorch DDP, SkyPilot, and
other multi-node/multi-process environments.

## Overview

The logging helpers automatically detect distributed environments and can display rank/node information in log messages.
This is useful for debugging distributed training, data processing pipelines, and multi-node deployments.

## Basic Usage

### Simplest Usage - The `log()` function

```python
from logging_helpers import log

# Just start logging - initialization happens automatically
log("Starting process")
log("Processing data", level=logging.DEBUG)
log("Training complete", master_only=True)  # Only rank 0 logs this
```

The `log()` function automatically initializes logging on first use with sensible defaults (showing rank in distributed
environments).

### Manual Initialization

For more control over logging configuration:

```python
from logging_helpers import init_logging
import logging

# Standard logging without rank display
init_logging(level="INFO")
logger = logging.getLogger(__name__)
logger.info("This is a regular log message")
# Output: [12:34:56.789] INFO     This is a regular log message
```

### Distributed Mode

```python
from logging_helpers import log, init_logging, log_master

# Using the simple log() function
log("Worker ready")  # Auto-shows rank in distributed environments
# Output: [12:34:56.789] [0] INFO     Worker ready
# Output: [12:34:56.790] [1] INFO     Worker ready

# Or with manual initialization
init_logging(level="INFO", show_rank=True)
logger = logging.getLogger(__name__)
logger.info("Worker ready")

# Master-only logging
log("Starting distributed job", master_only=True)
# Or
log_master("Starting distributed job")
# Output: [12:34:56.791] [0] INFO     Starting distributed job
```

## Supported Environments

The logging helpers automatically detect rank from these environment variables (defined in `constants.RANK_ENV_VARS`):

- `SKYPILOT_NODE_RANK` - SkyPilot clusters
- `RANK` - PyTorch Distributed Data Parallel (DDP)
- `OMPI_COMM_WORLD_RANK` - OpenMPI

## PyTorch Distributed Training

```python
import torch
import torch.distributed as dist
from logging_helpers import log, init_logging

def main():
    # Initialize PyTorch distributed
    dist.init_process_group(backend='nccl')

    # Option 1: Use the simple log() function
    log("Loading dataset", master_only=True)
    log(f"GPU {torch.cuda.current_device()} ready")

    # Option 2: Initialize with file logging
    init_logging(level="INFO", show_rank=True, run_dir="./runs/experiment1")

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(...)

        # Only master logs progress
        log(f"Epoch {epoch}: loss={train_loss:.4f}", master_only=True)

        # Master saves checkpoints
        if dist.get_rank() == 0:
            save_checkpoint(model, epoch)
            log(f"Saved checkpoint_{epoch}.pt")
```

## SkyPilot Multi-Node Jobs

```python
from logging_helpers import log, get_node_rank

# Simple logging with automatic rank detection
log("Starting worker process")
log("Head node: Setting up cluster", master_only=True)

# Get rank for conditional logic
rank = get_node_rank() or "0"
if rank == "0":
    # Head node specific tasks
    log("Downloading dataset to shared storage")
else:
    # Worker node tasks
    log(f"Worker {rank} waiting for data")
```

## Data Processing Pipelines

```python
from logging_helpers import log, get_node_rank
import os

def distributed_data_pipeline():
    rank = get_node_rank() or "0"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Master coordinates
    log(f"Starting pipeline with {world_size} workers", master_only=True)

    # Each worker processes a shard
    shard_id = int(rank)
    log(f"Processing shard {shard_id}/{world_size}")

    # Process data...
    records_processed = process_shard(shard_id)

    # Report results
    log(f"Completed: {records_processed} records")

    # Master aggregates results
    if rank == "0":
        total = gather_results()
        log(f"Pipeline complete: {total} total records")
```

## File Logging in Distributed Mode

```python
from logging_helpers import init_logging, log

# Each rank gets its own log file
init_logging(
    level="INFO",
    show_rank=True,
    run_dir="/shared/logs/experiment1"
)
# Creates:
# - /shared/logs/experiment1/logs/script.log (rank 0)
# - /shared/logs/experiment1/logs/script_1.log (rank 1)
# - /shared/logs/experiment1/logs/script_2.log (rank 2)
# etc.

# Then just use log() as normal
log("This goes to both console and file")
```

## API Reference

### Functions

#### `log(message, level=logging.INFO, show_rank=True, master_only=False)`

Universal logging function that handles initialization automatically.

- `message`: Message to log
- `level`: Log level (default: INFO)
- `show_rank`: Whether to show rank prefix (default: True)
- `master_only`: Whether to only log on rank 0 (default: False)

```python
from logging_helpers import log
import logging

log("Info message")  # Auto-initializes on first call
log("Debug info", level=logging.DEBUG)
log("Master checkpoint saved", master_only=True)
```

#### `init_logging(level=None, run_dir=None, show_rank=False)`

Initialize the logging system manually.

- `level`: Log level ("DEBUG", "INFO", "WARNING", "ERROR")
- `run_dir`: Directory for log files (creates `logs/` subdirectory)
- `show_rank`: Whether to show `[rank]` prefix in messages

#### `log_master(message, logger=None, level=logging.INFO)`

Log only on rank 0 ("master node").

- `message`: Message to log
- `logger`: Logger instance (default: root logger)
- `level`: Log level (default: INFO)

#### `get_node_rank() -> str | None`

Get current node/rank from environment variables. Returns None if not in distributed context.

### Best Practices

1. **Use `log()` for simplicity**

   ```python
   from logging_helpers import log

   # It just works - no setup needed
   log("Starting application")
   ```

2. **Use `master_only=True` for coordination messages**

   ```python
   log("Starting distributed training", master_only=True)
   log("All workers synchronized", master_only=True)
   log("Saving final results", master_only=True)
   ```

3. **Initialize explicitly for file logging or custom configuration**

   ```python
   init_logging(
       level="DEBUG",
       show_rank=True,
       run_dir="./experiments/run_001"
   )
   ```

4. **Include rank-specific information when relevant**

   ```python
   rank = get_node_rank() or "0"
   log(f"Worker {rank}: Processing batch {batch_id}")
   ```

5. **Use appropriate log levels**
   ```python
   log("Starting", level=logging.INFO)
   log("Warning: GPU memory low", level=logging.WARNING)
   log("Critical error", level=logging.ERROR)
   ```

## Environment Variables

### Setting Log Level

```bash
# Override log level via environment
export LOG_LEVEL=DEBUG
python train.py
```

### Distributed Environment Variables

```bash
# PyTorch DDP
torchrun --nproc_per_node=4 train.py

# SkyPilot (automatically set)
sky launch --num-nodes 4 task.yaml

# Manual testing
RANK=2 python worker.py
```

## Troubleshooting

### Logs not showing rank

- If using `log()`, rank is shown by default in distributed environments
- If using `init_logging()`, ensure `show_rank=True` is set
- Check that RANK, SKYPILOT_NODE_RANK, or OMPI_COMM_WORLD_RANK is set in environment

### Different timestamps on different nodes

- This is normal - each node has its own system clock
- Use log aggregation tools if precise ordering is needed

### File permission issues in shared storage

- Ensure run_dir is writable by all nodes
- Consider node-local logging if shared storage is slow
