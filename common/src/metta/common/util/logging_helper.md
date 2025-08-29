# Logging Guide

This guide explains how to use the logging helpers for distributed computing contexts like PyTorch DDP, SkyPilot, and
other multi-node/multi-process environments.

## Overview

The logging helpers automatically detect distributed environments and can display rank/node information in log messages.
This is useful for debugging distributed training, data processing pipelines, and multi-node deployments.

## Basic Usage

### Single-Node (Default)

```python
from logging_helpers import init_logging

# Standard logging without rank display
init_logging(level="INFO")
logger = logging.getLogger(__name__)
logger.info("This is a regular log message")
# Output: [12:34:56.789] INFO     This is a regular log message
```

### Distributed Mode

```python
from logging_helpers import init_logging, log_master

# Enable rank display for distributed contexts
init_logging(level="INFO", show_rank=True)
logger = logging.getLogger(__name__)

# All ranks log with their rank shown
logger.info("Worker ready")
# Output: [12:34:56.789] [0] INFO     Worker ready
# Output: [12:34:56.790] [1] INFO     Worker ready

# Only rank 0 logs this
log_master("Starting distributed job")
# Output: [12:34:56.791] [0] INFO     Starting distributed job
```

## Supported Environments

The logging helpers automatically detect rank from these environment variables:

- `SKYPILOT_NODE_RANK` - SkyPilot clusters
- `RANK` - PyTorch Distributed Data Parallel (DDP)
- `OMPI_COMM_WORLD_RANK` - OpenMPI

## PyTorch Distributed Training

```python
import torch
import torch.distributed as dist
from logging_helpers import init_logging, log_master

def main():
    # Initialize PyTorch distributed
    dist.init_process_group(backend='nccl')

    # Initialize logging with rank display
    init_logging(level="INFO", show_rank=True, run_dir="./runs/experiment1")
    logger = logging.getLogger("training")

    # Master-only messages
    log_master("Loading dataset", logger=logger)
    log_master("Hyperparameters: lr=0.001, batch_size=32", logger=logger)

    # All workers log
    logger.info(f"GPU {torch.cuda.current_device()} ready")

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(...)

        # Only master logs progress
        log_master(f"Epoch {epoch}: loss={train_loss:.4f}", logger=logger)

        # Master saves checkpoints
        if dist.get_rank() == 0:
            save_checkpoint(model, epoch)
            log_master(f"Saved checkpoint_{epoch}.pt", logger=logger)
```

## SkyPilot Multi-Node Jobs

```python
# Option 1: Use the SkyPilot wrapper
from skypilot_logging import log_master, log_all, log_error

# Automatically includes rank in output
log_master("Head node: Setting up cluster")
log_all("All nodes: Starting worker process")
log_error("Error on this node!")

# Option 2: Use logging_helpers directly
from logging_helpers import init_logging, log_master
import logging

init_logging(show_rank=True)
logger = logging.getLogger("skypilot_task")

log_master("Downloading dataset to shared storage")
logger.info("Processing local shard")
```

## Data Processing Pipelines

```python
from logging_helpers import init_logging, log_master, get_node_rank
import logging

def distributed_data_pipeline():
    init_logging(level="INFO", show_rank=True)
    logger = logging.getLogger("pipeline")

    rank = get_node_rank() or "0"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Master coordinates
    log_master(f"Starting pipeline with {world_size} workers", logger=logger)

    # Each worker processes a shard
    shard_id = int(rank)
    logger.info(f"Processing shard {shard_id}/{world_size}")

    # Process data...
    records_processed = process_shard(shard_id)

    # Report results
    logger.info(f"Completed: {records_processed} records")

    # Master aggregates results
    if rank == "0":
        total = gather_results()
        log_master(f"Pipeline complete: {total} total records", logger=logger)
```

## File Logging in Distributed Mode

```python
from logging_helpers import init_logging

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
```

## API Reference

### Functions

#### `init_logging(level=None, run_dir=None, show_rank=False)`

Initialize the logging system.

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

1. **Enable rank display only when needed**

   ```python
   # Conditional rank display
   is_distributed = get_node_rank() is not None
   init_logging(show_rank=is_distributed)
   ```

2. **Use log_master for coordination messages**

   ```python
   log_master("Starting distributed training")
   log_master("All workers synchronized")
   log_master("Saving final results")
   ```

3. **Create named loggers for different components**

   ```python
   data_logger = logging.getLogger("data")
   model_logger = logging.getLogger("model")
   metrics_logger = logging.getLogger("metrics")
   ```

4. **Include rank-specific information when relevant**

   ```python
   rank = get_node_rank() or "0"
   logger.info(f"Worker {rank}: Processing batch {batch_id}")
   ```

5. **Use appropriate log levels**
   ```python
   log_master("Starting", level=logging.INFO)
   log_master("Warning: GPU memory low", level=logging.WARNING)
   log_master("Critical error", level=logging.ERROR)
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

- Ensure `show_rank=True` is set in `init_logging()`
- Check that RANK or SKYPILOT_NODE_RANK is set in environment

### Different timestamps on different nodes

- This is normal - each node has its own system clock
- Use log aggregation tools if precise ordering is needed

### File permission issues in shared storage

- Ensure run_dir is writable by all nodes
- Consider node-local logging if shared storage is slow
