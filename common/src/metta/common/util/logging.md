## Usage

### Simple usage: `log()`

```python
from metta.common.util.logging import log
log("Starting application")
# Use `master_only=True` for coordination messages
log("Starting distributed training", master_only=True)
log("All workers synchronized", master_only=True)
log("Saving final results", master_only=True)
```

### Manual usage: `init_logging` and define your own logger

#### `init_logging`

`log` will automatically call `init_logging` for you with sensible defaults. `init_logging` automatically adds rank info
if available from env vars, has default settings to reduce log spew, and chooses between rich/non-rich output depending
on the detected environment.

It does not by default also send output to files. You can do so by calling `init_logging` again with a `run_dir`

```python
init_logging(
    run_dir="./experiments/run_001"
)
# Files created at:
# -  ./experiments/run_001/script.log for master
# - ./experiments/run_001/script_{rank}.log for non-master ranks
```

You can call `init_logging` multiple times safely, though if you call it with different run_dirs, then each will be used
to create file separate output handlers.

#### Defining and using your own logger

```python
import logging
logger = logging.getLogger("my_service_name")
```

This will include `"my_service_name"` in messages and, so long as you have previously called `init_logging`, will still
reflect its features.

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

- Check that RANK, SKYPILOT_NODE_RANK, or OMPI_COMM_WORLD_RANK is set in environment

### Different timestamps on different nodes

- This is normal - each node has its own system clock
- Use log aggregation tools if precise ordering is needed

### File permission issues in shared storage

- Ensure run_dir is writable by all nodes
- Consider node-local logging if shared storage is slow
