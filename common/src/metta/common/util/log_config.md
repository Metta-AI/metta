# Logging Configuration

## Quick Start

```python
from metta.common.util.log_config import init_logging

if __name__ == "__main__":
    init_logging(run_dir=Path("./train_dir/run_001"))  # Optional; only if you want file logging
```

## Features

### Automatic Formatting

- **Rich formatting**: Interactive terminals get colored, formatted output
- **Simple formatting**: Batch jobs and wandb runs get plain text with timestamps
- **Millisecond precision**: All timestamps include milliseconds `[HH:MM:SS.mmm]`

### Rank-Aware Logging

In distributed environments, logs automatically include rank prefixes:

```
[14:32:15.123] [2] INFO     Starting training on node 2
```

All loggers also have master-only methods:

```python
logger = logging.getLogger(__name__)
logger.info_master("Only shows on rank 0")  # Also: debug_master, warning_master, etc.
```

### File Output

When `run_dir` is specified:

- Master (rank 0): `{run_dir}/logs/script.log`
- Workers: `{run_dir}/logs/script_{rank}.log`

## Environment Variables

| Variable             | Purpose                 | Example           |
| -------------------- | ----------------------- | ----------------- |
| `LOG_LEVEL`          | Set default level       | `LOG_LEVEL=DEBUG` |
| `RANK`, `LOCAL_RANK` | Add rank prefixes       | Set by torchrun   |
| `NO_RICH_LOGS`       | Force simple formatting | `NO_RICH_LOGS=1`  |

The following variables automatically trigger simple (non-Rich) formatting: `WANDB_MODE`, `AWS_BATCH_JOB_ID`,
`SKYPILOT_TASK_ID`, `NO_HYPERLINKS`

## Troubleshooting

**Logs not appearing?**

- Ensure `init_logging()` is called first
- Check log level: `export LOG_LEVEL=DEBUG`
- Verify logger.propagate is True (default)

**No rank prefix?**

- Rank environment variables must be set (usually by your distributed framework)
