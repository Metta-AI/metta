# Replay Mining Tools

Mine game replays into supervised learning datasets using Parquet + DuckDB.

## Quick Start

```python
from metta.tools.replay_dataset import ReplayDataset

# Load datasets from S3
dataset = ReplayDataset(start_date="2025-10-10", end_date="2025-10-17")
print(f"Loaded {len(dataset)} samples")

# Use with PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=256, shuffle=True)
```

## Mining Replays

This is a one-off/manual tool for creating datasets from existing replays. Run it locally or as needed to process replay data into training datasets.

### Basic Usage

```bash
# Process yesterday's replays (default)
uv run python -m metta.tools.replay_dataset.replay_mine

# Process specific date
uv run python -m metta.tools.replay_dataset.replay_mine --date 2025-10-15
```

### Date Range Processing

```bash
# Process explicit date range (sequential processing)
uv run python -m metta.tools.replay_dataset.replay_mine \
  --start-date 2025-10-10 \
  --end-date 2025-10-17

# Full backfill: earliest replay → yesterday (auto-discovers earliest date)
uv run python -m metta.tools.replay_dataset.replay_mine --backfill-all

# Full backfill up to specific date
uv run python -m metta.tools.replay_dataset.replay_mine \
  --backfill-all \
  --end-date 2025-10-20
```

### Output Options

```bash
# Save to production S3 (default)
uv run python -m metta.tools.replay_dataset.replay_mine --date 2025-10-15

# Save to local directory (for testing)
uv run python -m metta.tools.replay_dataset.replay_mine \
  --date 2025-10-15 \
  --output-prefix ./local_datasets

# Save to custom S3 bucket
uv run python -m metta.tools.replay_dataset.replay_mine \
  --date 2025-10-15 \
  --output-prefix s3://my-test-bucket/replay-datasets
```

### Advanced Options

```bash
# Filter by environment
uv run python -m metta.tools.replay_dataset.replay_mine \
  --date 2025-10-15 \
  --environment arena

# Minimum reward threshold
uv run python -m metta.tools.replay_dataset.replay_mine \
  --date 2025-10-15 \
  --min-reward 10.0

# Custom stats database
uv run python -m metta.tools.replay_dataset.replay_mine \
  --date 2025-10-15 \
  --stats-db-uri https://my-stats-server.com
```

## Loading with Filters

```python
# Filter by agent
dataset = ReplayDataset(
    start_date="2025-10-15",
    end_date="2025-10-17",
    filters={"agent_id": "= 5"}
)

# Multiple filters
dataset = ReplayDataset(
    start_date="2025-10-15",
    end_date="2025-10-17",
    filters={"agent_id": "IN (1, 2, 3)", "action": "> 0"}
)
```

## DuckDB Queries

```python
import duckdb

con = duckdb.connect()

# Query S3 directly without downloading
result = con.execute("""
    SELECT agent_id, COUNT(*) as samples
    FROM 's3://softmax-public/datasets/replays/replays_*.parquet'
    WHERE date BETWEEN '2025-10-10' AND '2025-10-17'
    GROUP BY agent_id
""").df()
```

## Dataset Format

Files: `s3://softmax-public/datasets/replays/replays_YYYYMMDD.parquet`

Columns:
- `observation` (str): JSON-serialized observation dict
- `action` (int): Action taken
- `agent_id` (int): Agent identifier
- `episode_id` (str): Episode identifier
- `timestep` (int): Timestep in episode
- `date` (str): Date in YYYY-MM-DD format

## Files

- `replay_mine.py`: Mining script - processes replays into datasets
- `replay_dataset.py`: PyTorch Dataset for loading Parquet files via DuckDB

## Features

- **Idempotent**: Running the same date multiple times produces the same output
- **Flexible Output**: Save to S3 (default) or local directories
- **Date Range Support**: Process multiple dates with `--start-date` and `--end-date`
- **Auto-Discovery**: `--backfill-all` queries database for earliest replay date
- **One file per day**: Format is `replays_YYYYMMDD.parquet`
- **Manual Execution**: Run locally or on-demand to process replay data

## Common Workflows

```bash
# Quick local test
uv run python -m metta.tools.replay_dataset.replay_mine \
  --date 2025-10-15 \
  --output-prefix ./test_datasets

# Production backfill (saves to S3)
uv run python -m metta.tools.replay_dataset.replay_mine --backfill-all

# Weekly update (last 7 days)
END_DATE=$(date -v-1d +%Y-%m-%d)  # Yesterday
START_DATE=$(date -v-7d +%Y-%m-%d)  # 7 days ago
uv run python -m metta.tools.replay_dataset.replay_mine \
  --start-date $START_DATE \
  --end-date $END_DATE
```
