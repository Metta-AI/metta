# Replay Mining Tools

Mine game replays into supervised learning datasets using Parquet + DuckDB.

## Quick Start

```python
from metta.tools.replay import ReplayDataset

# Load datasets from S3
dataset = ReplayDataset(start_date="2025-10-10", end_date="2025-10-17")
print(f"Loaded {len(dataset)} samples")

# Use with PyTorch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=256, shuffle=True)
```

## Mining Replays

```bash
# Process yesterday's replays (default)
uv run python -m metta.tools.replay.replay_mine

# Process specific date
uv run python -m metta.tools.replay.replay_mine --date 2025-10-15
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

## Deployment

For Kubernetes deployment, see [devops/charts/replay-mining-cronjob/](../../../devops/charts/replay-mining-cronjob/README.md).

Daily cronjob runs at 2 AM UTC, processing yesterday's replays automatically.
