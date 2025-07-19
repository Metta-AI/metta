# Curriculum Server Architecture

## Overview

The curriculum server enables centralized curriculum management for distributed training. It consists of:

1. **Curriculum Server**: HTTP server that wraps a curriculum object (runs on master node)
2. **Curriculum Client**: HTTP client that fetches tasks from the server (runs on all nodes)

During distributed training, multiple worker nodes need to sample tasks from the same curriculum. The curriculum server provides:

- Centralized task distribution
- Batch fetching to reduce network overhead
- Transparent curriculum interface for environments
- Centralized statistics collection on the master node

## Architecture

```
┌─────────────────┐
│   Master Node   │
│                 │
│  ┌───────────┐  │        HTTP GET /tasks
│  │Curriculum │  │      ┌──────────────────┐
│  │  Server   │◄─┼──────┤ Worker Node 1    │
│  │ (Port 12346)  │      │ ┌──────────────┐ │
│  └───────────┘  │      │ │  Curriculum  │ │
│        ▲        │      │ │   Client     │ │
│        │        │      │ └──────────────┘ │
│  ┌───────────┐  │      └──────────────────┘
│  │Curriculum │  │
│  │  Client   │  │      ┌──────────────────┐
│  └───────────┘  │      │ Worker Node 2    │
│                 │      │ ┌──────────────┐ │
└─────────────────┘      │ │  Curriculum  │ │
                         │ │   Client     │ │
                         │ └──────────────┘ │
                         └──────────────────┘
```

## Implementation Details

### Server Setup (Master Node)

The curriculum server is created in `tools/train.py` when distributed training is enabled:

```python
# In tools/train.py
if torch.distributed.is_initialized() and cfg.trainer.get("curriculum_server", {}).get("enabled", False):
    if is_master:
        # Master process runs the curriculum server
        curriculum_server = CurriculumServer(
            curriculum,
            host="0.0.0.0",
            port=curriculum_server_port
        )
        curriculum_server.start(background=True)

        # Create client for local use
        curriculum_client = CurriculumClient(
            server_url=f"http://localhost:{curriculum_server_port}",
            batch_size=batch_size
        )
        curriculum = curriculum_client  # Replace curriculum with client
```

### Client Setup (All Nodes)

Worker nodes connect to the master's server:

```python
# In tools/train.py
else:
    # Non-master ranks connect to master's server
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    curriculum_client = CurriculumClient(
        server_url=f"http://{master_addr}:{curriculum_server_port}",
        batch_size=batch_size
    )
    curriculum = curriculum_client  # Replace curriculum with client
```

The trainer then receives the curriculum (or client) as a parameter and uses it transparently:

```python
# Trainer instantiation in train.py
trainer = hydra.utils.instantiate(
    cfg.trainer,
    cfg,
    wandb_run=wandb_run,
    policy_store=policy_store,
    sim_suite_config=train_job.evals,
    stats_client=stats_client,
    curriculum=curriculum,  # Pass curriculum or client
)
```

### Configuration

Enable the curriculum server in your trainer config:

```yaml
# configs/trainer/curriculum_server_example.yaml
defaults:
  - trainer

curriculum_server:
  enabled: true
  port: 12346        # Server port (default: 12346)
  batch_size: 100   # Tasks per batch (default: 100)
```

### Client Behavior

1. **Batch Fetching**: Clients fetch tasks in batches to reduce network overhead
2. **Random Selection**: Tasks are randomly selected from the cached batch for variety
3. **Auto-refetch**: When the batch is exhausted, a new batch is automatically fetched
4. **Retry Logic**: Failed requests are retried with exponential backoff

### Statistics Collection

The trainer collects curriculum statistics in `_process_stats()`:

- Stats are only collected on the master node (from the server's curriculum)
- Stats are logged to wandb with appropriate prefixes:
  - `curriculum/{stat_name}` for custom stats
  - `curriculum/task_prob/{task_id}` for task probabilities
  - `task_completions/{task_id}` for completion rates

## Usage Example

```python
# Full example of distributed training with curriculum server
# This is handled automatically by train.py

# 1. Master node (rank 0):
# - Creates curriculum from config
# - Starts curriculum server
# - Creates local client
# - Passes client to trainer

# 2. Worker nodes (rank > 0):
# - Create client connecting to master
# - Pass client to trainer

# 3. All nodes:
# - Trainer uses curriculum/client transparently
# - Environments fetch tasks through the client
# - Stats are collected on master only
```

## Benefits

1. **Centralized State**: All curriculum logic remains on the master node
2. **Network Efficient**: Batch fetching reduces HTTP requests
3. **Transparent**: No changes needed to environments or trainer logic
4. **Consistent**: All workers sample from the same curriculum state

## Troubleshooting

1. **Connection Errors**: Ensure `MASTER_ADDR` environment variable is set correctly
2. **Port Conflicts**: Change the port in config if 12346 is already in use
3. **Slow Performance**: Increase `batch_size` to reduce network overhead
4. **Docker/AWS Networking**:
   - The server binds to `0.0.0.0` by default for Docker compatibility
   - Set `MASTER_ADDR` to the actual IP address or hostname of the master container
   - Ensure security groups allow traffic on the curriculum server port
   - For AWS ECS/EKS, use service discovery or internal DNS names
