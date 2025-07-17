# Curriculum Server/Client System

The curriculum server/client system enables distributed training with centralized curriculum management. This ensures consistent task progression and statistics across all training workers.

## Overview

In distributed training:
- **Master rank** runs the curriculum server and uses the actual curriculum object (for stats logging)
- **Worker ranks** use curriculum clients that fetch tasks from the server
- All curriculum state and logic remains on the server
- Clients randomly select from batches of tasks provided by the server

## Architecture

```
Master Node (Rank 0)
├── Curriculum Server (port 5555)
├── Actual Curriculum Object
└── Trainer (logs curriculum stats)

Worker Nodes (Rank 1+)
├── Curriculum Client
└── Trainer (no stats)
```

## Configuration

Enable the curriculum server in your trainer config:

```yaml
trainer:
  curriculum_server:
    enabled: true
    port: 5555      # Server port
    batch_size: 100 # Tasks per batch
```

## How It Works

1. **Server Creation** (Master only):
   - Creates HTTP server wrapping the curriculum
   - Serves tasks via `/tasks` endpoint
   - Maintains all curriculum state

2. **Client Operation** (All workers):
   - Fetches batches of tasks from server
   - Randomly selects from the batch for each `get_task()` call
   - No-ops for stats methods (server handles all stats)

3. **Task Distribution**:
   - Server calls `curriculum.get_task()` for each task in the batch
   - Tasks are serialized as JSON with name and env_cfg
   - Clients recreate Task objects from the JSON data

## Benefits

- **Consistent Progression**: All workers see the same curriculum state
- **Centralized Stats**: Only the master logs curriculum statistics
- **Memory Efficiency**: Curriculum state only exists on master
- **Scalability**: Workers can be on different machines

## Implementation Details

The `CurriculumClient` implements the `Curriculum` interface but:
- Returns empty dicts for all stats methods
- `complete_task()` is a no-op (server handles completion)
- Randomly selects from cached task batches

This design ensures the trainer code remains unchanged while enabling distributed curriculum management.