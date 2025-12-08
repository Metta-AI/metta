# SkyDeck Dashboard

Web-based dashboard and controller for managing SkyPilot experiments with declarative desired state management, automatic reconciliation, and persistent job history tracking.

## Features

- **Declarative State Management**: Set desired state (RUNNING/STOPPED/TERMINATED), system automatically reconciles
- **Job History Tracking**: Complete history of all job executions per experiment
- **Dynamic Flag Configuration**: Type-safe configuration using flags with autocomplete
- **Automatic Reconciliation**: Background reconciler ensures current state matches desired state
- **SQLite Persistence**: All state survives restarts
- **Real-time Updates**: Web UI polls and displays current status

## Quick Start

```bash
# Install dependencies
cd packages/skydeck
uv pip install -e .

# Run the dashboard
uv run python -m skydeck.run

# Access at http://localhost:8000
```

## Architecture

### Core Components

1. **Data Models** (`models.py`): Job and Experiment Pydantic models
2. **Database Layer** (`database.py`): SQLite async operations
3. **Flag Schema** (`flag_schema.py`): Dynamic flag inference from Pydantic models
4. **State Manager** (`state_manager.py`): Cache of current SkyPilot state
5. **Desired State Manager** (`desired_state.py`): CRUD for experiments
6. **Background Poller** (`poller.py`): Polls SkyPilot every 30s
7. **Reconciler** (`reconciler.py`): Brings current state → desired state
8. **FastAPI Backend** (`app.py`): REST API and static file serving
9. **Web UI** (`static/`): Single-page dashboard

### Data Model

**Experiment**: Configuration template that spawns jobs
- Has desired_state (what you want) and current_state (actual status)
- Configuration stored as flags: `Dict[str, Union[str, int, float, bool]]`
- Only one running job per experiment at a time

**Job**: Single SkyPilot job execution
- Linked to parent experiment
- Full execution history: timestamps, status, logs, exit code
- Status: INIT, PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED

### Reconciliation Loop

```
User sets desired_state=RUNNING
  ↓
Reconciler detects mismatch
  ↓
Calls sky.launch() with experiment flags
  ↓
Creates new Job record
  ↓
Poller updates job status from SkyPilot
  ↓
Experiment current_state updated
```

## Usage Examples

### Create and Run Experiment

```python
experiment = Experiment(
    id="ppo_4layer",
    name="PPO 4 Layers",
    flags={
        "trainer.losses.ppo.enabled": True,
        "policy_architecture.core_resnet_layers": 4,
    },
    base_command="lt",
    run_name="daveey.ppo_4layer",
    nodes=4,
    gpus=4,
    desired_state=DesiredState.RUNNING,
)
await db.save_experiment(experiment)
```

### Grid Search

```python
for layers in [1, 4, 16, 64]:
    experiment = Experiment(
        id=f"ppo_{layers}layer",
        name=f"PPO {layers} Layers",
        flags={
            "trainer.losses.ppo.enabled": True,
            "policy_architecture.core_resnet_layers": layers,
        },
        nodes=4,
        gpus=4,
        desired_state=DesiredState.RUNNING,
    )
    await db.save_experiment(experiment)
# Reconciler automatically launches all experiments!
```

### View Job History

```python
jobs = await db.get_jobs_for_experiment("ppo_4layer", limit=10)
for job in jobs:
    print(f"{job.id}: {job.status} (exit={job.exit_code})")
```

## API Endpoints

### Experiments
- `GET /api/experiments` - List all experiments
- `POST /api/experiments` - Create new experiment
- `GET /api/experiments/{id}` - Get experiment details
- `DELETE /api/experiments/{id}` - Delete experiment
- `POST /api/experiments/{id}/state` - Update desired state
- `GET /api/experiments/{id}/status` - Full status with current job
- `GET /api/experiments/{id}/jobs` - Job history
- `POST /api/experiments/{id}/flags` - Update flags

### Jobs
- `POST /api/jobs/{id}/cancel` - Cancel running job

### System
- `GET /api/health` - System health status
- `GET /api/flag-schemas` - Flag metadata for autocomplete
- `POST /api/refresh` - Force SkyPilot state refresh
- `POST /api/reconcile` - Force reconciliation

## Configuration

Environment variables:
- `SKYDECK_DB_PATH`: Database file path (default: `skydeck.db`)
- `SKYDECK_POLL_INTERVAL`: Poll interval in seconds (default: 30)
- `SKYDECK_RECONCILE_INTERVAL`: Reconcile interval in seconds (default: 60)

Command line options:
```bash
python -m skydeck.run --host 0.0.0.0 --port 8000 --db-path /path/to/db.sqlite
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run with auto-reload
uv run uvicorn skydeck.app:app --reload
```

## Key Design Principles

1. **Experiments are templates, jobs are instances** - Like classes vs objects
2. **Flags are first-class** - Dynamically typed, inferred from Pydantic
3. **SQLite for everything** - Survives restarts, single dependency
4. **Automatic reconciliation** - Set desired state, system does the rest
5. **Job history tracking** - Never lose experiment outcomes
6. **One running job per experiment** - Enforced by reconciler

## License

MIT
