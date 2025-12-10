# SkyDeck Quick Start Guide

## Installation

From the metta workspace root:

```bash
# Install (automatically done with workspace)
uv sync

# The 'skydeck' command is now available globally
skydeck --help
```

## Quick Commands

```bash
# Open dashboard in browser (starts server automatically)
skydeck

# Start server in background
skydeck start

# Stop server
skydeck stop

# Restart server
skydeck restart

# Check status
skydeck status

# View logs
skydeck logs
```

## Creating Your First Experiment

### Option 1: Via Web UI

1. Run `skydeck` to open the dashboard
2. Click "New Experiment"
3. Fill in the form:
   - **ID**: `test_ppo` (unique identifier)
   - **Name**: `Test PPO Training` (human-readable)
   - **Base Command**: `lt` (your training command)
   - **Run Name**: `daveey.test_ppo` (for run=...)
   - **Nodes**: 1
   - **GPUs**: 4
   - **Flags**:
     ```json
     {
       "trainer.losses.ppo.enabled": true,
       "policy_architecture.core_resnet_layers": 4
     }
     ```
4. Check "Start Immediately" if you want it to launch right away
5. Click "Create"

The table will show your experiment with all flags as columns. Click the row to expand and see details, job history, and controls.

### Option 2: Via Python Script

```python
import asyncio
from skydeck.database import Database
from skydeck.desired_state import DesiredStateManager
from skydeck.models import CreateExperimentRequest, DesiredState

async def main():
    db = Database(str(Path.home() / ".skydeck" / "skydeck.db"))
    await db.connect()

    dsm = DesiredStateManager(db)

    experiment = await dsm.create_experiment(CreateExperimentRequest(
        id="test_ppo",
        name="Test PPO Training",
        flags={
            "trainer.losses.ppo.enabled": True,
            "policy_architecture.core_resnet_layers": 4,
        },
        base_command="lt",
        run_name="daveey.test_ppo",
        nodes=1,
        gpus=4,
        desired_state=DesiredState.RUNNING,
    ))

    print(f"Created: {experiment.id}")
    await db.close()

asyncio.run(main())
```

## Table-Based UI

The dashboard uses a **table layout** where:
- **Each row** = one experiment
- **Each column** = one flag (dynamically added based on all experiments)
- **Click any row** to expand and see:
  - Full configuration details
  - Job history with status and exit codes
  - Start/Stop/Edit/Delete controls

### Features

- **Dynamic Columns**: Flags are automatically discovered across all experiments and shown as columns
- **Expandable Rows**: Click any row to see detailed information
- **Real-time Updates**: Table refreshes every 5 seconds
- **Status Badges**: Color-coded badges for desired/current state
- **Boolean Highlighting**: `true` values shown in green, `false` in red

## Common Operations

### Starting/Stopping Experiments

Click the "Start" or "Stop" button in the Actions column. The reconciler will:
- **Start**: Launch a new job on SkyPilot
- **Stop**: Cancel running jobs and stop the cluster

### Editing Flags

1. Click "Edit" button in the Actions column
2. Modify the JSON
3. Click "Save"

To apply changes, stop and restart the experiment.

### Viewing Job History

Click any row to expand it. The right side shows all past job runs with:
- Job ID
- Status (PENDING, RUNNING, SUCCEEDED, FAILED, etc.)
- Exit code
- Creation time

### Grid Search

See `examples/grid_search.py` for creating multiple experiments at once:

```bash
uv run python packages/skydeck/examples/grid_search.py
```

This creates a 2×4 grid over nodes × layers and starts them all!

## How It Works

### Architecture

```
User clicks "Start" in UI
  ↓
API call to update desired_state=RUNNING
  ↓
Saved to SQLite database
  ↓
Reconciler (60s loop) detects mismatch
  ↓
Calls sky.launch() with experiment flags
  ↓
Creates Job record in database
  ↓
Poller (30s loop) checks sky.queue()
  ↓
Updates Job status → Experiment current_state
  ↓
UI refreshes (5s) and shows updated status
```

### Data Persistence

Everything is stored in `~/.skydeck/skydeck.db`:
- Experiments and their configurations
- Complete job history
- Cluster status cache
- Flag schemas for typeahead

The dashboard survives restarts without losing any state!

## Configuration

The server runs on `http://127.0.0.1:8000` by default.

To customize:
```bash
skydeck start --host 0.0.0.0 --port 8080
```

Environment variables:
- `SKYDECK_DB_PATH`: Database location (default: `~/.skydeck/skydeck.db`)
- `SKYDECK_POLL_INTERVAL`: Seconds between SkyPilot polls (default: 30)
- `SKYDECK_RECONCILE_INTERVAL`: Seconds between reconciliation cycles (default: 60)

## Troubleshooting

### Dashboard won't start
```bash
# Check status
skydeck status

# View logs
skydeck logs

# Try different port
skydeck start --port 8080
```

### Experiments not launching
- Check logs: `skydeck logs`
- Verify SkyPilot is installed: `sky status`
- Check cloud credentials are configured

### Server won't stop
```bash
# Force stop
kill $(cat ~/.skydeck/skydeck.pid)
rm ~/.skydeck/skydeck.pid
```

## File Locations

- **PID file**: `~/.skydeck/skydeck.pid`
- **Logs**: `~/.skydeck/skydeck.log`
- **Database**: `~/.skydeck/skydeck.db`

## Next Steps

- Read the full [README.md](README.md) for architecture details
- Check out [examples/](examples/) for more use cases
- Explore the [API documentation](skydeck/app.py) for automation

Happy experimenting!
