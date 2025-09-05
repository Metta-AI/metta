# Live Sweep Monitor

A real-time, color-coded monitoring utility for Metta AI sweep runs with automatic refresh and rich terminal display.

## Features

- 🎨 **Color-coded status indicators**:
  - Completed runs: **Blue** (sorted to bottom)
  - In training: **Green**
  - Pending: **Gray**
  - Training done, no eval: **Orange**
  - In evaluation: **Cyan**
  - Failed: **Red**

- 📊 **Live metrics display**:
  - Progress in Gsteps (billions of steps): `0.189/2.5 Gsteps (7.6%)`
  - Cost tracking in USD format: `$4.31`
  - Run counts by status
  - Total sweep cost
  - Runtime tracking

- 🔄 **Auto-refresh**: Updates every 30 seconds (configurable)
- 📱 **In-place updates**: No scrolling output, clean terminal display
- 🧪 **Test mode**: Mock data for development and testing

## Usage

### Standalone Script

Run the live monitor directly from command line:

```bash
# Basic usage
./tools/live_sweep_monitor.py my_sweep_name

# Custom refresh interval (15 seconds)
./tools/live_sweep_monitor.py my_sweep_name --refresh 15

# Different WandB entity/project
./tools/live_sweep_monitor.py my_sweep_name --entity myteam --project myproject

# Test mode with mock data (no WandB required)
./tools/live_sweep_monitor.py test_sweep --test --refresh 5

# No screen clearing (append mode)
./tools/live_sweep_monitor.py my_sweep_name --no-clear
```

### Python API

Import and use in your own scripts:

```python
from metta.sweep.utils import live_monitor_sweep, live_monitor_sweep_test

# Monitor a real sweep
live_monitor_sweep(
    sweep_id="my_sweep_name",
    refresh_interval=30,
    entity="metta-research",
    project="metta"
)

# Test mode with mock data
live_monitor_sweep_test(
    sweep_id="test_sweep",
    refresh_interval=10,
    clear_screen=True
)
```

### Rich Table API

Generate static rich tables for embedding in other displays:

```python
from metta.sweep.utils import make_rich_monitor_table
from metta.sweep.stores.wandb import WandbStore

# Get runs data
store = WandbStore(entity="metta-research", project="metta")
runs = store.fetch_runs({"sweep_id": "my_sweep"})

# Create rich table
table = make_rich_monitor_table(runs)

# Display with rich console
from rich.console import Console
console = Console()
console.print(table)
```

## Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `sweep_id` | - | Required | Sweep ID to monitor |
| `--refresh` | `-r` | 30 | Refresh interval in seconds |
| `--entity` | `-e` | "metta-research" | WandB entity name |
| `--project` | `-p` | "metta" | WandB project name |
| `--test` | - | False | Run in test mode with mock data |
| `--no-clear` | - | False | Don't clear screen, append output |

## Display Format

The monitor shows a banner with sweep information followed by a color-coded table:

```
🔄 LIVE SWEEP MONITOR: my_sweep_name
⏱️  Runtime: 2:34:15
📊 Runs: 12 total | ✅ 4 completed | 🔄 2 training | ⏳ 3 pending | ❌ 1 failed
💰 Total Cost: $127.43
🔄 Last Update: 2024-01-15 14:30:22
────────────────────────────────────────────────────────────────────────────────

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━┓
┃ Run ID                ┃ Status                ┃ Progress                   ┃ Score       ┃ Cost  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━┩
│ trial_0001            │ IN_TRAINING           │ 1.89/2.5 Gsteps (75.6%)   │ N/A         │ $8.76 │
│ trial_0002            │ PENDING               │ -                          │ N/A         │ $0.00 │
│ trial_0003            │ COMPLETED             │ 2.5/2.5 Gsteps (100.0%)   │ 0.9234      │ $12.45│
└───────────────────────┴───────────────────────┴────────────────────────────┴─────────────┴───────┘
```

## Requirements

- Python packages: `rich`, `wandb`
- WandB authentication (run `wandb login`)
- Access to the specified WandB entity/project

## Examples

### Monitor a Running Sweep

```bash
# Start monitoring
./tools/live_sweep_monitor.py protein_optimization_v2

# Press Ctrl+C to stop
```

### Quick Testing During Development

```bash
# Test the display without WandB
./tools/live_sweep_monitor.py my_test --test --refresh 2
```

### Integration with Other Tools

```python
# In a Jupyter notebook or script
from metta.sweep.utils import make_rich_monitor_table
from metta.sweep.stores.wandb import WandbStore

store = WandbStore(entity="metta-research", project="metta")
runs = store.fetch_runs({"sweep_id": "my_sweep"})

# Display once
table = make_rich_monitor_table(runs)
print(table)

# Or start live monitoring
live_monitor_sweep("my_sweep", refresh_interval=60)
```

## Troubleshooting

### WandB Authentication Issues

```bash
# Login to WandB
wandb login

# Verify access
wandb status
```

### No Runs Found

- Check sweep ID spelling
- Verify entity/project parameters
- Ensure runs exist in WandB with the specified group

### Import Errors

```bash
# Install required packages
pip install rich wandb

# Or using uv
uv add rich wandb
```

### Performance Issues

- Increase refresh interval for large sweeps: `--refresh 60`
- Use `--no-clear` if screen clearing causes issues

## Development

The live monitor consists of several components:

- `live_monitor_sweep()`: Main monitoring function
- `make_rich_monitor_table()`: Rich table generation
- `live_monitor_sweep_test()`: Test mode with mock data
- `create_sweep_banner()`: Status banner generation

Test mode is useful for:
- Development without WandB access
- UI/UX testing
- CI/CD pipeline testing
- Documentation screenshots

## Integration

The monitor integrates with the existing Metta sweep infrastructure:

- Uses `WandbStore` for data fetching
- Leverages `RunInfo` models for standardized data
- Compatible with all sweep schedulers and optimizers
- Works with existing sweep controller outputs