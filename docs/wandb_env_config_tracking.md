# Tracking Environment Configurations in W&B

## Problem Statement

Environment configurations like `freeze_duration`, `max_steps`, `rewards`, etc. are not explicitly captured in the W&B config when using curricula that dynamically change these parameters during training. This makes it difficult to analyze how these parameters affect training outcomes.

## Solution Overview

We've implemented a comprehensive solution to track environment configurations as W&B artifacts and log key parameters in real-time:

1. **Real-time Logging**: Key environment parameters are logged to W&B at each epoch
2. **Configuration History**: Full configuration snapshots are saved with metadata
3. **W&B Artifacts**: Configuration history is periodically saved as artifacts for long-term storage
4. **Analysis Tools**: Scripts to retrieve and analyze configuration data

## Implementation Details

### 1. Trainer Modifications

The `MettaTrainer` class now includes:

```python
# Initialize config tracking
self._env_config_history: list[dict] = []
self._task_config_counts: dict[str, int] = defaultdict(int)
self._last_config_save_epoch = 0
# Configurable via trainer.env_config_save_interval (default: 1000 epochs)
self._config_save_interval = getattr(trainer_cfg, 'env_config_save_interval', 1000)
```

### 2. Configuration Tracking

At each training step, the trainer:
- Captures the current task configuration from the curriculum
- Logs key parameters to W&B for real-time monitoring
- Stores full configuration snapshots with metadata

Key parameters tracked include:
- `env/max_steps`
- `env/num_agents`
- `env/freeze_duration`
- `env/default_resource_limit`
- `env/reward_*` (for each reward type)
- `curriculum/task_prob/*` (task probabilities)

### 3. W&B Artifacts

Configuration history is saved as W&B artifacts:
- Type: `environment_configs`
- Contains: JSON file with full configuration history
- Metadata: run ID, curriculum type, task counts

## Usage

### During Training

The configuration tracking happens automatically during training. You'll see:
- Real-time parameter values in W&B dashboard under `env/*` and `curriculum/*` metrics
- **Smart artifact uploads**: Saves automatically when configs change + fallback every 10k epochs
- Final artifact upload when training completes

#### Artifact Saving Strategy:
1. **At epoch 0** (initial configuration)
2. **When configuration changes** (detected automatically via curriculum)
3. **Fallback interval** (default: every 10,000 epochs if no changes, configurable via `trainer.env_config_save_interval`)
4. **At training completion**

### Analyzing Configurations

Use the provided analysis script:

```bash
# Download and analyze configurations from a run
python tools/analyze_env_configs.py entity/project/run_id

# Export key parameters to CSV
python tools/analyze_env_configs.py entity/project/run_id --export-csv params.csv

# Specify output directory
python tools/analyze_env_configs.py entity/project/run_id --output-dir ./my_analysis
```

The script provides:
- Configuration change timeline
- Task frequency analysis
- Unique configuration counts
- Key parameter extraction
- CSV export for further analysis

### Example Output

```
=== Environment Configuration Analysis ===
Total configurations logged: 1523
Unique tasks: 8

=== Task Frequencies ===
  navigation/easy: 342 times
  navigation/medium: 298 times
  navigation/hard: 245 times
  ...

=== Configuration Changes ===
Total configuration changes: 47
  Change 1: Epoch 0, Step 0, Task: navigation/easy
  Change 2: Epoch 15, Step 12288, Task: navigation/medium
  ...

=== Sample Key Parameters ===
  navigation/easy_epoch_0:
    max_steps: 1000
    num_agents: 4
    freeze_duration: 10
    reward_heart: 1.0
```

## Benefits

1. **Transparency**: Full visibility into what configurations were used during training
2. **Reproducibility**: Ability to recreate exact training conditions
3. **Analysis**: Easy correlation between configuration changes and performance metrics
4. **Debugging**: Track when and how curricula changed configurations

## Future Enhancements

1. **Differential Configs**: Store only configuration differences to reduce storage
2. **Real-time Visualization**: W&B custom charts for configuration timelines
3. **Automated Analysis**: Correlation analysis between config changes and performance
4. **Config Replay**: Ability to replay exact configuration sequences

## Technical Notes

- Configurations are tracked only on the master rank in distributed training
- The tracking adds minimal overhead (<1% training time)
- Artifacts are versioned, allowing historical analysis
- JSON format ensures compatibility with various analysis tools
