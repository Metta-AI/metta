# Metta Tools Documentation

This document provides a practical guide to tools in the Metta ecosystem for training, evaluation, visualization, and development workflows.

> **Technical Details**: For in-depth information about the tool system architecture (tools, recipes, discovery), see [common/src/metta/common/tool/README.md](../common/src/metta/common/tool/README.md)

## Quick Reference

| Category          | Tool                         | Purpose                                       | GPU Required |
| ----------------- | ---------------------------- | --------------------------------------------- | ------------ |
| **Training**      | `run.py train <recipe>`      | Train policies with recipe configurations     | ✓            |
|                   | `sweep_init.py`              | Initialize hyperparameter sweep experiments   | ✗            |
|                   | `sweep_eval.py`              | Evaluate policies from sweep runs             | ✓            |
| **Evaluation**    | `run.py evaluate <recipe>`   | Run policy evaluation with recipe system      | ✓            |
|                   | `run.py analyze <recipe>`    | Analyze evaluation results with recipes       | ✗            |
| **Visualization** | `run.py play <recipe>`       | Interactive gameplay via recipe system        | ✗            |
|                   | `run.py replay <recipe>`     | Generate replay files via recipe system       | ✓            |
|                   | `renderer.py`                | Real-time ASCII/Miniscope rendering (legacy)  | ✓            |
|                   | `dashboard.py`               | Generate dashboard data for web visualization | ✗            |
| **Map Tools**     | `map/gen.py`                 | Generate maps from configuration files        | ✗            |
|                   | `map/view.py`                | View stored maps in various formats           | ✗            |
| **Utilities**     | `stats_duckdb_cli.py`        | Interactive DuckDB CLI for stats analysis     | ✗            |
|                   | `validate_config.py`         | Validate and print Hydra configurations       | ✗            |

## Quick Start

### Basic Usage

```bash
# Train a policy
./tools/run.py train arena run=my_experiment

# Evaluate the trained policy
./tools/run.py evaluate arena policy_uri=file://./train_dir/my_experiment/checkpoints

# Interactive play
./tools/run.py play arena policy_uri=file://./train_dir/my_experiment/checkpoints

# View replay
./tools/run.py replay arena policy_uri=file://./train_dir/my_experiment/checkpoints
```

### Syntax Shortcuts

The runner supports flexible invocation syntax:

```bash
# Two-token form: <tool> <recipe>
./tools/run.py train arena run=test

# Dot notation: <recipe>.<tool>
./tools/run.py arena.train run=test

# Full path
./tools/run.py experiments.recipes.arena.train run=test
```

### Discovering Available Tools

```bash
# List all tools in a recipe
./tools/run.py arena --list
./tools/run.py navigation --list

# List all recipes supporting a specific tool
./tools/run.py train --list
./tools/run.py evaluate --list
```

### Configuration Overrides

```bash
# Override nested configuration
./tools/run.py train arena \
  run=my_experiment \
  system.device=cpu \
  wandb.enabled=false \
  trainer.total_timesteps=1000000

# Show argument classification
./tools/run.py train arena run=test --verbose
```

## Core Workflows

### Training and Evaluation Pipeline

```bash
# 1. Train a policy
./tools/run.py train navigation run=nav_experiment_001

# 2. Evaluate the trained policy
./tools/run.py evaluate navigation policy_uri=s3://bucket/checkpoints/nav_experiment_001/nav_experiment_001:v8.pt

# 3. Analyze results
./tools/run.py analyze navigation eval_db_uri=./train_dir/eval_nav_experiment_001/stats.db

# 4. Interactive play with trained policy
./tools/run.py play navigation policy_uri=s3://bucket/checkpoints/nav_experiment_001/nav_experiment_001:v8.pt
```

### Hyperparameter Sweep Workflow

```bash
# 1. Initialize sweep
./tools/sweep_init.py sweep_name=hyperparam_search_001 \
  sweep_params=configs/sweep/navigation_sweep.yaml

# 2. Training happens automatically via sweep system

# 3. Evaluate sweep runs
./tools/sweep_eval.py run=<run_id> sweep_name=hyperparam_search_001

# 4. Interactive play with best policy
./tools/run.py play arena policy_uri="s3://bucket/checkpoints/sweeps/hyperparam_search_001/best:v42.pt"
```

### Map Development Workflow

```bash
# 1. Generate a new map
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py configs/env/mettagrid/maps/template.yaml \
  --output-uri=./my_map.yaml "seed=42"

# 2. View and iterate
./tools/map/view.py ./my_map.yaml

# 3. Normalize if needed
./tools/map/normalize_ascii_map.py ./my_map.yaml --in-place

# 4. Test with renderer
./tools/renderer.py env.game.map=@file://./my_map.yaml
```

## Training Tools

### run.py train

Train Metta policies using PPO with distributed training support, automatic hyperparameter configuration, and real-time metrics tracking.

**Key Features**:
- Distributed training across multiple GPUs/nodes
- Wandb integration for experiment tracking
- Checkpoint saving and policy versioning
- Configurable evaluation during training

**Usage**:
```bash
# Basic training
./tools/run.py train arena run=my_experiment

# With custom parameters
./tools/run.py train arena run=my_experiment \
  trainer.total_timesteps=1000000 \
  system.device=cpu \
  wandb.enabled=false
```

### sweep_init.py

Initialize hyperparameter sweep experiments using Wandb sweeps and Metta Protein optimization.

**Usage**:
```bash
# Initialize a new sweep
./tools/sweep_init.py sweep_name=lr_search sweep_params=configs/sweep/fast.yaml

# Distributed sweep (master node)
NODE_INDEX=0 ./tools/sweep_init.py sweep_name=distributed_exp
```

### sweep_eval.py

Evaluate policies generated during hyperparameter sweeps and update Protein observations.

**Usage**:
```bash
# Evaluate a sweep run
./tools/sweep_eval.py run=<run_id> sweep_name=<sweep_name>
```

## Evaluation Tools

### run.py evaluate

Run comprehensive policy evaluation with simulation suites and statistics export.

**Usage**:
```bash
# Evaluate a single policy
./tools/run.py evaluate navigation policy_uri=s3://bucket/checkpoints/experiment_001/experiment_001:v12.pt

# Evaluate with arena recipe
./tools/run.py evaluate arena policy_uri=file://./train_dir/my_run/checkpoints/my_run:v12.pt
```

**Key Features**:
- Multiple policy evaluation in one run
- Flexible policy selection (latest, top-k by metric)
- Replay generation for visualization
- Stats database export for analysis

### run.py analyze

Generate detailed analysis reports from evaluation results, including performance metrics and behavior analysis.

**Usage**:
```bash
# Analyze arena evaluation results
./tools/run.py analyze arena eval_db_uri=./train_dir/eval_experiment/stats.db

# Analyze navigation evaluation results
./tools/run.py analyze navigation eval_db_uri=./train_dir/eval_experiment/stats.db
```

## Visualization Tools

### run.py replay

Generate replay files for detailed post-hoc analysis in MettaScope.

**Usage**:
```bash
# Generate replay for a policy
./tools/run.py replay arena policy_uri=s3://bucket/checkpoints/abc123/abc123:v5.pt

# Generate replay from local checkpoint
./tools/run.py replay arena policy_uri=file://./train_dir/my_run/checkpoints/my_run:v12.pt
```

**Key Features**:
- Generates `.replay` files for MettaScope
- Automatic browser launch on macOS
- Local server for replay viewing

### run.py play

Interactive gameplay interface allowing humans to control Metta agents.

**Usage**:
```bash
# Start interactive session
./tools/run.py play arena

# Interactive play with specific policy
./tools/run.py play arena policy_uri=s3://bucket/checkpoints/my_experiment/my_experiment:v20.pt
```

**Key Features**:
- WebSocket-based real-time control
- Browser-based interface via MettaScope
- Human-in-the-loop testing

### renderer.py (Legacy)

Real-time visualization of agent behavior with ASCII or Miniscope rendering.

**Usage**:
```bash
# Visualize random policy
./tools/renderer.py renderer_job.policy_type=random

# Visualize trained policy
./tools/renderer.py renderer_job.policy_type=trained policy_uri="s3://bucket/checkpoints/experiment/experiment:v20.pt"
```

### dashboard.py

Generate dashboard data for web-based visualization of training runs and evaluations.

**Usage**:
```bash
# Generate dashboard and upload to S3
./tools/dashboard.py dashboard.output_path=s3://bucket/dashboards/experiment.json

# Generate local dashboard file
./tools/dashboard.py dashboard.output_path=./local_dashboard.json
```

## Map Tools

### map/gen.py

Generate MettaGrid maps from configuration files using various procedural algorithms (WFC, ConvChain, Random, Template-based).

**Usage**:
```bash
# Generate and display a single map
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py configs/env/mettagrid/maps/maze_9x9.yaml

# Save map to file
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py configs/env/mettagrid/maps/wfc_dungeon.yaml --output-uri=./dungeon.yaml

# Generate 100 maps to S3
./packages/mettagrid/python/src/mettagrid/mapgen/tools/gen.py configs/env/mettagrid/maps/random.yaml --output-uri=s3://bucket/maps/ --count=100
```

### map/view.py

View stored maps from various sources (local files, S3, etc.).

**Usage**:
```bash
# View a specific map
./packages/mettagrid/python/src/mettagrid/mapgen/tools/view.py ./my_map.yaml

# View random map from directory
./packages/mettagrid/python/src/mettagrid/mapgen/tools/view.py s3://bucket/maps/
```

### map/normalize_ascii_map.py

Normalize ASCII map characters to ensure consistency across different encodings.

**Usage**:
```bash
# Print normalized map
./tools/map/normalize_ascii_map.py map.txt

# Normalize in-place
./tools/map/normalize_ascii_map.py map.txt --in-place
```

## Utility Tools

### stats_duckdb_cli.py

Interactive DuckDB CLI for exploring evaluation statistics databases with automatic downloading from remote sources.

**Usage**:
```bash
# Open stats from Wandb
./tools/stats_duckdb_cli.py +eval_db_uri=wandb://stats/navigation_eval_v2

# Open stats from S3
./tools/stats_duckdb_cli.py +eval_db_uri=s3://bucket/evaluations/experiment_001.db
```

**Example Queries**:
```sql
-- Get average rewards by policy
SELECT policy_name, AVG(value) as avg_reward
FROM agent_metrics
WHERE metric = 'reward'
GROUP BY policy_name;
```

### validate_config.py

Load and validate Hydra configuration files for debugging config issues.

**Usage**:
```bash
# Validate environment configuration
./tools/validate_config.py configs/env/mettagrid/navigation.yaml

# Validate trainer configuration
./tools/validate_config.py trainer/trainer.yaml
```

### autotune.py

Auto-tune vectorization parameters for optimal performance using pufferlib.

**Usage**:
```bash
# Run autotuning
./tools/autotune.py
```

**Key Features**:
- Finds optimal batch size
- Determines max environments
- Memory usage optimization

## Environment Variables

Key environment variables used by tools:

- `RANK`: Distributed training rank
- `LOCAL_RANK`: Local GPU rank
- `NODE_INDEX`: Node index in multi-node setup
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `WANDB_API_KEY`: Wandb authentication
- `AWS_PROFILE`: AWS credentials for S3 access

## Common Issues and Solutions

### GPU Memory Issues

```bash
# Use CPU for testing
./tools/run.py train arena run=cpu_test system.device=cpu

# Reduce training time for quick testing
./tools/run.py train arena run=quick_test trainer.total_timesteps=10000
```

### Local Testing Without External Services

```bash
# Disable Wandb and use CPU
./tools/run.py train arena run=local_test wandb.enabled=false system.device=cpu
```

## Best Practices

1. **Always validate configs** before long-running experiments
2. **Use meaningful run names** for easy identification (e.g., `local.alice.experiment_001`)
3. **Export evaluation data** to S3 for persistence
4. **Monitor GPU memory** usage during training
5. **Use sweep tools** for systematic hyperparameter search
6. **Generate replays** for debugging unexpected behaviors
7. **Document custom configurations** in version control

## CLI Cheat Sheet

| Task                    | Command                                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------------------- |
| Train (arena)           | `./tools/run.py train arena run=my_experiment`                                                                 |
| Train (navigation)      | `./tools/run.py train navigation run=my_experiment`                                                            |
| Play (browser)          | `./tools/run.py play arena`                                                                                    |
| Replay (policy)         | `./tools/run.py replay arena policy_uri=s3://bucket/checkpoints/local.alice.1/local.alice.1:v10.pt`           |
| Evaluate (arena)        | `./tools/run.py evaluate arena policy_uris=s3://bucket/checkpoints/local.alice.1/local.alice.1:v10.pt`        |
| Evaluate (navigation)   | `./tools/run.py evaluate navigation policy_uris=s3://bucket/checkpoints/local.alice.1/local.alice.1:v10.pt`   |

Running these commands mirrors our CI configuration and helps keep the codebase consistent.
