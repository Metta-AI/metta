# Metta Testing Commands

This file contains simple commands for testing the metta codebase. These commands are designed to run quickly and
provide useful feedback about whether the system is working correctly.

## Prerequisites

The project uses `uv` for Python package management. All Python commands should be run with `uv run` to ensure proper
environment activation.

## Quick Test Commands (30-60 seconds total)

### Set Test ID First

```bash
# Set a unique test ID for this testing session
export TEST_ID=$(date +%Y%m%d_%H%M%S)
echo "Test ID: $TEST_ID"
```

### 1. Train for 30 seconds

```bash
# Basic training with arena recipe (will run indefinitely, terminate with Ctrl+C after ~30 seconds)
uv run ./tools/run.py train arena run=test_$TEST_ID

# Training with navigation recipe
uv run ./tools/run.py train navigation run=test_$TEST_ID

# Limited training for testing (you can also terminate with Ctrl+C)
uv run ./tools/run.py train arena run=cursor_$TEST_ID trainer.total_timesteps=100000
```

### 2. Run simulations on trained model

```bash
# Run evaluations on the checkpoint from step 1
uv run ./tools/run.py eval arena policy_uri=file://./train_dir/test_$TEST_ID/checkpoints

# Run navigation evaluations
uv run ./tools/run.py eval navigation policy_uri=file://./train_dir/test_$TEST_ID/checkpoints

# Using wandb artifact
uv run ./tools/run.py eval arena policy_uri=wandb://run/test_$TEST_ID
```

### 3. Analyze results

```bash
# Analyze the simulation results from step 2
uv run ./tools/run.py analyze arena eval_db_uri=./train_dir/eval_$TEST_ID/stats.db

# Analyze navigation results
uv run ./tools/run.py analyze navigation eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

## One-Line Test Commands

### Basic 30-second test (copy-paste friendly)

```bash
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID" && uv run ./tools/run.py train arena run=test_$TEST_ID trainer.total_timesteps=10000
# After training completes or you Ctrl+C:
uv run ./tools/run.py eval arena policy_uri=file://./train_dir/test_$TEST_ID/checkpoints
uv run ./tools/run.py analyze arena eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

### Quick arena test (auto-limited training)

```bash
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID"
uv run ./tools/run.py train arena run=cursor_$TEST_ID trainer.total_timesteps=100000
uv run ./tools/run.py eval arena policy_uri=file://./train_dir/cursor_$TEST_ID/checkpoints
uv run ./tools/run.py analyze arena eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

## Full Integration Test (2-3 minutes)

```bash
# Set test ID
export TEST_ID=$(date +%Y%m%d_%H%M%S)
echo "Running full integration test with ID: $TEST_ID"

# 1. Train for 100k steps (~1 minute on GPU)
uv run ./tools/run.py train arena run=test_$TEST_ID trainer.total_timesteps=100000

# 2. Run evaluations (~30 seconds)
uv run ./tools/run.py eval arena policy_uri=file://./train_dir/test_$TEST_ID/checkpoints

# 3. Analyze results
uv run ./tools/run.py analyze arena eval_db_uri=./train_dir/eval_$TEST_ID/stats.db

# 4. Check for wandb metrics filtering (if wandb is enabled)
grep -r "agent_raw" train_dir/test_$TEST_ID/wandb || echo "✓ No agent_raw metrics in wandb logs"
```

## Common Issues and Solutions

1. **Training runs forever**: Use `trainer.total_timesteps=X` to limit training steps
2. **Policy not found**: Ensure the policy_uri path matches the training run directory
3. **Recipe not found**: Check that recipe names match the short form (e.g., `arena`, `navigation`)
4. **Wrong directory picked up**: Always use the same TEST_ID across all commands
5. **Interactive tools hang**: Interactive play and replay tools may not work well in Claude Code due to browser
   requirements

## Interactive Tools

### Exploration and Debugging

```bash
# Interactive play for manual testing and exploration
uv run ./tools/run.py play arena policy_uri=file://./train_dir/test_$TEST_ID/checkpoints

# Interactive play with navigation environment
uv run ./tools/run.py play navigation policy_uri=file://./train_dir/test_$TEST_ID/checkpoints
```

## Navigation Evaluation Database

### Adding Policies to Evaluation Database

```bash
# Add a policy to the navigation evals database
uv run ./tools/run.py eval navigation policy_uri=POLICY_URI

# Analyze results with scorecard
uv run ./tools/run.py analyze navigation eval_db_uri=./path/to/eval/stats.db
```

## Recipe System

The system supports two-token syntax for conciseness:

- **Training**: `train arena` or `train navigation`
- **Evaluation**: `eval arena` or `eval navigation`
- **Analysis**: `analyze arena` or `analyze navigation`
- **Interactive**: `play arena` or `play navigation`
- **Replay**: `replay arena` for viewing recorded gameplay

Alternative dot-notation also works: `arena.train`, `navigation.eval`, etc.

### Discovering Available Tools

```bash
# List all tools in a specific recipe
uv run ./tools/run.py arena --list
uv run ./tools/run.py navigation --list

# List all recipes that provide a specific tool
uv run ./tools/run.py train --list          # Shows all recipes with train
uv run ./tools/run.py eval --list           # Shows all recipes with eval
```
