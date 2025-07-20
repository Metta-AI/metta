# Metta Testing Commands

This file contains simple commands for testing the metta codebase. These commands are designed to run quickly and
provide useful feedback about whether the system is working correctly.

## Prerequisites

The project uses `uv` for Python package management. All Python commands should be run with `uv run` to ensure proper environment activation.

## Quick Test Commands (30-60 seconds total)

### Set Test ID First

```bash
# Set a unique test ID for this testing session
export TEST_ID=$(date +%Y%m%d_%H%M%S)
echo "Test ID: $TEST_ID"
```

### 1. Train for 30 seconds

```bash
# Basic training (will run indefinitely, terminate with Ctrl+C after ~30 seconds)
uv run ./tools/train.py run=test_$TEST_ID +hardware=macbook trainer.num_workers=2

# Smoke test training (runs with deterministic settings for CI/CD)
uv run ./tools/train.py run=smoke_$TEST_ID +hardware=macbook trainer.num_workers=2 +smoke_test=true

# Using cursor config (limited to 100k steps)
uv run ./tools/train.py +user=cursor run=cursor_$TEST_ID trainer.num_workers=2
```

### 2. Run simulations on trained model

```bash
# Run evaluations on the checkpoint from step 1
uv run ./tools/sim.py run=eval_$TEST_ID policy_uri=file://./train_dir/test_$TEST_ID/checkpoints device=cpu

# Run smoke test simulation (limited simulations, deterministic)
uv run ./tools/sim.py run=smoke_eval_$TEST_ID policy_uri=file://./train_dir/smoke_$TEST_ID/checkpoints device=cpu +sim_job.smoke_test=true

# Using cursor config
uv run ./tools/sim.py run=cursor_eval_$TEST_ID policy_uri=file://./train_dir/cursor_$TEST_ID/checkpoints +user=cursor
```

### 3. Analyze results

```bash
# Analyze the simulation results from step 2
uv run ./tools/analyze.py run=analysis_$TEST_ID analysis.policy_uri=file://./train_dir/test_$TEST_ID/checkpoints analysis.eval_db_uri=./train_dir/eval_$TEST_ID/stats.db

# Using cursor config
uv run ./tools/analyze.py run=cursor_analysis_$TEST_ID +user=cursor analysis.eval_db_uri=./train_dir/cursor_eval_$TEST_ID/stats.db
```

## One-Line Test Commands

### Basic 30-second test (copy-paste friendly)

```bash
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID" && uv run ./tools/train.py run=test_$TEST_ID +hardware=macbook trainer.total_timesteps=10000 trainer.num_workers=2
# After training completes or you Ctrl+C:
uv run ./tools/sim.py run=eval_$TEST_ID policy_uri=file://./train_dir/test_$TEST_ID/checkpoints device=cpu sim=navigation
uv run ./tools/analyze.py run=analysis_$TEST_ID analysis.policy_uri=file://./train_dir/test_$TEST_ID/checkpoints analysis.eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

### Using cursor config (auto-limited training)

```bash
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID"
uv run ./tools/train.py +user=cursor run=cursor_$TEST_ID
uv run ./tools/sim.py run=cursor_eval_$TEST_ID policy_uri=file://./train_dir/cursor_$TEST_ID/checkpoints +user=cursor sim=navigation
uv run ./tools/analyze.py run=cursor_analysis_$TEST_ID +user=cursor analysis.eval_db_uri=./train_dir/cursor_eval_$TEST_ID/stats.db
```

## Full Integration Test (2-3 minutes)

```bash
# Set test ID
export TEST_ID=$(date +%Y%m%d_%H%M%S)
echo "Running full integration test with ID: $TEST_ID"

# 1. Train for 100k steps (~1 minute on GPU)
uv run ./tools/train.py run=test_$TEST_ID trainer.total_timesteps=100000 trainer.checkpoint.checkpoint_interval=50 trainer.simulation.evaluate_interval=0 trainer.num_workers=2

# 2. Run limited simulations (~30 seconds)
uv run ./tools/sim.py run=eval_$TEST_ID policy_uri=file://./train_dir/test_$TEST_ID/checkpoints sim=navigation device=cpu

# 3. Analyze results
uv run ./tools/analyze.py run=analysis_$TEST_ID analysis.policy_uri=file://./train_dir/test_$TEST_ID/checkpoints analysis.eval_db_uri=./train_dir/eval_$TEST_ID/stats.db

# 4. Check for wandb metrics filtering (if wandb is enabled)
grep -r "agent_raw" train_dir/test_$TEST_ID/wandb || echo "✓ No agent_raw metrics in wandb logs"
```

## Common Issues and Solutions

1. **CUDA not available on Mac**: Always use `device=cpu` on macOS
2. **Training runs forever**: Use `trainer.total_timesteps=X` to limit training
3. **Simulations take too long**: Use `sim=navigation` to run only navigation tasks
4. **Smoke test failures**: Check that agent_raw metrics are filtered from wandb
5. **Wrong directory picked up**: Always use the same TEST_ID across all commands

## Interactive Tools

### Exploration and Debugging

```bash
# Interactive simulation for manual testing and exploration
uv run ./tools/play.py run=my_experiment +hardware=macbook wandb=off

# Interactive play with specific policy
uv run ./tools/play.py run=play_$TEST_ID policy_uri=file://./train_dir/test_$TEST_ID/checkpoints +hardware=macbook
```

## Navigation Evaluation Database

### Adding Policies to Evaluation Database

```bash
# Add a policy to the navigation evals database
uv run ./tools/sim.py eval=navigation run=RUN_NAME eval.policy_uri=POLICY_URI +eval_db_uri=wandb://artifacts/navigation_db

# Analyze results with heatmap
uv run ./tools/analyze.py run=analyze +eval_db_uri=wandb://artifacts/navigation_db analyzer.policy_uri=POLICY_URI
```

## Smoke Test Mode

When `+smoke_test=true` is added:

- Training: Verifies wandb metrics structure
- Simulation: Runs limited sims and verifies stats DB structure
- Both use deterministic seeds and settings for reproducibility
