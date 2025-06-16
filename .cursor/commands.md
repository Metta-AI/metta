# Metta Testing Commands

This file contains simple commands for testing the metta codebase. These commands are designed to run quickly and provide useful feedback about whether the system is working correctly.

## Recent Changes (December 2025)

### MettaAgent Refactoring

The codebase underwent a major refactoring to separate the neural network implementation from the wrapper/metadata:
- `MettaAgent` is now a wrapper class that combines model functionality with metadata storage
- `BrainPolicy` contains the actual neural network implementation (previously called MettaAgent)
- `PolicyRecord` has been removed - all its functionality is now in MettaAgent
- This change enables future versioning support for loading old models when the code changes

**Impact on commands**: No changes to command syntax. The training, simulation, and analysis commands work the same way.

### macOS Compatibility

**Important for macOS users**: Always use `+hardware=macbook` or explicitly set `device=cpu` to avoid MPS (Metal Performance Shaders) errors. The system may try to use MPS by default which is not fully supported.

## Prerequisites

Make sure you have activated the virtual environment:

```bash
source .venv/bin/activate
```

## Quick Test Commands (30-60 seconds total)

### Set Test ID First

```bash
# Set a unique test ID for this testing session
export TEST_ID=$(date +%Y%m%d_%H%M%S)
echo "Test ID: $TEST_ID"
```

### 1. Train for 30 seconds

```bash
# Using cursor config (limited to 100k steps, auto-stops)
./tools/train.py +user=cursor run=cursor_$TEST_ID

# For macOS users - use hardware config
./tools/train.py +user=cursor run=test_$TEST_ID +hardware=macbook

# Manual training (will run indefinitely, terminate with Ctrl+C after ~30 seconds)
./tools/train.py +user=cursor run=test_$TEST_ID trainer.num_workers=2

# Smoke test training (runs with deterministic settings for CI/CD)
./tools/train.py +user=cursor run=smoke_$TEST_ID +smoke_test=true
```

### 2. Run simulations on trained model

```bash
# Run evaluations on the checkpoint from step 1
./tools/sim.py run=cursor_eval_$TEST_ID policy_uri=file://./train_dir/cursor_$TEST_ID/checkpoints +user=cursor

# Run smoke test simulation (limited simulations, deterministic)
./tools/sim.py run=smoke_eval_$TEST_ID policy_uri=file://./train_dir/smoke_$TEST_ID/checkpoints +user=cursor +sim_job.smoke_test=true

# Specific simulation task
./tools/sim.py run=eval_$TEST_ID policy_uri=file://./train_dir/cursor_$TEST_ID/checkpoints device=cpu sim=navigation
```

### 3. Analyze results

```bash
# Analyze the simulation results from step 2
./tools/analyze.py run=cursor_analysis_$TEST_ID +user=cursor analysis.eval_db_uri=./train_dir/cursor_eval_$TEST_ID/stats.db

# Analyze with specific checkpoint
./tools/analyze.py run=analysis_$TEST_ID +user=cursor analysis.policy_uri=file://./train_dir/cursor_$TEST_ID/checkpoints analysis.eval_db_uri=./train_dir/cursor_eval_$TEST_ID/stats.db
```

## One-Line Test Commands

### Basic 30-second test (copy-paste friendly)

```bash
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID"
./tools/train.py +user=cursor run=cursor_$TEST_ID
./tools/sim.py run=cursor_eval_$TEST_ID policy_uri=file://./train_dir/cursor_$TEST_ID/checkpoints +user=cursor sim=navigation
./tools/analyze.py run=cursor_analysis_$TEST_ID +user=cursor analysis.eval_db_uri=./train_dir/cursor_eval_$TEST_ID/stats.db
```

### Manual control test

```bash
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID" && ./tools/train.py +user=cursor run=test_$TEST_ID trainer.total_timesteps=10000 trainer.num_workers=2
# After training completes:
./tools/sim.py run=eval_$TEST_ID policy_uri=file://./train_dir/test_$TEST_ID/checkpoints device=cpu sim=navigation
./tools/analyze.py run=analysis_$TEST_ID analysis.policy_uri=file://./train_dir/test_$TEST_ID/checkpoints analysis.eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

### Using Pytorch Initial Policy

```bash
# Load an external PytorchAgent model as initial policy
./tools/train.py +user=cursor run=pytorch_test_$TEST_ID +hardware=macbook trainer.initial_policy.uri=pytorch://checkpoints/metta-new/metta.pt
```
