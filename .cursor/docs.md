## Testing the Codebase

To quickly test that training, simulation, and analysis are working correctly, use these commands:

### Quick 30-second test

```bash
# Set test ID for consistent file references
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID"

# 1. Train using cursor config (auto-stops after 100k steps):
./tools/train.py +user=cursor run=cursor_$TEST_ID

# 2. Run a few simulations
./tools/sim.py run=eval_$TEST_ID policy_uri=file://./train_dir/cursor_$TEST_ID/checkpoints +user=cursor sim=navigation

# 3. Analyze results
./tools/analyze.py run=analysis_$TEST_ID +user=cursor analysis.policy_uri=file://./train_dir/cursor_$TEST_ID/checkpoints analysis.eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

For more testing commands and options, see `.cursor/commands.md`.

### User and Hardware Configurations

- **+user=** loads user-specific configs from `configs/user/`:
  - `cursor`: Limited training (100k steps), CPU device - recommended for testing
  - `relh`, `alex`, `berekuk`, `daphne`: Team member configurations
  - See `configs/user/` for all available user configs
- **+hardware=** loads hardware configs from `configs/hardware/` (e.g., `macbook` for CPU-only)

### Smoke Tests

The codebase includes smoke tests that verify:

- WandB metrics are numeric (not string formatted)
- No `agent_raw/*` entries are created (prevented in mettagrid_env.py to avoid thousands of metrics)
- Stats database structure is correct

Run smoke tests with:

```bash
export TEST_ID=$(date +%Y%m%d_%H%M%S)
./tools/train.py +user=cursor run=smoke_$TEST_ID +smoke_test=true
./tools/sim.py run=smoke_eval_$TEST_ID +sim_job.smoke_test=true policy_uri=file://./train_dir/smoke_$TEST_ID/checkpoints +user=cursor
```

### Troubleshooting

If you get "Unrecognized file format" errors when resuming training, remove the old checkpoint directory:
```bash
rm -rf train_dir/YOUR_RUN_NAME
```
