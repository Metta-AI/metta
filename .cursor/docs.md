## Testing the Codebase

To quickly test that training, simulation, and analysis are working correctly, use these commands:

### Quick 30-second test

```bash
# Set test ID for consistent file references
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID"

# 1. Train for 30 seconds (terminate with Ctrl+C) or use cursor config for auto-stop
uv run ./tools/train.py run=test_$TEST_ID
# OR with cursor config (auto-stops after 100k steps):
uv run ./tools/train.py +user=cursor run=cursor_$TEST_ID

# 2. Run a few simulations
uv run ./tools/sim.py run=eval_$TEST_ID policy_uri=file://./train_dir/test_$TEST_ID/checkpoints device=cpu sim=navigation

# 3. Analyze results
uv run ./tools/analyze.py run=analysis_$TEST_ID analysis.policy_uri=file://./train_dir/test_$TEST_ID/checkpoints analysis.eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

For more testing commands and options, see `.cursor/commands.md`.

### Smoke Tests

The codebase includes smoke tests that verify:

- WandB metrics are numeric (not string formatted)
- No `agent_raw/*` entries are created (prevented in mettagrid_env.py to avoid thousands of metrics)
- Stats database structure is correct

Run smoke tests with:

```bash
export TEST_ID=$(date +%Y%m%d_%H%M%S)
uv run ./tools/train.py run=smoke_$TEST_ID +smoke_test=true
uv run ./tools/sim.py run=smoke_eval_$TEST_ID +sim_job.smoke_test=true policy_uri=file://./train_dir/smoke_$TEST_ID/checkpoints device=cpu
```
