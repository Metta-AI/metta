## Testing the Codebase

To quickly test that training, simulation, and analysis are working correctly, use these commands:

### Quick 30-second test

```bash
# Set test ID for consistent file references
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID"

# 1. Train for 30 seconds (terminate with Ctrl+C) or limit with total_timesteps
uv run ./tools/run.py experiments.recipes.arena.train run=test_$TEST_ID
# OR with limited timesteps for auto-stop:
uv run ./tools/run.py experiments.recipes.arena.train run=test_$TEST_ID trainer.total_timesteps=100000

# 2. Run evaluations
uv run ./tools/run.py experiments.recipes.arena.evaluate policy_uri=file://./train_dir/test_$TEST_ID/checkpoints

# 3. Analyze results
uv run ./tools/run.py experiments.recipes.arena.analyze eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

For more testing commands and options, see `.cursor/commands.md`.

### Different Recipe Types

The system supports different training and evaluation environments:

- **Arena Recipe**: Multi-agent competitive environments

  ```bash
  uv run ./tools/run.py experiments.recipes.arena.train run=my_experiment
  uv run ./tools/run.py experiments.recipes.arena.evaluate policy_uri=file://./checkpoints
  ```

- **Navigation Recipe**: Single-agent navigation tasks

  ```bash
  uv run ./tools/run.py experiments.recipes.navigation.train run=my_experiment
  uv run ./tools/run.py experiments.recipes.navigation.evaluate policy_uri=file://./checkpoints
  ```

- **Interactive Testing**: Browser-based interactive testing (Note: may not work well in Claude Code)
  ```bash
  uv run ./tools/run.py experiments.recipes.arena.play policy_uri=file://./checkpoints
  ```
