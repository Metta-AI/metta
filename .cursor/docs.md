## Testing the Codebase

To quickly test that training, simulation, and analysis are working correctly, use these commands:

### Discovering Available Tools

Use `--list` to see what tools are available:

```bash
# List all tools in a specific recipe
uv run ./tools/run.py arena --list

# List all recipes that provide a specific tool (e.g., train, eval)
uv run ./tools/run.py train --list
uv run ./tools/run.py eval --list
```

### Quick 30-second test

```bash
# Set test ID for consistent file references
export TEST_ID=$(date +%Y%m%d_%H%M%S) && echo "Test ID: $TEST_ID"

# 1. Train for 30 seconds (terminate with Ctrl+C) or limit with total_timesteps
uv run ./tools/run.py train arena run=test_$TEST_ID
# OR with limited timesteps for auto-stop:
uv run ./tools/run.py train arena run=test_$TEST_ID trainer.total_timesteps=100000

# 2. Run evaluations
uv run ./tools/run.py eval arena policy_uri=file://./train_dir/test_$TEST_ID/checkpoints

# 3. Analyze results
uv run ./tools/run.py analyze arena eval_db_uri=./train_dir/eval_$TEST_ID/stats.db
```

For more testing commands and options, see `.cursor/commands.md`.

### Different Recipe Types

The system supports different training and evaluation environments:

- **Arena Recipe**: Multi-agent competitive environments

  ```bash
  uv run ./tools/run.py train arena run=my_experiment
  uv run ./tools/run.py eval arena policy_uri=file://./train_dir/my_run/checkpoints
  ```

- **Navigation Recipe**: Single-agent navigation tasks

  ```bash
  uv run ./tools/run.py train navigation run=my_experiment
  uv run ./tools/run.py eval navigation policy_uri=file://./train_dir/my_run/checkpoints
  ```

- **Interactive Testing**: Browser-based interactive testing (Note: may not work well in Claude Code)
  ```bash
  uv run ./tools/run.py play arena policy_uri=file://./train_dir/my_run/checkpoints
  ```
