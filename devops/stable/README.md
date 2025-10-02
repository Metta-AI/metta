# SkyPilot Recipe Tests

A unified script for launching and checking SkyPilot test jobs across all active recipes.

## Basic Use

```bash
# Launch 5 jobs: one per recipe
./recipe_test.py launch

# Check results
./recipe_test.py check

# Check with detailed logs
./recipe_test.py check -l
```

## Test Configurations

- **Recipes tested**:
  - `arena_basic_easy_shaped`: Basic arena with easy shaping
  - `arena`: Standard arena recipe
  - `icl_resource_chain`: In-context learning resource chain
  - `navigation`: Navigation task
  - `navigation_sequence`: Sequential navigation task

- **Fixed parameters**:
  - Single node configuration
  - 50,000 timesteps
  - CI tests disabled

## Termination Tracking

The framework automatically parses and color-codes:

- **Exit codes**: 0 (green), non-zero (red)
- **Termination reasons**:
  - `job_completed` (green)
  - `heartbeat_timeout`, `max_runtime_reached` (yellow)
  - Other reasons (red)
- **Restart counts**: 0 (green), >0 (yellow)

## Command Options

### Launch Command

```bash
./recipe_test.py launch [options]
```

- `--base-name`: Base name for test runs (default: recipe_test)
- `--output-file`: JSON file to save results (default: recipe_test_jobs.json)
- `--skip-git-check`: Skip git state validation

### Check Command

```bash
./recipe_test.py check [options]
```

- `-f, --input-file`: JSON file to check (default: recipe_test_jobs.json)
- `-l, --logs`: Show detailed logs for each job
- `-n, --tail-lines`: Number of log lines to tail (default: 200)

## Output Format

Launching jobs produces a JSON file with the structure:

```json
{
  "test_run_info": {
    "base_name": "...",
    "launch_time": "...",
    "total_jobs": N,
    "successful_launches": N,
    "failed_launches": N
  },
  "launched_jobs": [...],
  "failed_launches": [...]
}
```

Each job entry includes:

- Recipe name and module path
- Test configuration (timesteps, nodes, CI status)
- Job ID and status information

## Examples

```bash
# Launch with custom base name
./recipe_test.py launch --base-name my_recipe_validation

# Check specific output file with more log lines
./recipe_test.py check -f my_recipes.json -n 500

# Launch without git validation
./recipe_test.py launch --skip-git-check

# Get help for specific commands
./recipe_test.py launch --help
./recipe_test.py check --help
```

## Purpose

This test suite validates that all active recipes can:

- Launch successfully on SkyPilot
- Complete training without errors
- Exit cleanly after the specified timesteps

It serves as a smoke test to ensure recipe configurations remain valid across updates.
