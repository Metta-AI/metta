# SkyPilot Cluster Tests

A unified script for launching and checking multi-node SkyPilot test jobs with various exit conditions.

## Basic Use

```bash
# Launch 12 jobs: 3 node configs × 4 exit conditions
./cluster_test.py launch

# Check results
./cluster_test.py check

# Check with detailed logs
./cluster_test.py check -l

# Kill all jobs
./cluster_test.py kill
```

## Test Configurations (3×3 Matrix)

- **Node configurations**: 1, 2, 4 nodes
- **Exit conditions**:
  - Normal completion (50k timesteps)
  - Heartbeat timeout (1 second)
  - Runtime timeout (0.03 hours)
- **CI tests**: Enabled for runtime timeout jobs

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
./cluster_test.py launch [options]
```

- `--base-name`: Base name for test runs (default: cluster_test)
- `--output-file`: JSON file to save results (default: cluster_test_jobs.json)
- `--skip-git-check`: Skip git state validation

### Check Command

```bash
./cluster_test.py check [options]
```

- `-f, --input-file`: JSON file to check (default: cluster_test_jobs.json)
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

## Examples

```bash
# Launch with custom base name
./cluster_test.py launch --base-name my_cluster_test

# Check specific output file with more log lines
./cluster_test.py check -f my_output.json -n 500

# Launch without git validation
./cluster_test.py launch --skip-git-check

# Get help for specific commands
./cluster_test.py launch --help
./cluster_test.py check --help
```
