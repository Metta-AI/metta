# Job Testing Framework

Simple parallel job dispatcher for testing multiple recipes/configs at once.

## Quick Start

### Test All Recipes

```bash
# Test all recipes with default settings (50k timesteps)
./devops/test_recipes.py

# Test specific recipes
./devops/test_recipes.py arena navigation

# Custom configuration
./devops/test_recipes.py --timesteps 100000 --gpus 2 --nodes 1
```

### Test Cluster Configurations

```bash
# Test 1, 2, and 4 node configs
./devops/test_clusters.py

# Test specific node counts
./devops/test_clusters.py --nodes 1 8

# Include timeout tests
./devops/test_clusters.py --test-timeouts
```

## Architecture

### Core Components

**1. `job_runner.py`** - Low-level job execution
- `Job` - Abstract base class
- `LocalJob` - Run subprocess locally
- `RemoteJob` - Run on SkyPilot
- `JobResult` - Unified result type

**2. `job_dispatcher.py`** - Parallel job management
- `JobDispatcher` - Run multiple jobs in parallel
- `run_jobs_parallel()` - Convenience function

**3. Test Scripts** - Pre-configured test scenarios
- `test_recipes.py` - Test multiple recipes
- `test_clusters.py` - Test cluster configs

## Examples

### Custom Test Script

```python
#!/usr/bin/env -S uv run
from devops.job_dispatcher import JobDispatcher
from devops.job_runner import RemoteJob

# Create dispatcher
dispatcher = JobDispatcher(name="my_test")

# Add jobs
for config in ["small", "medium", "large"]:
    job = RemoteJob(
        name=config,
        module="recipe.train",
        args=[f"model.size={config}"],
        base_args=["--gpus=4"],
    )
    dispatcher.add_job(job)

# Run all in parallel
dispatcher.run_all()

# Wait and check results
results = dispatcher.wait_all(timeout_s=3600)
dispatcher.print_summary()
```

### Mixed Local and Remote

```python
from devops.job_runner import LocalJob, RemoteJob
from devops.job_dispatcher import run_jobs_parallel

jobs = [
    # Run tests locally
    LocalJob(name="unit_tests", cmd=["pytest", "tests/"]),

    # Run training remotely
    RemoteJob(name="train", module="recipe.train", args=["run=test"]),
]

results = run_jobs_parallel(jobs, name="ci_check")
```

## State Persistence

Job dispatcher automatically saves state to:
```
devops/job_dispatcher/state/{name}_{timestamp}.json
```

This includes:
- Job names and configs
- Submission status
- Completion status
- Results (exit codes, logs paths, job IDs)

You can load this later to check what happened:

```python
from devops.job_dispatcher import JobDispatcher

state = JobDispatcher.load_state("devops/job_dispatcher/state/recipe_test_2025-10-08T12-00-00.json")
if state:
    print(f"Test run: {state.name}")
    print(f"Completed: {state.completed}")
    for name, status in state.jobs.items():
        print(f"  {name}: {status}")
```

## Comparison to Old System

### Old System (cluster_test.py, recipe_test.py)

**Pros:**
- Built-in CLI (launch/check/kill subcommands)
- Rich status display with colors
- Termination reason parsing
- Restart count tracking

**Cons:**
- Tightly coupled to testing_helpers
- Harder to customize
- Separate implementations for each test type

### New System

**Pros:**
- Simple, composable primitives
- Easy to build custom test scripts
- Unified API for local/remote jobs
- Async/sync flexibility
- Less code to maintain

**Cons:**
- No built-in "check" command for old runs
- Less fancy status output (for now)
- No termination reason parsing (yet)

### Migration

If you need the old test scripts, they're still available:
```bash
# Old scripts (restored but not actively maintained)
./devops/skypilot/tests/cluster_test.py launch
./tests/experiments/recipes/recipe_test.py launch
```

But the new system is simpler and more flexible for most cases.

## Advanced Usage

### Custom Polling Logic

```python
dispatcher = JobDispatcher(name="custom")

# Add jobs...
dispatcher.run_all()

# Custom wait with progress updates
start = time.time()
while not dispatcher.is_all_complete():
    results = dispatcher.get_results()
    completed = sum(1 for r in results.values() if r is not None)
    print(f"Progress: {completed}/{len(results)} ({time.time() - start:.0f}s)")
    time.sleep(30)

results = dispatcher.get_results()
```

### Conditional Job Launching

```python
dispatcher = JobDispatcher(name="staged")

# Stage 1: Quick tests
quick_jobs = [LocalJob(...), LocalJob(...)]
for job in quick_jobs:
    dispatcher.add_job(job)

dispatcher.run_all()
quick_results = dispatcher.wait_all(timeout_s=600)

# Only proceed if quick tests passed
if all(r.success for r in quick_results.values()):
    # Stage 2: Expensive training
    train_job = RemoteJob(...)
    dispatcher.add_job(train_job)
    dispatcher.run_all()
```

## Future Enhancements

Possible additions (if needed):

1. **CLI for checking old runs** - `./check_jobs.py {state_file}`
2. **Termination reason parsing** - Extract heartbeat_timeout, etc from logs
3. **Restart count tracking** - Parse SkyPilot restart info
4. **Progress dashboard** - Web UI showing live job status
5. **Job dependencies** - Run job B only if job A passes

For now, kept minimal - add features when actually needed.
