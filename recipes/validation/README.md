# Recipe Validation

### Two-Tier Testing System

**CI Suite** (`ci_suite.py`) - Fast smoke tests
- **Purpose**: Verify recipes don't crash, not performance
- **Runs**: On every commit (will run on GitHub pre-merge soon)
- **Characteristics**: 10k timesteps, 5-minute timeouts, runs locally
- **Run with**: `metta ci --stage recipe-tests`

**Stable Suite** (`stable_suite.py`) - Performance validation
- **Purpose**: Track end-to-end performance (SPS, learning outcomes)
- **Runs**: During releases on remote infrastructure
- **Characteristics**: 100M-2B timesteps, multi-GPU (1-16), acceptance criteria for metrics
- **Run with**: Part of release automation (or manually via job runner tools)

### When Adding a Prod Recipe

Add **both** test types to your recipe:

1. **CI test** in `ci_suite.py`: Minimal smoke test (just verify it runs)
2. **Stable test** in `stable_suite.py`: Full training run with performance criteria

### Quick Commands

```bash
# Run stable suite validation (performance tests on remote infrastructure)
python devops/stable/stable.py validate

# Run stable validation with job filtering
python devops/stable/stable.py validate --job "arena*"

# Retry failed stable validation jobs
python devops/stable/stable.py validate --retry-failed
```

---

## Test Suites

### CI Suite (`ci_suite.py`)

**Purpose**: Quick smoke tests to verify recipes don't crash

**When it runs**: On every commit (eventually on GitHub before merge via `metta ci --stage recipe-tests`)

**Characteristics**:
- **Fast**: Low timesteps (e.g., 10k), short timeouts (e.g., 5 minutes)
- **Goal**: Ensure basic functionality works, NOT performance
- **What it checks**: Processes run without crashing, basic operations work
- **Infrastructure**: Runs locally or on lightweight CI runners

**Example test**:
```python
arena_train = JobConfig(
    name=f"{run_prefix}.arena_train",
    module="prod.arena_basic_easy_shaped.train",
    args=[
        f"run={run_prefix}.arena_train",
        "trainer.total_timesteps=10000",  # Just enough to verify it works
        "checkpointer.epoch_interval=1",
    ],
    timeout_s=300,  # 5 minutes
    is_training_job=True,
)
```

### Stable Suite (`stable_suite.py`)

**Purpose**: Comprehensive performance validation for releases

**When it runs**: During release validation on remote infrastructure

**Characteristics**:
- **Comprehensive**: Realistic training runs (100M-2B timesteps)
- **Goal**: Track end-to-end performance (SPS, learning outcomes)
- **What it checks**: Performance metrics meet acceptance criteria
- **Infrastructure**: Multi-GPU clusters (1-16 GPUs), long-running jobs

**Example test**:
```python
arena_train_100m = JobConfig(
    name="arena_single_gpu_100m",
    module="prod.arena_basic_easy_shaped.train",
    args=["trainer.total_timesteps=100000000"],  # Real training
    timeout_s=7200,  # 2 hours
    remote=RemoteConfig(gpus=1, nodes=1),
    is_training_job=True,
    metrics_to_track=["overview/sps", "env_agent/heart.gained"],
    acceptance_criteria=[
        AcceptanceCriterion(metric="overview/sps", operator=">=", threshold=40000),
        AcceptanceCriterion(metric="env_agent/heart.gained", operator=">", threshold=0.1),
    ],
)
```

## Key Differences

| Aspect | CI Suite | Stable Suite |
|--------|----------|--------------|
| **Purpose** | Verify no crashes | Validate performance |
| **Timesteps** | 10k | 100M-2B |
| **Duration** | 5 minutes | Hours (up to 48h) |
| **Infrastructure** | Local/lightweight | Multi-GPU clusters |
| **Checks** | Basic functionality | SPS, learning curves, acceptance criteria |
| **When** | Every commit | Release validation |
| **Command** | `metta ci --stage recipe-tests` | Release automation |

## Adding Tests for a New Recipe

When adding a new production recipe to `recipes/prod/`, you should add both:

1. **CI smoke test** in `ci_suite.py`:
   - Keep it minimal (10k timesteps, 5-minute timeout)
   - Just verify train/eval/play don't crash
   - No performance requirements

2. **Stable performance test** in `stable_suite.py`:
   - Use realistic training configuration
   - Track relevant metrics (SPS, task-specific outcomes)
   - Set acceptance criteria for performance
   - Test both single-GPU and multi-GPU if applicable

## Running Tests

```bash
# Run CI smoke tests locally
metta ci --stage recipe-tests

# Stable suite runs automatically during releases
# Can also run manually with job runner tools
```

## Test Organization

Both test suites follow the same pattern:

```python
def get_ci_jobs() -> tuple[list[JobConfig], str]:
    """Return list of CI smoke test jobs and group name."""
    # Create job configs for each recipe
    # Keep fast and minimal

def get_stable_jobs() -> list[JobConfig]:
    """Return list of stable performance validation jobs."""
    # Create comprehensive training jobs
    # Include acceptance criteria
```

The `runner.py` module provides utilities for executing these job configurations.
