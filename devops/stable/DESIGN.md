# Stable Release System - Technical Design

## Architecture

```
release_stable.py (CLI + Gates + Plan)
   └─ orchestrator.py (run_validations + StateManager)
       ├─ devops/job_runner.py (run_local / run_remote)
       │   └─ devops/skypilot/utils/ (reuses existing infrastructure)
       ├─ acceptance.py (LogRegexMetrics + evaluate_thresholds)
       └─ models.py (data types)
```

## Files (4 in devops/stable/, 1 shared in devops/, ~1,036 lines total)

### models.py (70 lines)
Core data types using Pydantic:
- `Location` - LOCAL/REMOTE enum
- `Lifecycle` - PENDING/RUNNING/COMPLETED
- `Outcome` - PASSED/FAILED/SKIPPED/INCONCLUSIVE
- `ThresholdCheck` - Single threshold with operator, expected value, actual value
- `Artifact` - Log/JSON/image reference with type and URI
- `GateResult` - Result of a gate check (bug check, workflow validation)
- `RunResult` - Result of a single validation run
- `Validation` - Configuration for a validation run
- `ReleaseState` - Complete state of a release (gates + validations)

### acceptance.py (83 lines)
Metrics extraction and threshold evaluation:
- `evaluate_thresholds()` - Evaluates metrics against ThresholdCheck list
- `MetricsSource` - Protocol for custom metrics extractors
- `LogRegexMetrics` - Extracts SPS and eval_success_rate from logs

**Key decisions:**
- Missing metrics are **hard failures** with note "metric missing"
- Single metrics source (no chaining) - simpler to reason about
- Regex patterns are simple and focused (SPS, eval rate)

### devops/job_runner.py (176 lines) - General-Purpose Runner
Job execution abstraction (shared across codebase):
- `LocalJob` - Completed local job with logs and exit code
- `RemoteJob` - Remote SkyPilot job with polling and log fetching
- `run_local()` - Run arbitrary commands locally via subprocess
- `run_remote()` - Launch jobs on SkyPilot using existing infrastructure

**Key decisions:**
- Function-based API (not class-based) for simplicity
- Local jobs run synchronously (subprocess.run)
- Remote jobs use existing `SkyPilotTestLauncher` (no duplication)
- Uses `tail_job_log()` from `job_helpers.py` for log fetching
- Timeouts return exit code 124
- Located in `devops/` for sharing across stable, sweep, and adaptive systems

**Reuses existing infrastructure:**
- `devops/skypilot/utils/testing_helpers.py`:
  - `SkyPilotTestLauncher` - Job launching with retry logic
  - `LaunchedJob` - Job metadata tracking
- `devops/skypilot/utils/job_helpers.py`:
  - `tail_job_log()` - Log fetching via Sky SDK
  - No custom SkyPilot code, leverages tested infrastructure

### orchestrator.py (190 lines)
Validation orchestration with state management:
- `StateManager` - JSON state persistence (create/save/load)
- `run_validations()` - Main entrypoint that runs validations in parallel
- `record_workflow_gate()` - Records gate result in state
- `print_validation_summary()` - Pretty-prints validation results

**Key decisions:**
- Parallel execution with ThreadPoolExecutor (max 4 workers)
- State saved incrementally after each validation
- Logs **always** attached as artifacts (success, failure, timeout)
- Defaults to LogRegexMetrics (simple)

**Flow:**
1. Create or load state
2. Launch validations in parallel (ThreadPoolExecutor)
3. For each validation:
   - Run job (local or remote)
   - Wait for completion
   - Extract metrics from logs
   - Evaluate thresholds
   - Mark as completed/failed
   - Attach log artifact
   - Save state
4. Return final state

### release_stable.py (480 lines)
CLI script with gates and release plan:
- `Gate` - Protocol for release gates
- `BugGate` - Asana bug check gate
- `WorkflowGate` - Runs validation workflows
- `get_release_plan()` - Returns dict with gates and validations
- `step_1_prepare_release_branch()` - Creates and pushes branch
- `step_2_bug_status_check()` - Runs Asana check or manual confirmation
- `step_3_workflow_validation()` - Runs validations via WorkflowGate
- `step_4_release()` - Prints release instructions
- `step_5_announce()` - Prints announcement template

**Key decisions:**
- Gates and plan are inline (not separate files) - single release process
- Gates are Protocol-based for extensibility
- Steps 4 and 5 are print-only (don't execute git/PR operations)

## Design Principles

1. **Reuse existing infrastructure** - Don't duplicate SkyPilot job management
2. **Simple defaults** - LogRegexMetrics by default, no complex chaining
3. **Hard failures** - Missing metrics fail explicitly
4. **Always attach logs** - Debugging requires logs, attach them always
5. **Parallel where possible** - ThreadPoolExecutor for remote jobs
6. **Incremental state** - Save after each validation for resumability
7. **Type-safe** - Pydantic models, Location enum, no string literals
8. **Single release plan** - One plan inline in release_stable.py

## What's Implemented

✅ Parallel validation execution
✅ Local and remote runners
✅ Reuses `SkyPilotTestLauncher` and `LaunchedJob` from existing infrastructure
✅ Reuses `tail_job_log()` for fetching logs
✅ Regex metrics extraction (SPS, eval_success_rate)
✅ Threshold evaluation with operators
✅ State persistence to JSON
✅ Gate system (BugGate, WorkflowGate)
✅ Asana integration for bug checks
✅ CLI with 5 steps
✅ Check mode (dry-run)
✅ Log artifact tracking

## What's NOT Implemented (TODOs)

❌ W&B metrics extraction - regex only for now
❌ Streaming local logs - buffers everything
❌ PR body generation - step 4 just prints instructions
❌ Automatic git operations - user does git commands manually
❌ Retry logic for transient failures
❌ Regression checks (compare to previous release)
❌ Performance regression detection (SPS delta vs baseline)

## Iterating on the Design

### Adding a new validation

Edit `get_release_plan()` in `release_stable.py`:
```python
Validation(
    name="new_validation",
    module="experiments.recipes.new_recipe.train",
    location=Location.REMOTE,
    args=["trainer.total_timesteps=100000"],
    timeout_s=7200,
    acceptance=[
        ThresholdCheck(key="sps_max", op=">=", expected=50000),
    ],
)
```

### Adding a new metric

1. Add regex pattern to `LogRegexMetrics` in `acceptance.py`
2. Extract and return in `extract()` method
3. Use in `ThresholdCheck` in `get_release_plan()`

### Adding a new gate

1. Create class implementing `Gate` protocol in `release_stable.py`
2. Add to `gates` list in `get_release_plan()`
3. Gate runs in order, fails fast if `required=True`

### Testing locally

Run a single validation in isolation:
```python
from devops.stable.orchestrator import run_validations
from devops.stable.models import Location, ThresholdCheck, Validation

v = Validation(
    name="test",
    module="experiments.recipes.arena_basic_easy_shaped.train",
    location=Location.LOCAL,
    args=["run=test", "trainer.total_timesteps=1000", "wandb.enabled=false"],
    timeout_s=600,
    acceptance=[ThresholdCheck(key="sps_max", op=">=", expected=1000)],
)

state = run_validations(version="dev-test", validations=[v])
print(state.validation_summary)
```

## Performance

- **Local smoke test**: ~2-3 minutes (1k steps)
- **Remote 50k test**: ~10-15 minutes (depends on cluster)
- **Parallel execution**: Up to 4 validations concurrently
- **State overhead**: <1ms per save (JSON serialization)

## Code Complexity

- **Total lines**: ~1,036 (5 files)
- **Cyclomatic complexity**: Low (no deep nesting, single-path logic)
- **Dependencies**: Pydantic, asana (optional), existing devops/skypilot utils
- **Test coverage**: TODO (no tests yet)
