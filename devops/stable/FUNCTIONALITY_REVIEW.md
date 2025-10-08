# Functionality Review: cluster_test / recipe_test vs job_runner

## Overview
This document compares the functionality in the deleted `cluster_test.py` and `recipe_test.py` scripts with the new `job_runner.py` and `release.py` implementation.

## Key Features Comparison

### 1. ✅ KEPT: Basic Job Execution
**Old System:**
- Launch jobs to SkyPilot
- Execute local commands
- Track job IDs and request IDs

**New System:**
- `LocalJob` and `RemoteJob` classes
- Unified `JobResult` dataclass
- Both async (submit + poll) and sync (wait) modes

**Status:** ✅ Fully maintained and improved

---

### 2. ❌ LOST: Multi-Job Batch Testing Framework

**Old System (cluster_test.py):**
```python
# Launch 12 jobs in a matrix:
# - 3 node configs (1, 2, 4 nodes)
# - 4 exit conditions (normal, heartbeat timeout, runtime timeout, cmd_fails)
./cluster_test.py launch
./cluster_test.py check
./cluster_test.py kill
```

**Old System (recipe_test.py):**
```python
# Launch 7 jobs (one per recipe)
./recipe_test.py launch
./recipe_test.py check
```

**What's Missing:**
1. **Batch job launching** - Launch multiple jobs in one command
2. **Job matrix configuration** - Test multiple node configs × conditions
3. **Result tracking & persistence** - Save all job info to JSON
4. **Status checking** - Check status of all jobs from saved JSON
5. **Bulk job killing** - Kill all jobs from a test run
6. **Color-coded status display** - Pretty printing of job statuses
7. **Termination reason parsing** - Parse and display why jobs ended

**New System:**
- Only supports launching ONE job at a time
- No batch testing framework
- release.py has state tracking but only for release validations

**Status:** ❌ Major functionality gap

---

### 3. ✅ KEPT: Job Monitoring & Log Fetching

**Old System:**
- Poll job logs via `tail_job_log(job_id)`
- Parse completion markers
- Extract exit codes

**New System:**
- `RemoteJob.is_complete()` - Check if job done
- `RemoteJob.get_logs()` - Fetch logs from SkyPilot
- `RemoteJob.wait()` - Block until complete with polling

**Status:** ✅ Maintained with cleaner API

---

### 4. ❌ LOST: Detailed Job Analysis

**Old System Features:**
```python
# Parse termination reasons from logs
- job_completed (normal)
- heartbeat_timeout
- max_runtime_reached
- Other termination reasons

# Parse restart counts
- Track how many times job restarted

# Color-coded summaries
- Exit codes: 0 (green), non-zero (red)
- Reasons: completed (green), timeout (yellow), other (red)
```

**New System:**
- Only extracts exit code via regex: `Exit code: (\d+)`
- No termination reason parsing
- No restart count tracking
- Basic result data structure

**Status:** ❌ Lost rich job analysis

---

### 5. ✅ KEPT: Git State Validation

**Old System:**
```python
launcher.check_git_state()  # Via testing_helpers
```

**New System:**
```python
import gitta as git
git.get_current_commit()
git.has_unstaged_changes()
```

**Status:** ✅ Maintained (improved with gitta library)

---

### 6. ❌ LOST: Test Configuration Framework

**Old System:**
```python
@dataclass
class TestCondition:
    name: str
    extra_args: list[str]
    description: str
    ci: bool = False

TEST_CONDITIONS = {
    "normal_completion": TestCondition(...),
    "heartbeat_timeout": TestCondition(...),
    "runtime_timeout": TestCondition(...),
}
```

**New System:**
- No structured test condition framework
- Arguments passed directly to job constructors
- No test matrix support

**Status:** ❌ Lost test configuration abstraction

---

### 7. ❌ LOST: BaseTestRunner Framework

**Old System:**
```python
class BaseTestRunner:
    - create_parser()      # Subcommand CLI (launch/check/kill)
    - check_tests()        # Check job statuses with pretty output
    - kill_tests()         # Bulk kill jobs
    - launch_tests()       # Abstract method for subclasses
```

**Features:**
- Consistent CLI across test scripts
- Subcommand structure (launch/check/kill)
- JSON persistence
- Pretty status tables
- Detailed log viewing

**New System:**
- No test runner framework
- Each script implements its own CLI

**Status:** ❌ Lost reusable test infrastructure

---

## What We Gained

### 1. Unified Job Abstraction
```python
# Old: Different APIs for local vs remote
run_local(...)  # Returns LocalJobResult
run_remote(...) # Returns RemoteJob with different API

# New: Same API for both
job = LocalJob(...)  # or RemoteJob(...)
result = job.wait()  # Both return JobResult
```

### 2. Async/Sync Flexibility
```python
# Old: Only sync (blocking wait)
job = run_remote(...)
exit_code = job.wait()

# New: Both async and sync
job = RemoteJob(...)
job.submit()  # Async - returns immediately

# Option 1: Poll manually
while not job.is_complete():
    logs = job.get_logs()
    time.sleep(10)

# Option 2: Block and wait
result = job.wait(stream_output=True)
```

### 3. Better State Management (release.py)
- Persistent state with resume capability
- Validation result tracking
- Auto-resume on failure

---

## Recommendations

### High Priority: Restore Batch Testing Framework

The deleted test scripts provided critical infrastructure for:
1. **CI/CD Testing** - Validate cluster configs and recipes work
2. **Pre-release Validation** - Test multiple configurations before release
3. **Debugging** - Quickly check status of multiple test jobs

**Suggested Approach:**

Create `devops/test_runner.py` that provides:

```python
from devops.job_runner import RemoteJob, JobResult
from dataclasses import dataclass

@dataclass
class TestMatrix:
    """Configuration for a matrix of tests."""
    recipes: list[str]
    node_configs: list[int]
    conditions: dict[str, TestCondition]

class BatchTestRunner:
    """Run and track multiple jobs."""

    def __init__(self, name: str):
        self.name = name
        self.jobs: dict[str, RemoteJob] = {}

    def launch_matrix(self, matrix: TestMatrix) -> None:
        """Launch all jobs in matrix."""
        for recipe in matrix.recipes:
            for nodes in matrix.node_configs:
                for cond_name, condition in matrix.conditions.items():
                    job = self._create_job(recipe, nodes, condition)
                    job.submit()
                    self.jobs[f"{recipe}_{nodes}n_{cond_name}"] = job

    def wait_all(self) -> dict[str, JobResult]:
        """Wait for all jobs to complete."""
        results = {}
        for name, job in self.jobs.items():
            results[name] = job.wait()
        return results

    def save_state(self, path: str) -> None:
        """Save job info and results to JSON."""
        ...

    def check_status(self) -> None:
        """Print status table for all jobs."""
        ...
```

### Medium Priority: Rich Job Analysis

Add to `JobResult`:
```python
@dataclass
class JobResult:
    name: str
    exit_code: int
    logs_path: str
    job_id: Optional[str] = None
    duration_s: Optional[float] = None

    # Add these:
    termination_reason: Optional[str] = None  # heartbeat_timeout, max_runtime_reached, etc.
    restart_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
```

Parse termination info from logs in `RemoteJob._fetch_result()`.

### Low Priority: Test Runner CLI

Consider whether to:
1. **Keep separate** - `cluster_test.py`, `recipe_test.py` as thin wrappers around `BatchTestRunner`
2. **Unify** - Single `devops/test_runner.py` with subcommands for different test types
3. **Skip** - Just use release.py for validation, don't restore matrix testing

---

## Summary

**Kept:**
- ✅ Job execution (local & remote)
- ✅ Job monitoring & log fetching
- ✅ Git state validation
- ✅ Exit code extraction

**Lost:**
- ❌ Batch job launching
- ❌ Job matrix testing
- ❌ Result persistence & tracking (outside release.py)
- ❌ Status checking CLI
- ❌ Bulk job management
- ❌ Rich termination analysis
- ❌ Test configuration framework
- ❌ Reusable test runner infrastructure

**Gained:**
- ✨ Unified job abstraction
- ✨ Async/sync flexibility
- ✨ Cleaner API design
- ✨ Better state management (in release.py)

The new `job_runner.py` is a better **primitive** but we lost the **higher-level testing framework** built on top of it.
