# Issue: Skypilot Collector Hanging Despite Timeout Protection

**Date**: 2025-10-24
**Severity**: High - Both production and dev cronjobs are failing
**Status**: Under investigation

## Problem Description

The Skypilot collector is hanging indefinitely when attempting to start a local SkyPilot API server, causing the entire collection job to fail despite having 120-second timeout protection in place.

### Observed Symptoms

1. **Production cronjob**: Running for 81+ minutes (should complete in ~1-2 minutes)
2. **Dev cronjob**: Last 2 runs failed with `BackoffLimitExceeded`
3. **Both environments affected**: This is not specific to the recent GitHub metrics changes (sha-5d549e5)
4. **Pattern started**: ~4 hours ago (multiple consecutive failures)

### Evidence from Logs

From production pod `dashboard-cronjob-dashboard-cronjob-29355810-vqvkc`:

```
================================================================================
Running skypilot collector...
================================================================================
Collecting Skypilot job metrics...
[2mFailed to connect to SkyPilot API server at http://127.0.0.1:46580. Starting a local server.[0m
[33mYour SkyPilot API server machine only has 2.0GB memory available. At least 2GB is recommended to support higher load with better performance.[0m
[0m[32m✓ SkyPilot API server started. [0m[0m
```

The collector appears to start the SkyPilot API server successfully but then hangs - no further progress is logged.

### Current Timeout Protection

**File**: `devops/datadog/scripts/run_all_collectors.py`

```python
# Per-collector timeout in seconds (2 minutes default)
COLLECTOR_TIMEOUT = 120

# In main():
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(run_collector, name, True, False, False)
    try:
        metrics = future.result(timeout=COLLECTOR_TIMEOUT)
        # ... handle success
    except TimeoutError:
        # ... handle timeout
        print(f"❌ {name}: Timeout after {COLLECTOR_TIMEOUT}s")
        print("   Skipping to next collector...")
```

**Expected behavior**: After 120 seconds, the timeout should fire, log the error, and continue to the next collector.

**Actual behavior**: The timeout mechanism is not working - the job hangs indefinitely.

## Root Cause Analysis

### Why Timeout Protection Is Failing

The Python `ThreadPoolExecutor.future.result(timeout=...)` timeout only works for pure Python code. When the SkyPilot library starts a subprocess for the API server, the timeout doesn't kill the subprocess - it only abandons the Python thread.

**The problem**:
1. SkyPilot collector calls `sky.status()` or `sky.jobs.queue()`
2. This starts a subprocess: SkyPilot API server (`http://127.0.0.1:46580`)
3. The subprocess hangs or takes indefinitely long
4. Python's `future.result(timeout=120)` fires after 120s
5. **BUT**: The subprocess is still running in the background
6. The main process can't exit because the subprocess is still alive
7. Job hangs indefinitely despite timeout protection

### Why It Started Happening Now

This could be caused by:
- SkyPilot API server configuration changes
- Network connectivity issues to SkyPilot clusters
- Resource constraints (pod shows "only 2.0GB memory available")
- SkyPilot library version changes
- Changes in cluster state or job queue

## Proposed Solutions

### Option 1: Process-Level Timeout (Recommended)

Use `signal.alarm()` or subprocess timeout to kill the entire process tree:

```python
import signal
import subprocess

def timeout_handler(signum, frame):
    raise TimeoutError("Collector exceeded timeout")

# Set alarm before running collector
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(COLLECTOR_TIMEOUT)

try:
    metrics = run_collector(name, True, False, False)
    signal.alarm(0)  # Cancel alarm
except TimeoutError:
    signal.alarm(0)
    print(f"❌ {name}: Timeout after {COLLECTOR_TIMEOUT}s")
    # Kill any child processes
    # Continue to next collector
```

**Pros**: Kills subprocesses, works reliably
**Cons**: Unix-only (not Windows), more complex

### Option 2: Subprocess Wrapper

Run each collector in a subprocess with timeout:

```python
import subprocess

result = subprocess.run(
    ["python", "-m", "devops.datadog.scripts.run_collector", name, "--push"],
    timeout=COLLECTOR_TIMEOUT,
    capture_output=True,
)
```

**Pros**: Simple, kills subprocess tree automatically
**Cons**: Slower (process overhead), harder to get metrics back

### Option 3: Disable SkyPilot Local Server

Configure SkyPilot to not start a local server, require connection to remote API:

```python
# In skypilot collector
import os
os.environ['SKYPILOT_DISABLE_LOCAL_SERVER'] = '1'
```

**Pros**: Addresses root cause
**Cons**: May require infrastructure changes, might not work

### Option 4: Skip SkyPilot Temporarily

Comment out SkyPilot from collector list until fixed:

```python
COLLECTORS = [
    "github",
    "kubernetes",
    "ec2",
    # "skypilot",  # DISABLED: hangs indefinitely (ISSUE-skypilot-collector-timeout.md)
    "wandb",
    "asana",
    "health_fom",
]
```

**Pros**: Immediate fix, other collectors continue working
**Cons**: Lose SkyPilot metrics

## Troubleshooting Steps

### 1. Reproduce Locally

```bash
# Try to reproduce the hang locally
cd /Users/rwalters/GitHub/metta
uv run python devops/datadog/scripts/run_collector.py skypilot --verbose

# With timeout test
timeout 120 uv run python devops/datadog/scripts/run_collector.py skypilot --verbose
```

Expected: Should either complete successfully or timeout after 120s.

### 2. Check SkyPilot Configuration

```bash
# Check if SkyPilot can connect to clusters
sky status --all

# Check job queue
sky jobs queue --all
```

### 3. Test with Subprocess Timeout

Create a test wrapper to see if subprocess timeout works:

```bash
# Test subprocess timeout mechanism
python -c "
import subprocess
result = subprocess.run(
    ['python', '-m', 'devops.datadog.scripts.run_collector', 'skypilot', '--verbose'],
    timeout=30,
    capture_output=True
)
print(result.stdout.decode())
"
```

### 4. Check Kubernetes Pod Resources

```bash
# Check if resource constraints are causing issues
kubectl describe pod dashboard-cronjob-dashboard-cronjob-29355810-vqvkc -n monitoring

# Check pod memory/CPU usage
kubectl top pod dashboard-cronjob-dashboard-cronjob-29355810-vqvkc -n monitoring
```

## Immediate Action Required

1. **Verify timeout mechanism locally** - Reproduce the hang and test different timeout approaches
2. **Choose solution** - Likely Option 1 (process-level timeout) or Option 4 (skip temporarily)
3. **Deploy fix** - Either improve timeout or disable SkyPilot collector
4. **Monitor stability** - Ensure other collectors continue working

## Related Files

- `devops/datadog/scripts/run_all_collectors.py` - Main collection orchestrator
- `devops/datadog/collectors/skypilot/collector.py` - SkyPilot collector implementation
- `devops/charts/dashboard-cronjob/` - Kubernetes deployment

## Historical Context

- **Previous success**: Cronjobs were working 27-26 hours ago (last successful runs)
- **Failure pattern**: Started ~4 hours ago with consistent failures
- **Scope**: Affects both production and dev environments equally
- **GitHub metrics fix**: Unrelated - same issue occurs with old image (production) and new image (dev)

## Next Steps

- [ ] Reproduce the hang locally
- [ ] Test different timeout mechanisms
- [ ] Implement chosen solution
- [ ] Deploy to dev for testing
- [ ] Monitor for 24 hours
- [ ] Deploy to production
- [ ] Document findings and solution
