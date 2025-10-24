# Datadog Metrics Push Testing Plan

## Problem

GitHub collector claims to push metrics successfully, but metrics don't appear in Datadog.
Need systematic testing to understand where the issue is.

## Testing Strategy: Ratchet Confidence

### Phase 1: Basic API Connectivity ✓ (Expected to work)
**Goal**: Verify we can connect to Datadog API and have correct credentials.

**Test**: List existing metrics
- **Script**: `scripts/list_datadog_metrics.py --minutes 60`
- **Expected**: Should see existing metrics (wandb.*, asana.*, etc.)
- **Status**: ✓ Already verified - 169 metrics found

### Phase 2: Simple Test Metric (Sanity Check)
**Goal**: Verify we can push a simple metric and retrieve it.

**Test**: Push a test metric, wait, query it back
- **Script**: `scripts/test_metric_push.py`
- **Steps**:
  1. Push `test.sanity_check` with value 1.0
  2. Wait 60 seconds
  3. Query `test.sanity_check` back
  4. Verify value matches
- **Expected**: Should successfully push and retrieve
- **Status**: TODO

### Phase 3: Test GitHub Metric Format
**Goal**: Verify GitHub metrics can be pushed with same format as test metric.

**Test**: Push one GitHub metric with explicit format
- **Script**: `scripts/test_github_metric.py`
- **Steps**:
  1. Push `github.test.metric` with value 42.0
  2. Wait 60 seconds
  3. Query back
- **Expected**: Should successfully push and retrieve
- **Status**: TODO

### Phase 4: Full GitHub Collector
**Goal**: Verify full collector works if individual metrics work.

**Test**: Run GitHub collector with push
- **Script**: `scripts/run_collector.py github --push --verbose`
- **Steps**:
  1. Collect metrics (already works)
  2. Push to Datadog
  3. Wait 60 seconds
  4. List GitHub metrics
- **Expected**: All 28 metrics should appear
- **Status**: CURRENTLY FAILING

## Helper Scripts to Create

### 1. `scripts/list_datadog_metrics.py`
**Purpose**: List all metrics from Datadog in last N minutes

**Usage**:
```bash
# List all metrics from last 60 minutes
uv run python devops/datadog/scripts/list_datadog_metrics.py --minutes 60

# Filter by prefix
uv run python devops/datadog/scripts/list_datadog_metrics.py --minutes 60 --prefix github

# Show summary counts
uv run python devops/datadog/scripts/list_datadog_metrics.py --minutes 60 --summary
```

**Output**:
```
=== Datadog Metrics (last 60 minutes) ===
Found 169 metrics

By prefix:
  asana.*: 14 metrics
  ec2.*: 20 metrics
  wandb.*: 23 metrics
  github.*: 0 metrics ⚠️
  health.*: 0 metrics

Total: 169 metrics
```

### 2. `scripts/test_metric_push.py`
**Purpose**: Push a test metric and verify it appears in Datadog

**Usage**:
```bash
# Push test metric and verify
uv run python devops/datadog/scripts/test_metric_push.py

# Custom metric name and value
uv run python devops/datadog/scripts/test_metric_push.py --metric test.custom --value 123.45

# Skip wait (for manual verification)
uv run python devops/datadog/scripts/test_metric_push.py --no-wait
```

**Output**:
```
Pushing metric: test.sanity_check = 1.0
✅ Successfully submitted to Datadog

Waiting 60 seconds for propagation...
Done.

Querying metric back...
✅ Found: test.sanity_check = 1.0 (age: 62s)

✅ TEST PASSED: Metric successfully pushed and retrieved
```

### 3. `scripts/test_github_metric.py`
**Purpose**: Test pushing GitHub-formatted metrics

**Usage**:
```bash
# Test single GitHub metric
uv run python devops/datadog/scripts/test_github_metric.py

# Test with specific value
uv run python devops/datadog/scripts/test_github_metric.py --value 42
```

### 4. `scripts/verify_collector_push.py`
**Purpose**: Run collector, push, and verify metrics appear

**Usage**:
```bash
# Test GitHub collector end-to-end
uv run python devops/datadog/scripts/verify_collector_push.py github

# Test any collector
uv run python devops/datadog/scripts/verify_collector_push.py wandb
```

**Output**:
```
=== Testing github collector ===

Step 1: Collecting metrics...
✅ Collected 28 metrics

Step 2: Pushing to Datadog...
✅ Successfully pushed 28 metrics

Step 3: Waiting 60 seconds...
Done.

Step 4: Verifying metrics in Datadog...
Checking for github.* metrics...
❌ Found 0/28 metrics

FAILED: Metrics were pushed but don't appear in Datadog
```

## Decision Tree

```
Start
  ↓
Can we list existing metrics? (Phase 1)
  ├─ NO → Fix: Check API keys, connectivity
  └─ YES ✓
       ↓
Can we push test.sanity_check? (Phase 2)
  ├─ NO → Fix: API submission broken, check permissions
  └─ YES
       ↓
Can we push github.test.metric? (Phase 3)
  ├─ NO → Fix: Metric naming issue (github.* prefix blocked?)
  └─ YES
       ↓
Can we push real GitHub metrics? (Phase 4)
  ├─ NO → Fix: Collector-specific issue
  └─ YES → SUCCESS!
```

## Implementation Order

1. **Create `list_datadog_metrics.py`** - Useful for all phases
2. **Create `test_metric_push.py`** - Phase 2 test
3. **Run Phase 2** - Build confidence in basic push
4. **Create `test_github_metric.py`** - Phase 3 test
5. **Run Phase 3** - Verify GitHub namespace works
6. **Create `verify_collector_push.py`** - Phase 4 test
7. **Run Phase 4** - End-to-end verification

## Current Hypothesis

**Most Likely**: Metric naming or tag issue
- The API accepts the submission (returns success)
- But metrics are silently dropped or rejected
- Possible issues:
  - Metric names too long or invalid characters
  - Missing required tags
  - Wrong metric type for the name pattern
  - Datadog organization settings blocking custom metrics

**Less Likely**: API submission not actually working
- We see success messages
- Other collectors (wandb, asana) work fine
- Same DatadogClient code

## Notes

- Datadog metrics can take 1-5 minutes to propagate and be queryable
- The list_active_metrics API may have longer delay than query API
- We should test both APIs (list and query) to verify propagation
