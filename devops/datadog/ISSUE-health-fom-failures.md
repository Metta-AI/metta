# Issue: Health FoM Collector Causing CronJob Failures

## Status
**ACTIVE**: CronJobs failing since ~2025-10-24 20:00 UTC (2.5 hours before investigation)

## Problem
Dashboard collector CronJobs are hitting backoff limits and failing after health_fom collector runs. The pod crashes/restarts 3 times before giving up.

**Error Pattern**:
- 6 out of 7 collectors complete successfully (github, kubernetes, ec2, skypilot, wandb, asana)
- health_fom collector starts running
- Pod crashes or times out
- Kubernetes restarts pod (backoffLimit: 3)
- After 3 failures, job marked as Failed

**Impact**:
- Production CronJob failing for last ~2.5 hours
- Dev CronJob also failing with same symptoms
- Metrics ARE being collected from the 6 successful collectors before crash
- No health FoM metrics being generated

## Timeline

**Working Period** (>24 hours ago):
- Production jobs completing successfully
- Example: `dashboard-cronjob-dashboard-cronjob-29354265` - Completed 25h ago

**Failure Period** (started ~2.5 hours ago):
- Production jobs failing: `29355630`, `29355645`, `29355660`
- Dev test jobs failing: `manual-wandb-test-1761346413`, `test-dev-correct-1761346203`
- Currently running job: `29355690` (likely to fail)

## Observed Behavior

### From Live Logs (manual-wandb-test-1761346413)

**Successful Collectors:**
```
✅ github: 28 metrics collected in 10.97s
✅ kubernetes: 15 metrics collected in 1.34s
✅ ec2: 20 metrics collected in 1.51s
✅ skypilot: 30 metrics collected in 29.54s
✅ wandb: 20 metrics collected in 5.14s
✅ asana: 14 metrics collected in 0.37s
```

**Last Activity Before Crash:**
```
Running health_fom collector...
Collecting Health FoM metrics...
[logs stop here - pod crashes]
```

### Local Test Results

Running health_fom collector locally:
```bash
uv run python devops/datadog/scripts/run_collector.py health_fom --verbose
```

**Output:**
- Multiple warnings: "No data returned for metric X"
- Missing metrics queried:
  - `github.ci.tests_passing_on_main`
  - `github.ci.failed_workflows_7d`
  - `github.commits.hotfix`
  - `github.commits.reverts`
  - `github.ci.duration_p90_minutes`
  - `github.prs.stale_count_14d`
  - `github.prs.cycle_time_hours`
  - `wandb.runs.completed_7d` (old metric, replaced with 24h version)
  - `wandb.runs.failed_7d` (old metric, replaced with 24h version)
  - `wandb.metrics.best_accuracy`
  - `wandb.metrics.avg_accuracy_7d`
  - `wandb.metrics.latest_loss`
  - `wandb.training.gpu_utilization_avg`
  - `wandb.training.avg_duration_hours` (replaced with new metric name)

**Result:** `Collected 0 metrics from health_fom`

**Exit Code:** 0 (success) - completes without errors locally

## Root Cause Hypotheses

### 1. Memory/Resource Exhaustion (Most Likely)
- health_fom collector queries Datadog API for 14 historical metrics
- Each query fetches timeseries data from Datadog
- Pod has 2Gi memory limit
- Multiple API calls might be accumulating data
- Possible OOM (Out of Memory) kill by Kubernetes

### 2. Datadog API Timeout/Rate Limiting
- health_fom makes many sequential API queries
- Datadog API might be slow or timing out
- Could exceed 120s collector timeout
- But timeout should be caught and handled gracefully

### 3. Network/API Issues
- Transient network issues to Datadog API
- Started ~2.5 hours ago
- Could be Datadog service degradation

### 4. Code Issue in health_fom Collector
- Recent change to health_fom collector
- Memory leak in query loop
- Uncaught exception not visible in logs

## Evidence

**Why it's NOT a WandB collector issue:**
- Production using old image: `sha-12bdc0ab70b8ed4f742369e1a064f961efc1b102`
- Dev using new image: `sha-f2c2e44` (with WandB improvements)
- Both failing with same symptoms
- WandB collector completes successfully in both environments
- Failures started before WandB changes were deployed to production

**Why health_fom is suspicious:**
- Always the last collector running before crash
- Queries 14+ historical metrics from Datadog
- Collector has no data to return (all queries return None)
- Works locally but fails in Kubernetes

## Configuration

**Pod Resources:**
```yaml
resources:
  requests:
    cpu: 100m
    memory: 2Gi
  limits:
    cpu: 500m
    memory: 2Gi
```

**Timeouts:**
- Per-collector timeout: 120s (2 minutes)
- Job backoffLimit: 3
- Job will fail after 3 pod restarts

**health_fom Collector Behavior:**
- Queries Datadog API for historical metric values
- Uses DatadogClient.query_metric() for each metric
- Computes normalized 0.0-1.0 FoM values
- Returns empty dict if all queries return None

## Investigation Steps

### 1. Check Pod Events
```bash
kubectl describe job <job-name> -n monitoring
# Look for OOMKilled, Error, or other failure reasons
```

### 2. Monitor Resource Usage
```bash
# Watch a running job
kubectl top pods -n monitoring | grep dashboard-cronjob

# Check if memory approaching 2Gi limit
```

### 3. Check Datadog API Status
- Visit https://status.datadoghq.com/
- Check for service degradations
- Look for API rate limiting

### 4. Review health_fom Code Changes
```bash
# Check recent commits to health_fom collector
git log --oneline --since="3 days ago" -- devops/datadog/collectors/health_fom/
```

### 5. Add Debug Logging
Modify health_fom collector to:
- Log memory usage before/after each query
- Log query response sizes
- Add timing information
- Catch and log ALL exceptions

### 6. Test With Reduced Metrics
Temporarily modify health_fom to query only 1-2 metrics instead of 14:
- See if it completes successfully
- Identify if it's a volume/memory issue

## Potential Fixes

### Option 1: Increase Memory Limit (Quick Fix)
```yaml
# values.yaml
resources:
  limits:
    memory: 4Gi  # Double current limit
```

**Pros:** Fast, might solve OOM issue
**Cons:** Doesn't address root cause, wastes resources

### Option 2: Disable health_fom Collector (Temporary)
```python
# run_all_collectors.py
COLLECTORS = [
    "github",
    "kubernetes",
    "ec2",
    "skypilot",
    "wandb",
    "asana",
    # "health_fom",  # Temporarily disabled - see ISSUE-health-fom-failures.md
]
```

**Pros:** Unblocks other collectors, fast
**Cons:** Loses health FoM metrics

### Option 3: Optimize health_fom Queries (Proper Fix)
- Reduce time range queried (1 hour instead of default)
- Batch API requests
- Cache results
- Stream results instead of accumulating
- Skip metrics that consistently return None

### Option 4: Make health_fom Optional/Non-Blocking
- Run health_fom in separate cronjob
- Don't block main collector job on health_fom success
- Accept that health_fom might collect 0 metrics

## Workaround

**For immediate production fix**, comment out health_fom in run_all_collectors.py:

```python
COLLECTORS = [
    "github",
    "kubernetes",
    "ec2",
    "skypilot",
    "wandb",
    "asana",
    # "health_fom",  # Temporarily disabled - causing pod crashes
]
```

This allows the other 6 collectors to run successfully every 15 minutes while health_fom issue is investigated.

## Related Files

**health_fom Collector:**
- `devops/datadog/collectors/health_fom/collector.py` - Main implementation
- Queries: `_ci_foms()` (7 metrics) + `_training_foms()` (7 metrics)

**Runner Script:**
- `devops/datadog/scripts/run_all_collectors.py` - Orchestrator with timeout handling

**Datadog Client:**
- `devops/datadog/utils/datadog_client.py` - API query implementation
- `DatadogClient.query_metric()` method

**Configuration:**
- `devops/charts/dashboard-cronjob/values.yaml` - Resource limits
- `devops/charts/dashboard-cronjob/values-dev.yaml` - Dev overrides

## Success Criteria

- [ ] Identify root cause (OOM, timeout, API issue, code bug)
- [ ] Fix implemented and tested locally
- [ ] Docker image built with fix
- [ ] Dev cronjob running successfully for 1+ hour (4+ cycles)
- [ ] Production cronjob running successfully for 1+ hour
- [ ] health_fom metrics appearing in Datadog (or confirmed not needed)
- [ ] No pod crashes or restarts

## Priority

**Medium-High** - Production monitoring partially working but not critical:
- 6 out of 7 collectors working (85% success rate)
- Main metrics (github, wandb, ec2, kubernetes) being collected
- Only health FoM metrics missing
- Can temporarily disable health_fom as workaround

## Notes

- This is NOT related to WandB collector improvements (separate codepath)
- health_fom collector is attempting to query deprecated WandB metrics
- May need to update health_fom to use new WandB metric names
- Consider if health FoM metrics are actually needed/useful

## Next Steps

1. **Immediate**: Check production pod events for OOMKilled or other errors
2. **Short-term**: Consider disabling health_fom temporarily to unblock other collectors
3. **Long-term**: Investigate and fix health_fom collector (likely memory or query optimization needed)
