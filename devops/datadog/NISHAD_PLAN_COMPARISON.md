# Comparison: Implementation vs Nishad's Plan

## ✅ What We Have (Matches Plan)

### Commit History - All Good ✅
- ✅ Weekly num hotfix commits (< 5) → `metta.infra.cron.github.hotfix.count`
- ✅ Weekly num force merges (< 7) → `metta.infra.cron.github.force_merge.count`
- ✅ Weekly num reverts (< 1) → `metta.infra.cron.github.reverts.count`

---

## ⚠️ What's Missing

### CI - Latest state of main (Missing 3 checks)

**Nishad wants:**
1. ❌ **Tests that block merge passing** (> 0)
2. ❌ **Benchmarks passing** (> 0)
3. ❌ **Num other workflows whose latest run off main is failing** (< 2)

**What we have:**
- Only a generic `workflow.success` metric (success ratio)

**What we need:**
- Specific workflow checks for "tests blocking merge"
- Specific workflow checks for "benchmarks"
- Count of other workflows that are failing

---

## 🔧 What Needs Fixing (Wrong Thresholds)

### CI Smoothness

1. **P90 pre-merge CI checks duration**
   - **Nishad wants:** `< 5 minutes`
   - **We have:** `< 10 minutes` (default from env var `CI_P90_DURATION_MINUTES=10`)
   - **Fix:** Change default to 5, or set env var to 5

2. **Flakiness (check failed then succeeded on retry)**
   - **Nishad wants:** `< 10`
   - **We have:** `< 5` (default from env var `CI_FLAKY_TEST_THRESHOLD=5`)
   - **Fix:** Change default to 10, or set env var to 10

3. **Jobs cancelled due to timeout**
   - **Nishad wants:** Weekly count of jobs cancelled due to timeout
   - **We have:** All cancelled jobs (not just timeout)
   - **Note:** GitHub API doesn't distinguish timeout vs other cancellations easily
   - **Possible fix:** Check if cancelled jobs exceeded a duration threshold

---

## 📊 Current Implementation Status

| Nishad's Check | Status | Metric Name | Issue |
|----------------|--------|-------------|-------|
| Tests that block merge passing | ❌ Missing | - | Need specific workflow check |
| Benchmarks passing | ❌ Missing | - | Need specific workflow check |
| Num other workflows failing | ❌ Missing | - | Need to count failing workflows |
| Weekly hotfix commits | ✅ Good | `github.hotfix.count` | - |
| Weekly force merges | ✅ Good | `github.force_merge.count` | - |
| Weekly reverts | ✅ Good | `github.reverts.count` | - |
| P90 duration | ⚠️ Wrong threshold | `ci.workflow.duration.p90` | Should be < 5, not < 10 |
| Jobs cancelled (timeout) | ⚠️ Not specific | `ci.workflow.cancelled` | Counts all cancelled, not just timeout |
| Flakiness | ⚠️ Wrong threshold | `ci.workflow.flaky_tests` | Should be < 10, not < 5 |

---

## 🎯 Action Items

### High Priority (Missing Checks)
1. **Add "Tests that block merge passing" check**
   - Identify which workflow(s) represent "tests blocking merge"
   - Check if latest run is passing (> 0 means passing)

2. **Add "Benchmarks passing" check**
   - Identify which workflow(s) represent "benchmarks"
   - Check if latest run is passing (> 0 means passing)

3. **Add "Num other workflows failing" check**
   - Count workflows (excluding tests/benchmarks) whose latest run is failing
   - Condition: < 2

### Medium Priority (Fix Thresholds)
4. **Fix P90 duration threshold:** Change from 10 to 5 minutes
5. **Fix flakiness threshold:** Change from 5 to 10
6. **Improve cancelled jobs:** Try to distinguish timeout cancellations (optional)

---

## 💡 Implementation Notes

### For "Tests that block merge passing" and "Benchmarks passing"

You'll need to:
1. Identify workflow names/IDs for these specific checks
2. Query GitHub API for latest run of each workflow
3. Emit metrics with:
   - `workflow_name: "tests_blocking_merge"` or `"benchmarks"`
   - `check: "workflow_passing"`
   - `value: 1.0` if passing, `0.0` if failing
   - `condition: "> 0"`

### For "Num other workflows failing"

You'll need to:
1. Get list of all workflows
2. Exclude "tests blocking merge" and "benchmarks"
3. Check latest run status for each
4. Count how many are failing
5. Emit metric:
   - `workflow_name: "other_workflows"`
   - `check: "failing_count"`
   - `value: <count>`
   - `condition: "< 2"`

---

## 🔍 Current Default Thresholds (from code)

```python
flaky_threshold = 5  # Should be 10
duration_p90_minutes = 10.0  # Should be 5.0
cancelled_threshold = 10  # This is OK
```

These can be overridden with env vars, but defaults should match Nishad's plan.

