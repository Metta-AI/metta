# Code Review: Datadog Ingestion Implementation

**Reviewer**: AI Assistant
**Date**: 2025-01-17
**Status**: ⚠️ Needs fixes before merge

---

## ✅ What's Working Well

### 1. **Code Structure & Organization**
- ✅ Clean separation of concerns (collectors, client, models, CLI)
- ✅ Good use of abstract base classes (`BaseCollector`)
- ✅ Proper type annotations throughout (`from __future__ import annotations`)
- ✅ Registry pattern for collectors is clean and extensible

### 2. **Kubernetes Integration**
- ✅ Helm chart configuration looks correct
- ✅ Proper use of Datadog environment variables (`DD_ENV`, `DD_SERVICE`, `DD_VERSION`, `DD_SITE`)
- ✅ Good resource limits (2Gi memory, 500m CPU)
- ✅ Appropriate concurrency policy (`Forbid`)

### 3. **Error Handling**
- ✅ Good logging throughout
- ✅ Graceful handling of missing files/env vars
- ✅ Validation of record schemas

### 4. **CLI Design**
- ✅ Clean Typer-based CLI with `--dry-run` and `--push` flags
- ✅ Good separation between collection and submission

---

## ❌ Critical Issues (Must Fix)

### 1. **Metric Naming Mismatch** 🔴

**Problem**: The plan specifies `metta.infra.cron.ci.workflow.success` but your code emits `metta.ci.flaky_tests`, `metta.ci.workflow_duration.p90`, etc.

**Plan says** (from `DATADOG_INGESTION_PLAN.md`):
```
metta.infra.<pipeline>.{domain}.{signal}
Examples:
- metta.infra.cron.ci.workflow.success
- metta.infra.cron.github.reverts.count
```

**Your code emits** (from `ci_collector.py`):
- `metta.ci.flaky_tests`
- `metta.ci.workflow_duration.p90`
- `metta.ci.jobs_cancelled`
- `metta.ci.workflow_success_ratio`

**Fix Required**:
```python
# In ci_collector.py, change:
metric_namespace = "metta.ci"  # ❌ Wrong

# To:
metric_namespace = "metta.infra.cron"  # ✅ Correct
```

Then update metric names:
- `flaky_tests` → `ci.workflow.flaky_tests` (full: `metta.infra.cron.ci.workflow.flaky_tests`)
- `workflow_duration.p90` → `ci.workflow.duration.p90`
- `jobs_cancelled` → `ci.workflow.cancelled`
- `workflow_success_ratio` → `ci.workflow.success_ratio`

**Same issue in other collectors**:
- `TrainingHealthCollector`: `metta.training.*` → `metta.infra.stablesuite.training.*`
- `EvalHealthCollector`: `metta.eval.*` → `metta.infra.cron.eval.*` (or `metta.infra.stablesuite.eval.*`)

### 2. **Missing Required Tags** 🔴

**Problem**: The plan requires these tags on every metric:
- `env` (from Datadog config)
- `service` (should be `infra-health-dashboard`)

**Current state**: These tags are NOT being added in `BaseCollector._base_tags()`.

**Fix Required**:
```python
# In base.py, update _base_tags():
def _base_tags(self) -> Dict[str, str]:
    import os
    return {
        "source": self.source,
        "workflow_category": self.workflow_category,
        "service": "infra-health-dashboard",  # ✅ Add this
        "env": os.environ.get("DD_ENV", "production"),  # ✅ Add this
    }
```

### 3. **Missing Metric Type for Weekly Metrics** 🟡

**Problem**: In `ci_collector.py`, weekly metrics (hotfix, force_merge, revert) are correctly using `MetricKind.COUNT`, but the metric names don't match the plan.

**Plan says**: `metta.infra.cron.github.reverts.count`

**Your code**: `metta.ci.reverts.weekly` (wrong namespace + wrong suffix)

**Fix**: Update to match plan exactly:
- `reverts.weekly` → `github.reverts.count`
- `hotfix_commits.weekly` → `github.hotfix.count`
- `force_merges.weekly` → `github.force_merge.count`

---

## ⚠️ Important Issues (Should Fix)

### 4. **Dockerfile Dependencies** 🟡

**Current**:
```dockerfile
RUN pip install boto3 kubernetes pandas pyarrow typer datadog-api-client requests
```

**Issues**:
- No version pinning (could break on future releases)
- Missing `softmax` package (used in `datadog_client.py`: `from softmax.aws.secrets_manager`)
- Missing `metta.common` package (used in `datadog_client.py`: `from metta.common.datadog.config`)

**Fix**:
```dockerfile
# Option 1: Install from workspace (if devops/datadog becomes a package)
# Option 2: Copy and install dependencies properly
# Option 3: Use uv/pip with requirements.txt

# For now, you need to ensure the workspace is copied correctly
# and PYTHONPATH includes paths to metta.common and softmax
```

**Better approach**: Since you're copying the entire workspace (`.`), you might need to:
1. Install the workspace in editable mode: `pip install -e /app` (if there's a pyproject.toml at root)
2. Or ensure `PYTHONPATH` includes the right paths

### 5. **Secret Handling** 🟡

**Current**: `datadog_client.py` tries to get secrets from AWS Secrets Manager, but the CronJob might not have the right IAM permissions.

**Check**: Verify the service account has permissions to read:
- `datadog/api-key`
- `datadog/app-key`

**From `helmfile.yaml` comments**: Service accounts use `arn:aws:iam::751442549699:role/monitoring-cronjobs`. Verify this role can access these secrets.

### 6. **Metric Namespace Logic** 🟡

**Issue**: In `BaseCollector._metric_name()`, you check if suffix starts with `metta.` and return it as-is. But this could lead to inconsistent naming.

**Example**:
```python
# If someone passes "metta.infra.cron.ci.workflow.success"
# It returns as-is (good)

# But if they pass "ci.workflow.success"
# It becomes "metta.ci.workflow.success" (wrong - missing .infra.cron)
```

**Fix**: Either:
1. Enforce that collectors always use relative names (no `metta.` prefix)
2. Or update the logic to handle full names correctly

### 7. **Timestamp Handling** 🟡

**Issue**: In `ci_collector.py`, you're using `datetime.now(timezone.utc)` for all metrics, but for historical data (weekly metrics), you might want to use the actual event timestamps.

**Current**: All metrics get "now" as timestamp, even for weekly aggregates.

**Consideration**: This might be fine if you're just tracking "current state", but if you want historical accuracy, you'd need to track when events actually occurred.

---

## 💡 Suggestions (Nice to Have)

### 8. **Testing**

**Missing**: No unit tests for collectors or client.

**Suggestion**: Add tests for:
- Metric name generation
- Tag merging
- Record validation
- API client error handling

### 9. **Documentation**

**Good**: You have README.md and data/README.md.

**Suggestion**: Add docstrings to public methods, especially:
- `BaseCollector.collect()`
- `DatadogMetricsClient.submit()`
- `MetricSample.to_series()`

### 10. **Error Recovery**

**Current**: If GitHub API fails, the whole collection fails.

**Suggestion**: Consider partial success - if one collector fails, others can still succeed. Or at least log which part failed.

### 11. **Rate Limiting**

**Current**: No explicit rate limiting for GitHub API.

**Suggestion**: Add rate limit handling (GitHub API has 5000 requests/hour for authenticated users). Consider:
- Caching responses
- Using ETags
- Exponential backoff on 429 errors

### 12. **Metric Validation**

**Current**: You validate records in eval/training collectors, but not in CI collector.

**Suggestion**: Add validation to ensure all required tags are present before submission.

---

## 📋 Pre-Merge Checklist

Before merging, ensure:

- [ ] **Fix metric naming** to match `metta.infra.*` convention
- [ ] **Add missing tags** (`env`, `service`)
- [ ] **Fix Dockerfile dependencies** (ensure `softmax` and `metta.common` are available)
- [ ] **Test locally** with `--dry-run` flag
- [ ] **Verify secrets access** in Kubernetes (IAM role permissions)
- [ ] **Update metric names** in weekly metrics to match plan
- [ ] **Test end-to-end** in a dev environment before production

---

## 🎯 Summary

**Overall Assessment**: Good foundation, but needs fixes to match the plan specification.

**Strengths**:
- Clean architecture
- Good separation of concerns
- Proper error handling
- Well-structured Kubernetes config

**Critical Fixes Needed**:
1. Metric naming convention (must match plan)
2. Missing required tags
3. Dockerfile dependencies

**Estimated Fix Time**: 1-2 hours

---

## 🔍 Files to Review Closely

1. `devops/datadog/collectors/ci_collector.py` - Metric naming
2. `devops/datadog/collectors/base.py` - Tag generation
3. `devops/charts/cronjob/Dockerfile.dashboard` - Dependencies
4. `devops/datadog/datadog_client.py` - Secret access

