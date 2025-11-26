# How to Send Data to Datadog

This guide walks you through testing and deploying the Datadog metric collectors.

---

## 🧪 Local Testing (Before Deployment)

### Step 1: Set Up Environment Variables

Create a `.env` file or export these variables:

```bash
# Required for Datadog API
export DD_API_KEY="your-datadog-api-key"
export DD_APP_KEY="your-datadog-app-key"  # Optional but recommended
export DD_SITE="datadoghq.com"  # or "datadoghq.eu" for EU
export DD_ENV="development"  # or "production", "staging"

# Required for CI collector
export GITHUB_TOKEN="your-github-token"  # or GITHUB_DASHBOARD_TOKEN
export METTA_GITHUB_REPO="Metta-AI/metta"  # Optional, defaults to this
export METTA_GITHUB_BRANCH="main"  # Optional, defaults to this

# Optional: For training/eval collectors
export TRAINING_HEALTH_FILE="/path/to/training_health.json"
export EVAL_HEALTH_FILE="/path/to/eval_health.json"
```

**Getting API Keys:**
1. Go to Datadog → Organization Settings → API Keys
2. Create a new API key (or use existing)
3. For App Key: Organization Settings → Application Keys

**Getting GitHub Token:**
1. GitHub → Settings → Developer settings → Personal access tokens
2. Create token with `repo` scope (for higher rate limits)

### Step 2: Test with Dry Run (No Data Sent)

Test that collectors work without sending data:

```bash
# Test CI collector
uv run python -m devops.datadog.cli collect ci --dry-run

# Test training collector (requires TRAINING_HEALTH_FILE)
uv run python -m devops.datadog.cli collect training --dry-run

# Test eval collector (requires EVAL_HEALTH_FILE)
uv run python -m devops.datadog.cli collect eval --dry-run
```

**Expected Output:**
```json
[
  {
    "metric": "metta.infra.cron.ci.workflow.flaky_tests",
    "value": 2,
    "tags": {
      "source": "cron",
      "workflow_category": "ci",
      "workflow_name": "latest_state_of_main",
      "task": "tests_blocking_merge",
      "check": "flaky_tests",
      "condition": "< 5",
      "status": "pass",
      "service": "infra-health-dashboard",
      "env": "development"
    },
    "kind": "count",
    "timestamp": "2025-01-17T10:00:00Z"
  },
  ...
]
```

### Step 3: Test with Actual Push (Sends to Datadog)

**⚠️ Warning**: This will send real data to Datadog. Use a dev/test environment first.

```bash
# Push CI metrics
uv run python -m devops.datadog.cli collect ci --push

# Push training metrics
uv run python -m devops.datadog.cli collect training --push

# Push eval metrics
uv run python -m devops.datadog.cli collect eval --push
```

**Expected Output:**
```
[collector] Running 'ci' collector...
[collector] Submitting metrics to Datadog...
[collector] Submission complete.
```

### Step 4: Verify Data in Datadog

1. **Go to Datadog Metrics Explorer:**
   - Navigate to: Datadog → Metrics → Explorer
   - Search for: `metta.infra.cron.ci.workflow.*`
   - You should see metrics like:
     - `metta.infra.cron.ci.workflow.flaky_tests`
     - `metta.infra.cron.ci.workflow.duration.p90`
     - `metta.infra.cron.ci.workflow.success`
     - `metta.infra.cron.github.reverts.count`

2. **Check Tags:**
   - Click on a metric
   - Verify tags include: `source`, `workflow_category`, `workflow_name`, `task`, `check`, `condition`, `status`, `service`, `env`

3. **View Time Series:**
   - Select a time range (last 1 hour)
   - You should see data points

---

## 🚀 Kubernetes Deployment

### Step 1: Build and Push Docker Image

The GitHub Actions workflow should handle this automatically, but for manual testing:

```bash
# Build the image
cd devops/charts/cronjob
docker build -f Dockerfile.dashboard -t metta-cronjobs:test-dashboard ../../..

# Tag for ECR
docker tag metta-cronjobs:test-dashboard \
  751442549699.dkr.ecr.us-east-1.amazonaws.com/metta-cronjobs:test-dashboard

# Push to ECR (requires AWS credentials)
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  751442549699.dkr.ecr.us-east-1.amazonaws.com
docker push 751442549699.dkr.ecr.us-east-1.amazonaws.com/metta-cronjobs:test-dashboard
```

### Step 2: Deploy CronJob

```bash
# Deploy the dashboard cronjob
cd devops/charts
helmfile apply -l name=dashboard-cronjob

# Or deploy all cronjobs
helmfile apply -l name=dashboard-cronjob,training-health-cronjob,eval-health-cronjob
```

### Step 3: Verify CronJob is Running

```bash
# Check CronJob exists
kubectl get cronjobs -n monitoring

# Check recent jobs
kubectl get jobs -n monitoring | grep dashboard

# View logs from latest run
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=100

# Or trigger a manual run
kubectl create job --from=cronjob/dashboard-cronjob-cronjob manual-test -n monitoring
kubectl logs -n monitoring job/manual-test
```

### Step 4: Verify Secrets Access

The CronJob needs access to Datadog API keys from AWS Secrets Manager.

**Check IAM Role:**
```bash
# The service account should have this IAM role annotation:
kubectl get serviceaccount -n monitoring dashboard-cronjob-cronjob -o yaml

# Verify the role has permissions:
aws iam get-role-policy \
  --role-name monitoring-cronjobs \
  --policy-name SecretsManagerRead
```

**Required Permissions:**
- `secretsmanager:GetSecretValue` for `datadog/api-key`
- `secretsmanager:GetSecretValue` for `datadog/app-key`

**Test Secret Access (from a pod):**
```bash
# Create a test pod
kubectl run -it --rm test-secrets -n monitoring \
  --image=amazon/aws-cli:latest \
  --restart=Never \
  --overrides='{
    "spec": {
      "serviceAccountName": "dashboard-cronjob-cronjob"
    }
  }' \
  -- aws secretsmanager get-secret-value --secret-id datadog/api-key
```

---

## 📊 How Data Flows to Datadog

```
┌─────────────────┐
│  Kubernetes     │
│  CronJob        │
│  (every 10 min) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Python CLI     │
│  collect ci     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CICollector    │
│  - GitHub API   │
│  - Calculate    │
│    metrics      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MetricSample   │
│  objects        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DatadogClient  │
│  .submit()      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Datadog API    │
│  Metrics API v2 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Datadog        │
│  Platform       │
│  (Metrics DB)   │
└─────────────────┘
```

### Code Flow

1. **CronJob triggers** → Runs `python -m devops.datadog.cli collect ci --push`

2. **CLI collects metrics** → `CICollector.collect()`:
   - Calls GitHub API
   - Calculates metrics (flaky tests, durations, etc.)
   - Returns `List[MetricSample]`

3. **CLI submits** → `DatadogMetricsClient.submit()`:
   - Converts `MetricSample` → `MetricSeries` (Datadog format)
   - Builds `MetricPayload`
   - Calls `MetricsApi.submit_metrics()`

4. **Datadog API** → Stores metrics with tags

5. **Dashboard queries** → Datadog queries metrics by name/tags

---

## 🔍 Troubleshooting

### Issue: "Missing Datadog API key"

**Error:**
```
RuntimeError: Missing Datadog API key. Set DD_API_KEY or DATADOG_API_KEY.
```

**Fix:**
- Check environment variable is set: `echo $DD_API_KEY`
- For Kubernetes: Verify secret exists in AWS Secrets Manager
- For Kubernetes: Verify IAM role has `secretsmanager:GetSecretValue` permission

### Issue: "No metrics to submit"

**Error:**
```
[collector] No metrics to submit.
```

**Possible Causes:**
1. Collector found no data (e.g., no GitHub workflow runs)
2. File not found (for training/eval collectors)
3. All records failed validation

**Fix:**
- Check logs for warnings: `kubectl logs -n monitoring job/<job-name>`
- Verify GitHub token has access to repo
- For training/eval: Check `TRAINING_HEALTH_FILE` / `EVAL_HEALTH_FILE` paths

### Issue: "GitHub API rate limit"

**Error:**
```
GitHub API error (403): API rate limit exceeded
```

**Fix:**
- Use `GITHUB_DASHBOARD_TOKEN` instead of `GITHUB_TOKEN` (higher rate limits)
- Add token to AWS Secrets Manager and reference in CronJob env vars

### Issue: "Metrics not appearing in Datadog"

**Checklist:**
1. ✅ Verify API key is correct
2. ✅ Check `DD_SITE` matches your Datadog region
3. ✅ Wait 1-2 minutes (metrics may take time to appear)
4. ✅ Check metric names match exactly (case-sensitive)
5. ✅ Verify tags are correct (use Metrics Explorer to filter)

**Debug:**
```bash
# Check what was actually sent
uv run python -m devops.datadog.cli collect ci --dry-run --output /tmp/metrics.json
cat /tmp/metrics.json | jq '.[0]'

# Check Datadog API response
# (Add logging to datadog_client.py to see API responses)
```

### Issue: "Import errors in Docker"

**Error:**
```
ModuleNotFoundError: No module named 'softmax'
ModuleNotFoundError: No module named 'metta.common'
```

**Fix:**
- Ensure Dockerfile installs workspace packages correctly
- Check `PYTHONPATH` includes `/app`, `/app/softmax/src`, `/app/common/src`
- Verify `COPY . .` copies the entire workspace

---

## 📝 Example: Complete Local Test

```bash
# 1. Set up environment
export DD_API_KEY="your-key"
export DD_APP_KEY="your-app-key"
export GITHUB_TOKEN="your-token"
export DD_ENV="development"

# 2. Test dry run
uv run python -m devops.datadog.cli collect ci --dry-run

# 3. If dry run looks good, push
uv run python -m devops.datadog.cli collect ci --push

# 4. Verify in Datadog
# Go to: https://app.datadoghq.com/metric/explorer
# Search: metta.infra.cron.ci.workflow.*
```

---

## 🎯 Next Steps

After data is flowing:

1. **Build Dashboard** → Create Screenboard in Datadog UI
2. **Set Up TV Mode** → Enable TV/presentation mode
3. **Add Alerts** → Set up monitors for critical metrics
4. **Document** → Update runbooks with dashboard links

See `TV_DASHBOARD_IMPLEMENTATION_GUIDE.md` for dashboard setup.

---

## 📚 Reference

- **Datadog Metrics API**: https://docs.datadoghq.com/api/latest/metrics/
- **Metric Naming**: See `DATADOG_INGESTION_PLAN.md` section 2
- **Tag Schema**: See `DATADOG_INGESTION_PLAN.md` section 2
- **CLI Usage**: `uv run python -m devops.datadog.cli --help`

