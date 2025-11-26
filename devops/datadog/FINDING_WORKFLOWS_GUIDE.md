# Finding the Right Workflows for Nishad's Plan

## ✅ What I Fixed

1. **Thresholds Fixed:**
   - Flakiness: Changed from `< 5` to `< 10` ✅
   - P90 Duration: Changed from `< 10 minutes` to `< 5 minutes` ✅

2. **Missing Checks Added:**
   - Tests that block merge passing ✅
   - Benchmarks passing ✅
   - Num other workflows failing ✅

3. **New CLI Command:**
   - `list-workflows` - Lists all workflows so we can identify which ones to monitor

---

## 🔍 Step 1: List All Workflows

Run this command to see all workflows in your repo:

```bash
export GITHUB_TOKEN="your-token"
uv run python -m devops.datadog.cli list-workflows
```

This will show you:
- Workflow ID
- Workflow Name
- Workflow Path (file path)
- State (active/inactive)

**Example output:**
```
Found 25 workflows:

ID         Name                                              Path                                                          State
----------------------------------------------------------------------------------------------------------------------------------
123456     CI Tests                                           .github/workflows/ci-tests.yml                               active
123457     Benchmarks                                         .github/workflows/benchmarks.yml                              active
123458     Integration Tests                                  .github/workflows/integration.yml                             active
...
```

---

## 🎯 Step 2: Identify Which Workflows to Monitor

Based on Nishad's plan, we need to identify:

### 1. **Tests that block merge passing**
Look for workflows that:
- Run tests that must pass before merging
- Are typically triggered on PRs
- Names might include: "test", "ci", "lint", "check", "verify"
- Common names: "CI", "Tests", "Pre-merge checks", "Blocking tests"

### 2. **Benchmarks passing**
Look for workflows that:
- Run performance benchmarks
- Names might include: "benchmark", "bench", "performance"
- Common names: "Benchmarks", "Performance tests", "Bench"

### 3. **Other workflows** (for counting failures)
These are all workflows EXCEPT:
- The "tests blocking merge" workflows
- The "benchmarks" workflows

---

## ⚙️ Step 3: Configure Workflow Names

Once you identify the workflows, set environment variables:

```bash
# For tests that block merge (comma-separated list of workflow names)
export CI_TESTS_BLOCKING_MERGE_WORKFLOWS="CI Tests,Pre-merge checks"

# For benchmarks (comma-separated list of workflow names)
export CI_BENCHMARKS_WORKFLOWS="Benchmarks,Performance tests"
```

**Or add to your `.env` file:**
```bash
CI_TESTS_BLOCKING_MERGE_WORKFLOWS="CI Tests,Pre-merge checks"
CI_BENCHMARKS_WORKFLOWS="Benchmarks"
```

---

## 🧪 Step 4: Test the Configuration

Test with dry-run to see the new metrics:

```bash
# Set your workflow names
export CI_TESTS_BLOCKING_MERGE_WORKFLOWS="your-workflow-name"
export CI_BENCHMARKS_WORKFLOWS="your-benchmark-name"

# Test
uv run python -m devops.datadog.cli collect ci --dry-run
```

You should now see these new metrics:
- `metta.infra.cron.ci.workflow.tests_blocking_merge` (value: 1.0 if passing, 0.0 if failing)
- `metta.infra.cron.ci.workflow.benchmarks` (value: 1.0 if passing, 0.0 if failing)
- `metta.infra.cron.ci.workflow.other_failing` (count of other failing workflows)

---

## 📋 Example: Finding Workflows Together

Let's work through this step by step:

1. **Run the list command:**
   ```bash
   uv run python -m devops.datadog.cli list-workflows
   ```

2. **Share the output** - I can help identify which ones are:
   - Tests blocking merge
   - Benchmarks
   - Other workflows

3. **Set the env vars** based on what we find

4. **Test** with `--dry-run`

---

## 🔧 For Kubernetes Deployment

Once you've identified the workflows, add them to your CronJob configuration:

```yaml
# In helmfile.yaml or values file
extraEnv:
  - name: CI_TESTS_BLOCKING_MERGE_WORKFLOWS
    value: "CI Tests,Pre-merge checks"
  - name: CI_BENCHMARKS_WORKFLOWS
    value: "Benchmarks"
```

---

## 💡 Tips

- **Workflow names** are case-sensitive - use exact names from the list
- **Multiple workflows** can be comma-separated
- **If a workflow isn't found**, the collector will log a warning and skip it
- **The "other workflows" check** automatically excludes your test/benchmark workflows

---

## 🚀 Ready to Start?

Run this command and share the output:

```bash
uv run python -m devops.datadog.cli list-workflows
```

Then we can identify which workflows to monitor together! 🎯

