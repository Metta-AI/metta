# Metta Infra Health – Datadog Ingestion Plan

Author: Akshay Kalapgar  
Stakeholders: Nishad Singh, Infra Eng

This document covers Step 1 of Nishad's rollout plan: choosing the right Datadog ingestion surfaces (metrics vs events vs logs vs traces), defining schemas/tags, and describing how the new cron + stable-suite pipelines will push data into Datadog.

---

## 1. Datadog Surface Selection

| Data Type | Examples | Datadog Surface | Rationale |
| --- | --- | --- | --- |
| Real-time pass/fail health checks (CI green, eval queue healthy, workflow-specific guardrails) | GitHub workflow success, `./tools/request_eval` latency buckets, flaky counts | **Metrics (GAUGE + COUNT)** | Dashboards need time-series tiles + heatmap-style matrix. Metrics give us rollups, formulas, anomaly detection, SLO widgets. |
| Rate/volume data (PR throughput, revert counts, bug backlog deltas) | weekly hotfix count, bug queue by label | **Metrics (COUNT/DISTRIBUTION)** | We need aggregations (daily/weekly) and tagging. Distribution metrics let us query percentiles (e.g., workflow duration p90). |
| Slow-moving "stable suite" results (reward thresholds, SPS) | multi-node reward > 0.5, SPS target | **Metrics (GAUGE)** | Same visualization story as above; we can annotate tiles with `status:pass/fail` tags and still look at raw SPS numbers. |
| Qualitative incident notes / explanations | "Leaderboard down due to deploy" | **Events** | Optional future work. Events stream can annotate dashboards but should not drive health coloring. |
| Stack traces / request timings | Already handled by APM elsewhere | **Traces/Logs (existing)** | Out of scope for this dashboard; we only reference them via links. |

=> Conclusion: **Metrics are the primary primitive.** Events/logs are additive only when we want free-form text or to link to deeper debugging context.

---

## 2. Metric Naming & Schema

### Global naming convention
```
metta.infra.<pipeline>.{domain}.{signal}

Examples:
- metta.infra.cron.ci.workflow.success
- metta.infra.cron.github.reverts.count
- metta.infra.cron.eval.latency.p90
- metta.infra.stablesuite.training.reward
- metta.infra.stablesuite.training.sps
```

### Required tags on every metric

| Tag | Description |
| --- | --- |
| `source` | `cron` or `stable_suite` (or future emitters) |
| `workflow_category` | e.g., `ci`, `training`, `evaluation`, `leaderboard`, `benchmarks` |
| `workflow_name` | Friendly identifier (e.g., `arena_multi_gpu` or GitHub workflow name) |
| `task` | Sub-task label (maps to mockup row names: "Hearts", "Leaderboard Refresh", etc.) |
| `check` | Name of the condition being evaluated (e.g., `reward_threshold`, `latency_p90`) |
| `condition` | Human-readable success criterion, e.g., `>0.5`, `<10m`, `==success` |
| `status` | `pass`, `fail`, or `unknown` (used to color dashboard tiles) |
| `env` | `prod`, `staging`, etc. (propagated from Datadog config) |
| `service` | `infra-health-dashboard` (shared service tag for discoverability) |

Optional tags when available:
- `repo`, `branch`, `commit_sha`
- `runner` (e.g., `skypilot`, `gha`, `onprem`)
- `region` / `zone`

### Data encoding

- **Boolean/health checks:** Emit GAUGE (1 = pass, 0 = fail). Still attach the `status` tag, but the value lets us drive timeseries + conditional formatting.
- **Counts (reverts, hotfixes, bug backlog):** Use COUNT type. Aggregations default to sum per time bucket.
- **Durations/latency:** Use DISTRIBUTION metrics so Datadog can compute p50/p90 without pre-aggregation. If API limits force us to stay on GAUGE, emit explicit `p50/p90` gauges per interval.
- **Stable suite numeric outputs (reward, SPS, throughput):** GAUGE with `unit` tag (e.g., `steps_per_sec`).

---

## 3. Payload & API Strategy

### Client Library
- Use `datadog-api-client` v2 (already in repo via `uv.lock`).
- Submit metrics via `MetricsApi.submit_metrics`.

### Authentication
- Reuse existing secrets in AWS Secrets Manager (`datadog/api-key`, `datadog/app-key`).
- Helm chart injects them as env vars / projected secrets to the cron pod.

### Payload shape (example)

```python
MetricPayload(
  series=[
    MetricSeries(
      metric="metta.infra.cron.ci.workflow.success",
      type=MetricIntakeType.GAUGE,
      points=[MetricPoint(timestamp=ts, value=1.0)],
      tags=[
        "source:cron",
        "workflow_category:ci",
        "workflow_name:integration",
        "task:tests_blocking_merge",
        "check:workflow_success",
        "condition:==success",
        "status:pass",
        "service:infra-health-dashboard",
        "env:prod",
      ],
    ),
    MetricSeries(
      metric="metta.infra.cron.github.reverts.count",
      type=MetricIntakeType.COUNT,
      points=[MetricPoint(timestamp=ts, value=2.0)],
      tags=[
        "source:cron",
        "workflow_category:code",
        "workflow_name:weekly_summary",
        "task:reverts",
        "check:weekly_count",
        "condition:<=1",
        "status:fail",
        "service:infra-health-dashboard",
        "env:prod",
      ],
    ),
  ]
)
```

### Carry-forward logic
The spreadsheet mockup expects "light green = inferred". We'll implement this at query layer:
- Cron job emits raw datapoints only when it runs.
- Dashboard queries use `fill(last, 60*15)` to extend state, or we issue a synthetic GAUGE with `metric:metta.infra.state.carry_forward` if Datadog's query fill proves insufficient.

---

## 4. Pipeline Architecture

### 1️⃣ Cron collectors (high frequency)

```
K8s CronJob (devops/charts/cronjob)
  -> Python entrypoint `devops/datadog/cli.py dashboard push`
       -> Collector registry (GitHub, CI, Eval, Bug queue, Commit stats)
            -> Each collector returns {metric_name: value, tags: {...}}
       -> Metric builder normalizes names/tags & pushes via Metrics API
```

Implementation notes:
- Reuse Dockerfile in `devops/charts/cronjob/Dockerfile.dashboard`.
- Store collectors under `devops/datadog/collectors/<domain>/`.
- Provide dry-run mode (prints JSON, no push) for local testing.
- Each collector should own its own cache / rate-limit handling (e.g., ETags for GitHub).

### 2️⃣ Stable suite ingestion (low frequency)

```
Stable suite runner -> writes structured artifact (JSON or parquet) to S3
  -> Post-run hook (or periodic scraper) reads latest artifacts
       -> Translator normalizes into the same metric schema
       -> Option A: reuse cron container with `dashboard import-stable`
       -> Option B: add lightweight Lambda triggered by artifact upload
```

Key requirement: keep schema identical so Datadog widgets can mix `source:cron` and `source:stable_suite`.

---

## 5. Dashboard Modeling Implications

- **Heatmap rows:** Query `avg:last_15m:metta.infra.*{workflow_name:<row>,task:<task>}` grouped by `status`. Use Datadog "Query Table" or "Change" widget with conditional format (green = value==1, red=0).
- **CI smoothness tiles:** Distribution metrics allow `p90(workflow_duration)` charts directly.
- **Bug counts:** Query `sum:metta.infra.cron.github.bugs.count{label:Training}`.

Manual dashboard v1 will be built via Datadog UI once first metrics land. Later we can codify it via Terraform or the Datadog dashboard API.

---

## 6. Next Implementation Steps

1. Scaffold `devops/datadog` package (CLI, collector registry, shared dataclasses).
2. Wire the existing cron Helm chart to invoke the new CLI.
3. Implement collectors for:
   - GitHub (CI summaries, commits, reverts, flaky tests via GitHub API + Actions API).
   - Eval + local run health (call `./tools/request_eval` endpoints / internal APIs).
   - Bug queue (GitHub Issues + labels).
4. Emit metrics using the schema above; validate via `--dry-run` locally.
5. Enable push mode in cron (using Datadog secrets) and verify metrics in Datadog Metrics Explorer.
6. Once stable-suite translator is ready, emit with `source:stable_suite` tag and reuse dashboards.

This plan unblocks Step 2 (building the cron job) by locking down how data must look when it hits Datadog.
