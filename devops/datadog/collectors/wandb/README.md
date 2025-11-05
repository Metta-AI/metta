# WandB Collector
Collects per-run training metrics from Weights & Biases (wandb.ai).
**Architecture**: Emits **instantaneous per-run values** with tags, allowing Datadog to perform flexible aggregations. No pre-calculated averages or counts - the collector reports facts, Datadog does the math.
**Collection Window**: Last 24 hours with server-side filtering (avoids fetching 26k+ historical runs).
## Metrics Collected
### Metric Structure
All metrics follow the pattern:
```
wandb.(run_type).(state).(metric)
Run Types: ptm, sweep, stable, local, other
States: success, failure, active
Metrics: duration_hours, sps, hearts_gained, skypilot_latency_s
```
**Tags**: Each metric point includes:
- `run_id:{wandb_run_id}` - Deduplication key
- `run_type:{ptm|sweep|stable|local|other}` - Category of run
- `state:{success|failure|active}` - Run completion state
**Run Type Detection** (checked in order):
1. **ptm**: Starts with `github.sky.` (GitHub CI)
2. **sweep**: Contains "sweep" in name or has WandB sweep attribute
3. **stable**: Contains "stable" in name (baseline smoke tests)
4. **local**: Contains "local" in name (development runs)
5. **other**: Everything else (catch-all)
### Per-Run Type Metrics
Each run type emits the same set of metrics, differentiated by `run_type` tag:
**Core Metrics** (all run types):
- `wandb.(type).success.duration_hours` - Duration of successful runs
- `wandb.(type).success.sps` - Training throughput (steps/second)
- `wandb.(type).failure.duration_hours` - Duration of failed runs
- `wandb.(type).failure.sps` - Training throughput for failed runs
- `wandb.(type).active.duration_hours` - Current elapsed time for running experiments
**PTM-Only Metrics** (GitHub CI runs):
- `wandb.ptm.success.hearts_gained` - Hearts gained in environment
- `wandb.ptm.success.skypilot_latency_s` - SkyPilot queue latency
- `wandb.ptm.failure.*` - Same for failed runs
**Active Runs**: Running experiments emit only `duration_hours` (elapsed time so far). Final metrics (SPS, hearts, latency) are available only when runs complete.
### Run Type Breakdown
**PTM (Push-to-Main)**:
- Pattern: `github.sky.*`
- Purpose: Track baseline performance of GitHub CI runs
- Examples: `github.sky.main.abc123...`, `github.sky.pr42.def456...`
- Special metrics: hearts_gained, skypilot_latency_s
**Sweep (Hyperparameter Tuning)**:
- Pattern: Contains "sweep" in name or has WandB sweep attribute
- Purpose: Track experimental hyperparameter optimization runs
- Examples: `ak.sweep.muon_sweep.v0_trial_0064`
**Stable (Baseline Smoke Tests)**:
- Pattern: Contains "stable" in name
- Purpose: Track quick smoke test runs for baseline verification
- Examples: `stable.smoke.20251027_151840`
**Local (Development Runs)**:
- Pattern: Contains "local" in name
- Purpose: Track local development and debugging runs
- Examples: `local.ak.20251026.212243`
**Other (Catch-All)**:
- Pattern: Everything else
- Purpose: Experimental runs that don't fit other categories
- Examples: `msb_nav_cc_v26_async1`
**Total**: ~15 unique metric names, each with multiple data points per collection
## Example Data Points
When you run the collector, you'll see output like:
```
wandb.ptm.failure.sps: 12 data points
  [1] 751699.72 (run_id:github.sky.main.abc123..., run_type:ptm, state:failure)
  [2] 745341.29 (run_id:github.sky.main.def456..., run_type:ptm, state:failure)
  ...
wandb.ptm.failure.hearts_gained: 12 data points
  [1] 1.41 (run_id:github.sky.main.abc123..., run_type:ptm, state:failure)
  [2] 1.07 (run_id:github.sky.main.def456..., run_type:ptm, state:failure)
  ...
wandb.sweep.active.duration_hours: 4 data points
  [1] 1.14 (run_id:ak.sweep.muon_sweep.v0_trial_0085..., run_type:sweep, state:active)
  [2] 0.96 (run_id:ak.sweep.muon_sweep.v0_trial_0087..., run_type:sweep, state:active)
  ...
wandb.stable.success.sps: 8 data points
  [1] 87.65 (run_id:stable.smoke.20251027_151840, run_type:stable, state:success)
  [2] 793.85 (run_id:stable.smoke.20251027_152231, run_type:stable, state:success)
  ...
```
Each completed run emits 2+ metrics (duration, sps, and for PTM: hearts, latency). Active runs emit only duration (elapsed time so far).
## Dashboard Queries
Since metrics are per-run, use Datadog aggregation functions in dashboards:
### Average Metrics (24h)
```
avg:wandb.ptm.success.sps{*}  # Average SPS for successful PTM runs
avg:wandb.ptm.success.duration_hours{*}  # Average duration
avg:wandb.ptm.success.hearts_gained{*}  # Average hearts
```
### Percentiles
```
p50:wandb.ptm.success.sps{*}  # Median SPS
p95:wandb.ptm.success.duration_hours{*}  # 95th percentile duration
```
### Counts
```
count:wandb.ptm.success.duration_hours{*}  # Number of successful PTM runs
count:wandb.ptm.failure.duration_hours{*}  # Number of failed PTM runs
count:wandb.*.active.duration_hours{*}  # Total active runs (all types)
count:wandb.sweep.*.duration_hours{*}  # All sweep runs (success + failure + active)
```
### Success Rate
```
count:wandb.ptm.success.duration_hours{*} / (count:wandb.ptm.success.duration_hours{*} + count:wandb.ptm.failure.duration_hours{*}) * 100
```
### Time Series Over Different Windows
```
avg:wandb.ptm.success.sps{*}.rollup(avg, 3600)  # Hourly average
avg:wandb.ptm.success.sps{*}.rollup(avg, 86400) # Daily average
```
### Filter by State and Type
```
avg:wandb.*.duration_hours{state:success}  # All successful runs (any type)
avg:wandb.ptm.duration_hours{*}  # All PTM runs (any state)
avg:wandb.sweep.duration_hours{state:success}  # Only successful sweep runs
```
## Configuration
### Environment Variables
Required:
- `WANDB_API_KEY` - WandB API key (or stored in AWS Secrets Manager)
- `WANDB_ENTITY` - WandB entity/username (default: `metta-research`)
- `WANDB_PROJECT` - WandB project name (default: `metta`)
### AWS Secrets Manager
Store credentials in AWS Secrets Manager (recommended for production):
```bash
# Create secret with all WandB configuration
aws secretsmanager create-secret \
  --name dev/datadog/collectors/wandb \
  --secret-string '{
    "WANDB_API_KEY": "your_api_key_here",
    "WANDB_ENTITY": "metta-research",
    "WANDB_PROJECT": "metta"
  }'
```
### WandB API Key
Get your API key from https://wandb.ai/settings
## Usage
### Local Testing
```bash
# Test collection (dry run - no push to Datadog)
uv run python devops/datadog/scripts/run_collector.py wandb --verbose
# View collected metrics
uv run python devops/datadog/scripts/run_collector.py wandb
# Test with push to Datadog
uv run python devops/datadog/scripts/run_collector.py wandb --push --verbose
```
Expected output:
```
INFO:devops.datadog.utils.base.wandb:Fetched 52 runs from last 24h
INFO:devops.datadog.utils.base.wandb:Processed 52 runs, 4 active
Collected metrics:
  wandb.ptm.failure.sps: 12 data points
  wandb.ptm.failure.hearts_gained: 12 data points
  wandb.sweep.active.duration_hours: 4 data points
  wandb.sweep.success.duration_hours: 21 data points
  wandb.stable.success.sps: 8 data points
  ...
```
### Kubernetes Deployment
Deployed as part of the unified CronJob that runs all collectors every 15 minutes:
```bash
# Deploy dev environment
helm upgrade dashboard-cronjob-dev devops/charts/dashboard-cronjob \
  --namespace monitoring \
  --set nameOverride=dev \
  --set image.tag=sha-<commit>
# Deploy production
helm upgrade dashboard-cronjob devops/charts/dashboard-cronjob \
  --namespace monitoring
```
## Implementation Details
### Architecture: Instantaneous Metrics with Datadog Aggregation
**Why This Design**:
- **Flexible Time Windows**: Dashboard can show 1h, 24h, 7d without redeploying collector
- **Rich Aggregations**: Can compute avg(), p50(), p95(), count() in queries
- **Granular Filtering**: Filter by run type, state, or individual runs
- **Simpler Collector**: No aggregation logic, just reports facts
**How It Works**:
1. Fetch recent runs (last 24h) with server-side filtering
2. Categorize each run by type (ptm/sweep/all) and state (success/failure)
3. Extract metrics from each run (duration, SPS, hearts, latency)
4. Emit per-run data points with tags (run_id, run_type, state)
5. Datadog automatically deduplicates by run_id + timestamp
### Run Categorization
**Push-to-Main (PTM) Runs**:
- Pattern: `github.sky.*`
- Examples:
  - `github.sky.main.<commit>.<config>.<timestamp>`
  - `github.sky.pr<number>.<commit>.<config>.<timestamp>`
- Purpose: Track baseline performance of main branch
**Sweep Runs**:
- Detection: `.sweep` in name OR WandB sweep attribute set
- Example: `ak.sweep.muon_sweep.v0_trial_0064_e878c1`
- Purpose: Track hyperparameter optimization experiments
**Regular Runs**:
- Everything else (development runs, manual experiments)
- Category: `all`
### Performance Optimizations
**Server-Side Filtering**:
- Uses WandB API filters to fetch only last 24h
- Avoids downloading 26k+ historical runs
- Single API call fetches all runs at once
**Efficient Processing**:
- Categorizes runs in single pass through data
- Extracts all metrics from each run simultaneously
- No redundant API calls or data fetching
**Timeout Protection**:
- 120s timeout prevents indefinite hangs
- Gracefully handles API slowness or failures
### Data Extraction
**Duration Calculation**:
```python
created = datetime.fromisoformat(run.created_at)
heartbeat = datetime.fromisoformat(run.heartbeat_at)
duration_hours = (heartbeat - created).total_seconds() / 3600
```
**SPS Metric Priority**:
1. `overview/steps_per_second` (preferred)
2. `overview/sps` (fallback)
**PTM-Specific Metrics**:
- `env_agent/heart.gained` - Hearts gained in environment
- `skypilot/queue_latency_s` - SkyPilot queue wait time
### Crashed Run Support
The collector handles WandB API quirk where crashed runs return `_json_dict` as JSON string instead of dict:
```python
if isinstance(run.summary._json_dict, str):
    summary_dict = json.loads(run.summary._json_dict)  # Crashed run
else:
    summary_dict = run.summary._json_dict  # Finished run
```
This enables extracting performance metrics from failed CI runs, which is critical for debugging.
### Deduplication
**How Datadog Deduplicates**:
- Uses combination of: metric name + tags + timestamp
- Multiple collections of same run (same `run_id`) → single data point per timestamp window
- Allows collector to re-emit recent runs without duplicate counting
**Benefits**:
- Simpler collector logic (just emit all recent runs)
- No need to track "last collection time"
- Handles collector restarts gracefully
## Troubleshooting
### No metrics collected
- Verify `WANDB_API_KEY` is valid
- Check entity/project names are correct: `metta-research/metta`
- Ensure project has runs logged to it
- Check AWS Secrets Manager permissions (IRSA for Kubernetes)
### Missing PTM metrics
- Ensure CI runs follow naming pattern `github.sky.*`
- Check that runs log metrics to WandB summary:
  - `overview/steps_per_second` or `overview/sps`
  - `env_agent/heart.gained` (PTM only)
  - `skypilot/queue_latency_s` (PTM only)
- Verify runs reach "finished" or "crashed" state (running runs don't have final metrics)
### Missing sweep metrics
- Ensure sweep runs have `.sweep` in the name or WandB sweep attribute
- Check that sweep runs are completing successfully
- Verify runs are within the 24-hour window
### Fewer data points than expected
**This is normal** - the collector emits per-run metrics with deduplication:
- 12 PTM runs in last 24h → 12 data points for `wandb.ptm.success.sps`
- Not thousands of data points - one per unique run
### Metrics not appearing in Datadog
- Check collector logs for errors
- Verify Datadog API key is valid
- Ensure metrics are being pushed (`--push` flag or production environment)
- Wait 1-2 minutes for metrics to appear (Datadog ingestion delay)
- Search in Metric Explorer for `wandb.*`
### Authentication errors
- Regenerate API key if expired
- Verify AWS Secrets Manager permissions
- Check IAM role has `secretsmanager:GetSecretValue`
- For IRSA issues, verify service account annotation
### High metric volume warning
If you see high cardinality warnings in Datadog:
- Each unique `run_id` creates a new tag combination
- This is expected - we're tracking individual runs
- Datadog deduplicates by run_id + timestamp
- Limited to runs in last 24h (~50-100 runs typically)
## Dependencies
- `wandb` - Official WandB Python client
## Metric Evolution
### Previous Architecture (Pre-2025)
Collector pre-aggregated metrics:
```python
"wandb.training.avg_duration_hours": 2.5
"wandb.push_to_main.runs_completed_24h": 26
"wandb.push_to_main.overview.steps_per_second": 750000.0  # Latest value
```
**Limitations**:
- Hard-coded 24h time window
- No flexibility to query different time ranges
- Only avg() available, no p95 or other percentiles
- Couldn't filter by run state or type in dashboard
### Current Architecture (2025+)
Collector emits per-run instantaneous values:
```python
"wandb.ptm.success.sps": [
    (751699.72, ["run_id:abc123", "run_type:ptm", "state:success"]),
    (745341.29, ["run_id:def456", "run_type:ptm", "state:success"]),
    ...
]
```
**Benefits**:
- Query any time window in dashboard (1h, 24h, 7d, custom)
- Use any aggregation (avg, p50, p95, p99, max, min, count)
- Filter by state, type, or individual runs
- Historical data can be re-queried with new aggregations
## Related Documentation
- [Metric Best Practices](../../docs/METRIC_BEST_PRACTICES.md) - Design guide for choosing metrics
- [WandB Python API](https://docs.wandb.ai/ref/python/public-api/)
- [Adding New Collector](../../docs/ADDING_NEW_COLLECTOR.md)
- [Collectors Architecture](../../docs/COLLECTORS_ARCHITECTURE.md)
- [Datadog Metrics Guide](https://docs.datadoghq.com/metrics/)
