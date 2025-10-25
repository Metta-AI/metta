# Kubernetes Resource Efficiency & Health Collector

Monitors Kubernetes cluster efficiency, pod health, and resource waste to optimize costs and reliability.

## Overview

This collector tracks three key areas:

1. **Resource Efficiency** - CPU/memory waste from overallocation
2. **Pod Health** - Crashes, failures, and restarts
3. **Underutilization** - Idle and underperforming pods

**Collection Frequency**: Every 5-10 minutes (recommended)

## Metrics Collected (15 total)

### Resource Efficiency Metrics (5)

| Metric                                | Type  | Description                                  | Unit  |
| ------------------------------------- | ----- | -------------------------------------------- | ----- |
| `k8s.resources.cpu_waste_cores`       | gauge | Unused CPU cores (requested but not used)    | cores |
| `k8s.resources.memory_waste_gb`       | gauge | Unused memory (requested but not used)       | GB    |
| `k8s.resources.cpu_efficiency_pct`    | gauge | Percentage of requested CPU actually used    | %     |
| `k8s.resources.memory_efficiency_pct` | gauge | Percentage of requested memory actually used | %     |
| `k8s.resources.overallocated_pods`    | gauge | Pods using <20% of requested resources       | count |

**Why This Matters**: Skypilot API was using only 0.4% of requested CPU (8 cores) - wasting ~$200/month

### Pod Health Metrics (6)

| Metric                       | Type  | Description                                  | Unit  |
| ---------------------------- | ----- | -------------------------------------------- | ----- |
| `k8s.pods.crash_looping`     | gauge | Pods in CrashLoopBackOff state               | count |
| `k8s.pods.failed`            | gauge | Pods in Failed state                         | count |
| `k8s.pods.pending`           | gauge | Pods stuck in Pending state                  | count |
| `k8s.pods.oomkilled_24h`     | gauge | Pods killed due to out-of-memory in last 24h | count |
| `k8s.pods.high_restarts`     | gauge | Pods with >5 container restarts              | count |
| `k8s.pods.image_pull_errors` | gauge | Pods failing to pull container images        | count |

**Why This Matters**: Catch production issues before they impact users

### Underutilization Metrics (4)

| Metric                          | Type  | Description                         | Unit  |
| ------------------------------- | ----- | ----------------------------------- | ----- |
| `k8s.pods.idle_count`           | gauge | Pods using <1m CPU and <10Mi memory | count |
| `k8s.pods.low_cpu_usage`        | gauge | Pods using <10m CPU                 | count |
| `k8s.pods.low_memory_usage`     | gauge | Pods using <50Mi memory             | count |
| `k8s.deployments.zero_replicas` | gauge | Deployments scaled to 0 replicas    | count |

**Why This Matters**: Identify candidates for cost reduction or removal

## Configuration

### Prerequisites

**Kubernetes Permissions**: The collector needs read access to:

- Pods (all namespaces)
- Deployments (all namespaces)
- Pod metrics (metrics.k8s.io/v1beta1)

**Required RBAC**:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kubernetes-collector
rules:
  - apiGroups: ['']
    resources: ['pods']
    verbs: ['get', 'list']
  - apiGroups: ['apps']
    resources: ['deployments']
    verbs: ['get', 'list']
  - apiGroups: ['metrics.k8s.io']
    resources: ['pods']
    verbs: ['get', 'list']
```

**Python Dependencies**:

```bash
pip install kubernetes
```

### Environment Variables

| Variable           | Required | Default         | Description                              |
| ------------------ | -------- | --------------- | ---------------------------------------- |
| `K8S_CLUSTER_NAME` | No       | `main`          | Cluster name for metric tagging          |
| `DD_API_KEY`       | Yes\*    | -               | Datadog API key                          |
| `DD_APP_KEY`       | Yes\*    | -               | Datadog application key                  |
| `DD_SITE`          | No       | `datadoghq.com` | Datadog site (e.g., `us5.datadoghq.com`) |

\*Required only when pushing metrics with `--push`

### Kubernetes Configuration

The collector automatically loads Kubernetes config:

- **In-cluster**: Uses service account token (for CronJob deployment)
- **Local**: Uses `~/.kube/config` (for testing)

## Usage

### Local Testing

```bash
# Test collection (dry-run)
uv run python devops/datadog/scripts/run_collector.py kubernetes --verbose

# Test and push to Datadog
uv run python devops/datadog/scripts/run_collector.py kubernetes --push

# JSON output
uv run python devops/datadog/scripts/run_collector.py kubernetes --json
```

### Sample Output

```
Collecting Kubernetes efficiency and health metrics...
Collected 15 metrics

Collected metrics:
  k8s.deployments.zero_replicas: 0
  k8s.pods.crash_looping: 0
  k8s.pods.failed: 0
  k8s.pods.high_restarts: 0
  k8s.pods.idle_count: 3
  k8s.pods.image_pull_errors: 0
  k8s.pods.low_cpu_usage: 12
  k8s.pods.low_memory_usage: 8
  k8s.pods.oomkilled_24h: 0
  k8s.pods.pending: 0
  k8s.resources.cpu_efficiency_pct: 2.45
  k8s.resources.cpu_waste_cores: 15.47
  k8s.resources.memory_efficiency_pct: 34.12
  k8s.resources.memory_waste_gb: 23.56
  k8s.resources.overallocated_pods: 4
```

**Interpretation**:

- 🚨 **Only 2.45% CPU efficiency** - wasting 15.47 cores!
- 🚨 **4 pods overallocated** - using <20% of requested resources
- ✅ **No health issues** - all pods running healthy
- ⚠️ **3 idle pods** - candidates for removal

### Kubernetes CronJob Deployment

Deploy as a CronJob to collect metrics every 5 minutes:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: kubernetes-collector
  namespace: monitoring
spec:
  schedule: '*/5 * * * *'
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: kubernetes-collector
          containers:
            - name: collector
              image: your-registry/datadog-collector:latest
              command:
                - uv
                - run
                - python
                - devops/datadog/scripts/run_collector.py
                - kubernetes
                - --push
              env:
                - name: K8S_CLUSTER_NAME
                  value: 'production'
                - name: DD_ENV
                  value: 'production'
                - name: DD_SERVICE
                  value: 'kubernetes-collector'
              envFrom:
                - secretRef:
                    name: datadog-secret
          restartPolicy: OnFailure
```

## Dashboard Recommendations

### 1. Resource Waste Dashboard

**Widget**: Timeseries

- `k8s.resources.cpu_waste_cores` - Track CPU waste trend
- `k8s.resources.memory_waste_gb` - Track memory waste trend

**Widget**: Query Value

- `k8s.resources.cpu_efficiency_pct` - Show current efficiency
- Alert if < 30% (very wasteful)

**Widget**: Top List

- `k8s.resources.overallocated_pods` by pod name
- Shows worst offenders for rightsizing

### 2. Pod Health Dashboard

**Widget**: Heatmap

- `k8s.pods.crash_looping` by namespace
- `k8s.pods.oomkilled_24h` by namespace

**Widget**: Query Value

- `sum:k8s.pods.failed{*}` - Total failed pods
- Alert if > 0

### 3. Cost Optimization Dashboard

**Widget**: Calculation

```
estimated_monthly_waste = (cpu_waste_cores * $30/core + memory_waste_gb * $4/GB) * 730
```

**Widget**: Top List

- Idle pods sorted by namespace
- Zero-replica deployments for cleanup

## Alerting Recommendations

### Critical Alerts

```yaml
# Alert: Pod Crashes
name: 'Kubernetes: Pod Crash Looping'
query: 'avg(last_5m):sum:k8s.pods.crash_looping{*} > 0'
message: |
  {{#is_alert}}
  ⚠️ Pods are crash looping in cluster {{cluster.name}}
  Count: {{value}}
  {{/is_alert}}
```

```yaml
# Alert: OOM Kills
name: 'Kubernetes: Out of Memory Kills'
query: 'avg(last_30m):sum:k8s.pods.oomkilled_24h{*} > 2'
message: |
  {{#is_alert}}
  🚨 Multiple pods killed due to OOM in cluster {{cluster.name}}
  Count: {{value}}
  Action: Increase memory limits for affected pods
  {{/is_alert}}
```

### Warning Alerts

```yaml
# Alert: Low Efficiency
name: 'Kubernetes: Very Low CPU Efficiency'
query: 'avg(last_1h):avg:k8s.resources.cpu_efficiency_pct{*} < 10'
message: |
  {{#is_warning}}
  💸 Cluster CPU efficiency is very low: {{value}}%
  This indicates significant resource overallocation.
  Review k8s.resources.overallocated_pods for rightsizing candidates.
  {{/is_warning}}
```

```yaml
# Alert: Pending Pods
name: 'Kubernetes: Pods Stuck Pending'
query: 'avg(last_15m):sum:k8s.pods.pending{*} > 3'
message: |
  {{#is_warning}}
  ⚠️ {{value}} pods stuck in Pending state
  Possible causes: insufficient cluster resources, node selectors, or scheduling constraints
  {{/is_warning}}
```

## Troubleshooting

### Issue: "No metrics collected"

**Cause**: Metrics server not installed or not responding

**Solution**:

```bash
# Check if metrics-server is installed
kubectl get deployment metrics-server -n kube-system

# If not installed, install it
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify it's working
kubectl top nodes
```

### Issue: "Permission denied" errors

**Cause**: Insufficient RBAC permissions

**Solution**:

```bash
# Apply RBAC permissions (see Configuration section)
kubectl apply -f rbac.yaml

# Verify permissions
kubectl auth can-i get pods --all-namespaces
kubectl auth can-i get deployments --all-namespaces
kubectl auth can-i get pods.metrics.k8s.io --all-namespaces
```

### Issue: CPU/Memory efficiency is 0%

**Cause**: Pod metrics not available yet (metrics-server needs time to collect)

**Solution**: Wait 1-2 minutes and try again. Metrics-server scrapes every 60 seconds.

### Issue: Only collecting some metrics

**Cause**: Partial metrics-server data or API connectivity issues

**Check**:

```bash
# Verify metrics-server is healthy
kubectl get apiservice v1beta1.metrics.k8s.io -o yaml

# Check metrics-server logs
kubectl logs -n kube-system deployment/metrics-server

# Test metrics API directly
kubectl get --raw /apis/metrics.k8s.io/v1beta1/pods
```

## Cost Impact Examples

### Example 1: Skypilot API Server

**Before**:

- Requested: 8 CPU, 16Gi memory
- Actual usage: 34m CPU, 5.3Gi memory
- Waste: 7.97 CPU cores, 10.7Gi memory
- **Monthly cost**: ~$240/month wasted

**After rightsizing** (recommended: 500m CPU, 8Gi memory):

- **Savings**: ~$180/month (75% reduction)

### Example 2: Cluster-Wide Optimization

**Current metrics**:

- CPU efficiency: 2.45%
- Memory efficiency: 34.12%
- Overallocated pods: 4

**Potential savings**: 15.47 cores _ $30/core _ 730 hours = **$3,388/month**

## Integration with Other Collectors

The Kubernetes collector complements other collectors:

- **EC2 Collector**: Shows node-level costs and utilization
- **GitHub Collector**: Correlate deployment frequency with pod crashes
- **Skypilot Collector**: Compare training job resource usage

**Combined Dashboard**: Track end-to-end infrastructure efficiency

## Maintenance

### Updating Metrics

To add new metrics:

1. Add metric collection logic to appropriate `_collect_*` method
2. Update this README with metric documentation
3. Add to dashboard templates
4. Create alert templates if needed

### Testing

```bash
# Run tests
pytest devops/datadog/tests/collectors/test_kubernetes.py -v

# Test against live cluster
uv run python devops/datadog/scripts/run_collector.py kubernetes --verbose
```

## References

- [Kubernetes Metrics Server](https://github.com/kubernetes-sigs/metrics-server)
- [Kubernetes Python Client](https://github.com/kubernetes-client/python)
- [Datadog Kubernetes Integration](https://docs.datadoghq.com/integrations/kubernetes/)
- [Resource Requests and Limits](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)

## Contributing

When improving this collector:

- Maintain backward compatibility with existing metrics
- Add comprehensive error handling
- Update documentation and examples
- Include tests for new functionality
