# Helm CronJob Conventions Analysis

## Question

Is it standard Helm practice to keep CronJob specifications in `devops/charts/`?

## TL;DR

**Yes, this is absolutely standard Helm practice.** The current location `devops/charts/dashboard-cronjob/` follows established Helm conventions and best practices.

## Analysis

### Current Setup

The project has CronJob specifications in `devops/charts/dashboard-cronjob/` with the following structure:

```
devops/charts/dashboard-cronjob/
├── Chart.yaml              # Helm chart metadata
├── README.md               # Documentation
├── templates/
│   ├── _helpers.tpl        # Template helper functions
│   ├── cronjob.yaml        # CronJob Kubernetes resource
│   └── serviceaccount.yaml # ServiceAccount for IAM role
└── values.yaml             # Default configuration values
```

### Why This Is Standard Practice

#### 1. Helm Chart Convention

A Helm chart is a **collection of Kubernetes resource templates** organized in a specific directory structure. The standard Helm chart structure is:

```
<chart-name>/
├── Chart.yaml      # Chart metadata
├── values.yaml     # Default configuration
├── templates/      # Kubernetes resource templates
│   ├── NOTES.txt   # (optional) Post-install notes
│   └── *.yaml      # Resource templates (Deployment, Service, CronJob, etc.)
└── charts/         # (optional) Dependencies
```

The `dashboard-cronjob` chart follows this convention **exactly**.

#### 2. Infrastructure as Code Best Practice

Keeping Helm charts in version control (like `devops/charts/`) is standard practice because:

- **Declarative Infrastructure**: Charts define infrastructure declaratively
- **Version Control**: Changes are tracked in git
- **Code Review**: Infrastructure changes go through PR review
- **Deployment Automation**: Charts can be deployed via CI/CD (as this project does with helmfile)

#### 3. Comparison with Public Helm Charts

Public Helm repositories (like [Artifact Hub](https://artifacthub.io/)) contain charts for CronJobs following the same structure:

- [kubernetes-cronhpa](https://artifacthub.io/packages/helm/tosone/kubernetes-cronhpa)
- [cronjobs](https://artifacthub.io/packages/helm/pnnl-miscscripts/cronjobs)
- [kube-janitor](https://artifacthub.io/packages/helm/kube-janitor/kube-janitor) (uses CronJob internally)

All follow the same `charts/<name>/templates/*.yaml` pattern.

#### 4. Project Convention Consistency

Looking at other charts in this project:

```
devops/charts/
├── dashboard-cronjob/   # CronJob for metrics collection
├── home/                # Deployment for home page
├── observatory/         # Deployment for observatory UI
├── observatory-backend/ # Deployment for API backend
├── orchestrator/        # Deployment for orchestrator
└── library/             # Deployment for library service
```

Every service/deployment in this project follows the **same pattern**: charts in `devops/charts/<name>/`.

It would be **inconsistent** to move CronJobs elsewhere while keeping Deployments in `devops/charts/`.

### Helmfile Orchestration

The project uses `helmfile.yaml` to deploy all charts:

```yaml
releases:
  - name: dashboard-cronjob
    chart: ./dashboard-cronjob
    version: 0.1.0
    namespace: monitoring
```

This is a common pattern for managing multiple Helm releases in a single repository.

### Reusability Pattern

The README shows an important design pattern:

```yaml
# Same chart, different deployments with different schedules
- name: dashboard-cronjob
  chart: ./dashboard-cronjob

- name: weekly-report
  chart: ./dashboard-cronjob  # Reuse same chart!
  values:
    - schedule: "0 9 * * MON"
      command: ["uv", "run", "python", "-m", "softmax.reports.weekly"]
```

This **single chart, multiple deployments** pattern is a Helm best practice for:
- Reducing duplication
- Ensuring consistency
- Simplifying maintenance

### Alternative (Not Recommended)

Some organizations use alternative structures:

```
# Alternative 1: Monorepo with dedicated infrastructure repo
infrastructure/
└── helm/
    └── charts/
        └── dashboard-cronjob/

# Alternative 2: Separate charts repository
charts-repo/
└── dashboard-cronjob/
```

However, these alternatives are typically used when:
- Charts are shared across multiple projects (not applicable here)
- Infrastructure is managed by a separate team
- Charts are published to a Helm repository

For this project, keeping charts in `devops/charts/` is the **right choice**.

## Comparison with Planned Collector Architecture

In the collector architecture documentation (`COLLECTORS_ARCHITECTURE.md`), we proposed:

```
devops/datadog/
└── charts/
    └── collector-cronjobs/  # Single chart for all collectors
```

This would create a **duplicate location** for Helm charts. Instead, we should:

### Recommended Adjustment

Keep Helm charts in the **existing convention**: `devops/charts/`

```
devops/charts/
├── dashboard-cronjob/       # Current GitHub metrics (keep as-is)
└── datadog-collectors/      # NEW: Multi-collector chart
    ├── Chart.yaml
    ├── values.yaml
    └── templates/
        ├── github-cronjob.yaml
        ├── skypilot-cronjob.yaml
        ├── wandb-cronjob.yaml
        ├── ec2-cronjob.yaml
        └── serviceaccount.yaml
```

**Alternative**: Migrate `dashboard-cronjob` into the new unified chart when ready.

## Recommendations

### 1. Keep Current Location ✅

**Action**: No change needed for `devops/charts/dashboard-cronjob/`

**Reason**: Follows Helm conventions and project patterns

### 2. Update Collector Architecture Docs

**Action**: Update `COLLECTORS_ARCHITECTURE.md` to use `devops/charts/` instead of `devops/datadog/charts/`

**Change**:
```diff
- devops/datadog/charts/collector-cronjobs/
+ devops/charts/datadog-collectors/
```

### 3. Maintain Consistency

**Action**: All future Helm charts should go in `devops/charts/<name>/`

**Examples**:
- `devops/charts/datadog-collectors/` - Multi-collector CronJob chart
- `devops/charts/training-jobs/` - If we add Kubernetes Jobs for training
- `devops/charts/evaluation-workers/` - If we add worker deployments

### 4. Consider Chart Consolidation (Future)

Once the new collector architecture is stable, consider:

**Option A: Keep Separate**
```
devops/charts/
├── dashboard-cronjob/      # GitHub metrics only
└── datadog-collectors/     # Skypilot, WandB, EC2, Asana
```

**Option B: Consolidate (Recommended)**
```
devops/charts/
└── datadog-collectors/     # ALL collectors including GitHub
```

Consolidation would:
- Reduce duplication
- Simplify deployment (one chart, one helmfile entry)
- Maintain single source of truth

## Summary Table

| Question | Answer |
|----------|--------|
| Is `devops/charts/` standard for Helm? | ✅ Yes, absolutely standard |
| Should we move it elsewhere? | ❌ No, keep it where it is |
| Is this consistent with other charts? | ✅ Yes, all charts are in `devops/charts/` |
| Should new collectors go here? | ✅ Yes, in `devops/charts/datadog-collectors/` |
| Is the helmfile approach standard? | ✅ Yes, common for multi-chart repos |

## References

- [Helm Chart Best Practices](https://helm.sh/docs/chart_best_practices/)
- [Helm Chart Template Guide](https://helm.sh/docs/chart_template_guide/)
- [Helmfile Documentation](https://helmfile.readthedocs.io/)
- [Artifact Hub - Public Helm Charts](https://artifacthub.io/)

## Conclusion

The current location (`devops/charts/dashboard-cronjob/`) is **exactly where it should be** according to Helm best practices. The only adjustment needed is updating the collector architecture documentation to use `devops/charts/` instead of `devops/datadog/charts/` for consistency.
