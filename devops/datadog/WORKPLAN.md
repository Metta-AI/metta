# Datadog Observability System - Work Plan

**Current Status**: Production deployment with 7 collectors, 8 dashboards (all using Jsonnet), and 130+ metrics

**Branch**: `robb/1022-datadog` **PR**: [#3384](https://github.com/Metta-AI/metta/pull/3384)

**Dashboard Framework**: Comprehensive Jsonnet library with layouts, presets, and 7 component libraries (2,550+ lines of documentation)

**✅ All Dashboards Migrated**: Every collector now has a Jsonnet-based dashboard with component libraries

---

## ✅ What's Deployed and Working

### 🔌 Seven Collectors (137+ metrics)

Running in production via unified CronJob (every 15 minutes):

1. **GitHub Collector** (28 metrics) - PRs, commits, CI/CD, code changes, developers
2. **Skypilot Collector** (30 metrics) - Job status, runtime, GPU utilization, regional distribution
3. **Asana Collector** (14 metrics) - Tasks, velocity, cycle time, bugs workflow
4. **EC2 Collector** (19 metrics) - Instances, EBS volumes, snapshots, cost estimates
5. **WandB Collector** (20 metrics) - Training runs (24h), push-to-main CI tracking (5 SPS metrics), sweep metrics, GPU hours
6. **Kubernetes Collector** (15 metrics) - Pods, deployments, node health, resource waste
7. **Health FoM Collector** (14 metrics) - CI/CD and Training health scores (0.0-1.0 scale)

---

### 📊 Dashboard Status

**✅ Production Collector Dashboards** (8 dashboards):
- `github_cicd.jsonnet` → GitHub CI/CD metrics (ID: 7gy-9ub-2sq)
- `skypilot_jobs.jsonnet` → Skypilot job tracking (ID: mtw-y2p-4ed)
- `ec2.jsonnet` → AWS EC2 infrastructure (ID: 4ue-n4w-b7a)
- `asana.jsonnet` → Project management (ID: srz-bhk-zr2)
- `policy_evaluator.jsonnet` → APM evaluation metrics (ID: gpk-2y2-9er)
- `kubernetes.jsonnet` → Kubernetes cluster health (ID: 687-i5n-ncf)
- `wandb.jsonnet` → WandB training metrics with push-to-main CI tracking (ID: dr3-pdj-rrw)
- `health_fom` → System health scores (Python-generated)


**🔧 Python-Generated** (complex visualizations, no migration needed):
- `system_health_rollup.json` → 14×7 FoM grid (121 widgets) via `generate_health_grid.py` (ID: h3w-ibt-gkv)
- `system_health_rollup_wildcard.json` → Vega-Lite heatmap via `generate_wildcard_fom_grid.py`


### 🏗️ Infrastructure

- **Helm Chart**: `devops/charts/dashboard-cronjob/` - Unified CronJob deployment
- **Security**: All credentials in AWS Secrets Manager with IRSA
- **Management**: CLI via `metta datadog` + Python scripts in `devops/datadog/scripts/`
- **Documentation**: Full guides in `devops/datadog/docs/`

---

## 🎯 Next Steps

### Completed

- [x] **Fixed health_fom collector crashes**
  - Added try/except around collect_metrics() to prevent job crashes
  - Collector gracefully returns empty dict when source metrics unavailable
  - Reduced Datadog API lookback from 1 hour to 30 minutes
  - Changed missing metric logging from WARNING to DEBUG level

- [x] **Fixed GitHub metrics naming** (commits 2b65385e2a, f4512bdeb7)
  - Added `github.` prefix to all 28 metric names throughout codebase
  - Updated collector metric assignments, error handling, and dashboard components
  - Metrics now consistent with other collectors (wandb.*, asana.*, ec2.*)
  - Tested locally and pushed to Datadog - metrics appear in catalog
  - Health_fom collector already queries correct names

### Next Steps

- [ ] Build and deploy updated image with GitHub metrics fix
- [ ] Address PR feedback and merge to main
- [ ] Monitor production stability (95%+ collector success rate)

### Dashboard Migration - ✅ Complete!

All collector dashboards have been successfully migrated to the Jsonnet framework:
- ✅ Migrated `policy_evaluator` from legacy JSON to Jsonnet
- ✅ Created `kubernetes` dashboard with 15 metrics across 3 sections
- ✅ Created `wandb` dashboard with 10 metrics for ML training tracking

**Component Libraries Created**:
- `components/kubernetes.libsonnet` - 18 widget functions for cluster health
- `components/wandb.libsonnet` - 14 widget functions for ML training (now includes push-to-main CI and sweep metrics)

### Near-term (When Needed)

Ideas for future work - prioritize based on actual needs:

- **Alerting**: Set up critical alerts (CI failures, collector failures, cost spikes)
- **Dashboard improvements**: Add SLO markers, template variables, conditional formatting
- **Multi-environment**: Separate staging/production metrics with environment tags
- **New collectors**: Add based on team needs (evaluate quarterly)
