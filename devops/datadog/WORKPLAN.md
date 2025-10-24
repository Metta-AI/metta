# Datadog Observability System - Work Plan

**Current Status**: Production deployment with 7 collectors, 8 dashboards (all migrated to Jsonnet), and 123+ metrics

**Branch**: `robb/1022-datadog` **PR**: [#3384](https://github.com/Metta-AI/metta/pull/3384)

**Dashboard Framework**: Comprehensive Jsonnet library with layouts, presets, and 7 component libraries (2,550+ lines of documentation)

**✅ All Dashboards Migrated**: Every collector now has a Jsonnet-based dashboard with component libraries

---

## ✅ What's Deployed and Working

### 🔌 Seven Collectors (123+ metrics)

Running in production via unified CronJob (every 15 minutes):

1. **GitHub Collector** (28 metrics) - PRs, commits, CI/CD, code changes, developers
2. **Skypilot Collector** (30 metrics) - Job status, runtime, GPU utilization, regional distribution
3. **Asana Collector** (14 metrics) - Tasks, velocity, cycle time, bugs workflow
4. **EC2 Collector** (19 metrics) - Instances, EBS volumes, snapshots, cost estimates
5. **WandB Collector** (10 metrics) - Training runs, model performance, resource usage
6. **Kubernetes Collector** (15 metrics) - Pods, deployments, node health, resource waste
7. **Health FoM Collector** (7 metrics) - CI/CD health scores (0.0-1.0 scale)

---

### 📊 Dashboard Status

**✅ Migrated to Jsonnet Framework** (using layouts + presets):
- `github_cicd.jsonnet` → GitHub CI/CD metrics (ID: 7gy-9ub-2sq)
- `skypilot_jobs.jsonnet` → Skypilot job tracking (ID: mtw-y2p-4ed)
- `ec2.jsonnet` → AWS EC2 infrastructure (ID: 4ue-n4w-b7a)
- `asana.jsonnet` → Project management (ID: srz-bhk-zr2)
- `policy_evaluator.jsonnet` → APM evaluation metrics (ID: gpk-2y2-9er)
- `kubernetes.jsonnet` → Kubernetes cluster health (ID: 687-i5n-ncf)
- `wandb.jsonnet` → WandB training metrics (ID: dr3-pdj-rrw)
- `demo.jsonnet` → Demo/reference dashboard

**🔧 Python-Generated** (complex visualizations, no migration needed):
- `system_health_rollup.json` → 7×7 FoM grid (65 widgets) via `generate_health_grid.py`
- `system_health_rollup_wildcard.json` → Vega-Lite heatmap via `generate_wildcard_fom_grid.py`


### 🏗️ Infrastructure

- **Helm Chart**: `devops/charts/dashboard-cronjob/` - Unified CronJob deployment
- **Security**: All credentials in AWS Secrets Manager with IRSA
- **Management**: CLI via `metta datadog` + Python scripts in `devops/datadog/scripts/`
- **Documentation**: Full guides in `devops/datadog/docs/`

---

## 🎯 Next Steps

### Immediate (This Week)

- [ ] Address PR feedback and merge to main
- [ ] Monitor production stability (95%+ collector success rate)

### Dashboard Migration - ✅ Complete!

All collector dashboards have been successfully migrated to the Jsonnet framework:
- ✅ Migrated `policy_evaluator` from legacy JSON to Jsonnet
- ✅ Created `kubernetes` dashboard with 15 metrics across 3 sections
- ✅ Created `wandb` dashboard with 10 metrics for ML training tracking

**Component Libraries Created**:
- `components/kubernetes.libsonnet` - 18 widget functions for cluster health
- `components/wandb.libsonnet` - 14 widget functions for ML training

### Near-term (When Needed)

Ideas for future work - prioritize based on actual needs:

- **Alerting**: Set up critical alerts (CI failures, collector failures, cost spikes)
- **Dashboard improvements**: Add SLO markers, template variables, conditional formatting
- **Multi-environment**: Separate staging/production metrics with environment tags
- **New collectors**: Add based on team needs (evaluate quarterly)
