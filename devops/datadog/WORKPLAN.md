# Datadog Observability System - Work Plan

**Current Status**: Production deployment with 7 collectors, 5 migrated dashboards (Jsonnet framework), and 123+ metrics

**Branch**: `robb/1022-datadog` **PR**: [#3384](https://github.com/Metta-AI/metta/pull/3384)

**Dashboard Framework**: Comprehensive Jsonnet library with layouts, presets, and component libraries (2,550+ lines of documentation)

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
- `demo.jsonnet` → Demo/reference dashboard

**🔧 Python-Generated** (complex visualizations, no migration needed):
- `system_health_rollup.json` → 7×7 FoM grid (65 widgets) via `generate_health_grid.py`
- `system_health_rollup_wildcard.json` → Vega-Lite heatmap via `generate_wildcard_fom_grid.py`

**⚠️ Legacy JSON** (needs migration to Jsonnet):
- `policy_evaluator.json` → APM dashboard (eval-orchestrator/worker latency)

**❌ Missing Dashboards** (collectors without dashboards):
- Kubernetes collector (15 metrics) - no dashboard yet
- WandB collector (10 metrics) - no dashboard yet

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

### Dashboard Migration Tasks

**High Priority** (migrate existing dashboard):
- [ ] Migrate `policy_evaluator.json` to Jsonnet framework
  - Simple 3-widget APM dashboard (eval-orchestrator/worker latency)
  - Should take ~15 minutes using new framework

**Medium Priority** (create missing dashboards):
- [ ] Create Kubernetes dashboard (`kubernetes.jsonnet`)
  - 15 metrics available: pods, deployments, node health, resource waste
  - Use new framework with layouts + presets
- [ ] Create WandB dashboard (`wandb.jsonnet`)
  - 10 metrics available: training runs, model performance, resource usage
  - Use new framework with layouts + presets

### Near-term (When Needed)

Ideas for future work - prioritize based on actual needs:

- **Alerting**: Set up critical alerts (CI failures, collector failures, cost spikes)
- **Dashboard improvements**: Add SLO markers, template variables, conditional formatting
- **Multi-environment**: Separate staging/production metrics with environment tags
- **New collectors**: Add based on team needs (evaluate quarterly)
