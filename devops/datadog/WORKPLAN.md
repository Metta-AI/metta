# Datadog Observability System - Work Plan

**Current Status**: Production deployment with 7 collectors, 8 dashboards, and 123+ metrics

**Branch**: `robb/1022-datadog` **PR**: [#3384](https://github.com/Metta-AI/metta/pull/3384)

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

### 📊 Eight Dashboards Deployed

- `github_cicd.json` - CI/CD metrics and workflow health
- `skypilot_jobs.json` - Job tracking and GPU utilization
- `asana.json` - Project tracking and bug workflow
- `ec2.json` - Infrastructure monitoring and costs
- `system_health_rollup.json` - 7×7 FoM grid (65 widgets)
- `system_health_rollup_wildcard.json` - 7×7 FoM grid (Vega-Lite)
- `policy_evaluator.json` - Policy evaluation metrics
- `demo.json` - Demo/reference dashboard

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

### Near-term (When Needed)

Ideas for future work - prioritize based on actual needs:

- **Alerting**: Set up critical alerts (CI failures, collector failures, cost spikes)
- **Dashboard improvements**: Add SLO markers, template variables, conditional formatting
- **Multi-environment**: Separate staging/production metrics with environment tags
- **New collectors**: Add based on team needs (evaluate quarterly)
