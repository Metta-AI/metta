# Datadog Collector - Build and Deployment Guide

This guide covers the complete workflow for building the Docker image and deploying the Datadog collectors to Kubernetes
using Helm.

## Overview

The Datadog collector system uses:

- **Docker**: Multi-stage build for lightweight production images
- **ECR**: AWS Elastic Container Registry for image storage
- **EKS**: AWS Elastic Kubernetes Service for orchestration
- **Helm**: Package manager for Kubernetes deployments
- **CronJob**: Scheduled metric collection every 15 minutes

**Current Collectors**:

- **GitHub**: PRs, commits, CI/CD, branches, developers (24 metrics)
- **Skypilot**: Jobs, clusters, runtime stats, resources (30 metrics)
- **Asana**: Tasks, velocity, Bugs project tracking (30 metrics)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ GitHub Actions (CI/CD)                                  │
│  - Triggered on push to main                            │
│  - Builds Docker image                                  │
│  - Pushes to ECR                                        │
│  - Deploys to EKS with helm upgrade                     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ ECR (751442549699.dkr.ecr.us-east-1.amazonaws.com)      │
│  - softmax-dashboard:latest                             │
│  - softmax-dashboard:sha-abc123                         │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ EKS Cluster (main)                                      │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Namespace: monitoring                             │  │
│  │  - CronJob: dashboard-cronjob (every 15 min)     │  │
│  │  - ServiceAccount: IAM role for AWS Secrets      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ Datadog                                                 │
│  - Receives metrics via API                             │
│  - Tags: source:<collector>, env:production             │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### Required Tools

```bash
# Docker
docker --version  # Docker version 24.0+

# AWS CLI
aws --version  # AWS CLI 2.x

# kubectl
kubectl version --client  # v1.28+

# helm
helm version  # v3.12+
```

### Required Permissions

- **AWS**: IAM permissions to push to ECR and update EKS cluster
- **EKS**: Access to main cluster in us-east-1
- **Secrets Manager**: Read access to secrets (handled via service account in k8s)

### Setup AWS and EKS Access

```bash
# Login to AWS (if using SSO)
aws sso login --profile softmax-admin

# Configure kubectl for EKS cluster
aws eks update-kubeconfig --name main --region us-east-1 --profile softmax-admin

# Verify access
kubectl get nodes
```

## Local Build and Deploy Workflow

This section provides a complete end-to-end workflow for building and deploying locally, **instead of using the GitHub
Actions CI/CD pipeline**.

### When to Use Local Deployment

- **Testing changes** before pushing to main
- **Rapid iteration** during development
- **Debug builds** with verbose output
- **Custom configurations** not in the main workflow

### Complete Local Workflow

#### Step 1: Authenticate to AWS

```bash
# Login to AWS (if using SSO)
aws sso login --profile softmax-admin

# Configure kubectl for EKS
aws eks update-kubeconfig --name main --region us-east-1 --profile softmax-admin

# Verify cluster access
kubectl get nodes
```

#### Step 2: Build Docker Image

```bash
# IMPORTANT: Build for linux/amd64 (production platform)
# From repo root
docker buildx build \
  --platform linux/amd64 \
  -f softmax/Dockerfile \
  -t softmax-dashboard:local \
  --load \
  .

# This will take 5-10 minutes on first build
# Subsequent builds use cached layers
```

**Expected output:**

- Building workspace packages
- Installing dependencies via uv
- Verifying all three collectors import successfully
- Final message: "✓ All collectors imported successfully"

#### Step 3: Tag for ECR

```bash
# Set variables
export ECR_REGISTRY=751442549699.dkr.ecr.us-east-1.amazonaws.com
export IMAGE_NAME=softmax-dashboard
export TAG=local-$(date +%Y%m%d-%H%M%S)

# Tag the image
docker tag softmax-dashboard:local ${ECR_REGISTRY}/${IMAGE_NAME}:${TAG}
docker tag softmax-dashboard:local ${ECR_REGISTRY}/${IMAGE_NAME}:latest

# Verify tags
docker images | grep softmax-dashboard
```

#### Step 4: Test Locally (Optional)

Before pushing, you can test the image locally:

```bash
# Test GitHub collector
docker run --rm \
  -e AWS_PROFILE=softmax-admin \
  -v ~/.aws:/root/.aws:ro \
  softmax-dashboard:local \
  uv run python devops/datadog/run_collector.py github --verbose

# Test Skypilot collector
docker run --rm \
  -e AWS_PROFILE=softmax-admin \
  -v ~/.aws:/root/.aws:ro \
  softmax-dashboard:local \
  uv run python devops/datadog/run_collector.py skypilot --verbose

# Expected: Metrics collected successfully
```

#### Step 5: Login to ECR

```bash
# Get ECR login credentials and login
aws ecr get-login-password --region us-east-1 --profile softmax-admin | \
  docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Expected: "Login Succeeded"
```

#### Step 6: Push to ECR

```bash
# Push both tags
docker push ${ECR_REGISTRY}/${IMAGE_NAME}:${TAG}
docker push ${ECR_REGISTRY}/${IMAGE_NAME}:latest

# This will take 2-5 minutes depending on connection
```

#### Step 7: Deploy to Kubernetes

**Option A: Update existing deployment (recommended)**

```bash
# Update the GitHub collector CronJob with new image
helm upgrade -n monitoring dashboard-cronjob ./devops/charts/dashboard-cronjob \
  --set image.tag=${TAG}

# Expected: "Release dashboard-cronjob has been upgraded"
```

**Option B: Deploy all three collectors**

```bash
# GitHub collector (every 15 minutes)
helm upgrade -n monitoring github-collector ./devops/charts/dashboard-cronjob \
  --set command[0]=uv \
  --set command[1]=run \
  --set command[2]=python \
  --set command[3]=devops/datadog/run_collector.py \
  --set command[4]=github \
  --set command[5]=--push \
  --set schedule="*/15 * * * *" \
  --set datadog.service=github-collector \
  --set image.tag=${TAG} \
  --install

# Skypilot collector (every 10 minutes)
helm upgrade -n monitoring skypilot-collector ./devops/charts/dashboard-cronjob \
  --set command[0]=uv \
  --set command[1]=run \
  --set command[2]=python \
  --set command[3]=devops/datadog/run_collector.py \
  --set command[4]=skypilot \
  --set command[5]=--push \
  --set schedule="*/10 * * * *" \
  --set datadog.service=skypilot-collector \
  --set image.tag=${TAG} \
  --install

# Asana collector (every 30 minutes)
helm upgrade -n monitoring asana-collector ./devops/charts/dashboard-cronjob \
  --set command[0]=uv \
  --set command[1]=run \
  --set command[2]=python \
  --set command[3]=devops/datadog/run_collector.py \
  --set command[4]=asana \
  --set command[5]=--push \
  --set schedule="*/30 * * * *" \
  --set datadog.service=asana-collector \
  --set image.tag=${TAG} \
  --install
```

#### Step 8: Verify Deployment

```bash
# Check CronJob status
kubectl get cronjobs -n monitoring

# Check recent jobs
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp | tail -5

# View logs from most recent job
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=50

# Expected: "Successfully pushed X metrics to Datadog"
```

#### Step 9: Force a Manual Run (Testing)

```bash
# Trigger a job immediately without waiting for the schedule
kubectl create job --from=cronjob/dashboard-cronjob manual-test-$(date +%s) -n monitoring

# Watch the job
kubectl get jobs -n monitoring --watch

# View logs
kubectl logs -n monitoring job/manual-test-<timestamp>
```

### Quick Local Deploy (One-Liner)

For rapid iteration, combine all steps:

```bash
# Build, tag, push, and deploy in one command
export TAG=local-$(date +%Y%m%d-%H%M%S) && \
docker buildx build --platform linux/amd64 -f softmax/Dockerfile -t softmax-dashboard:local --load . && \
aws ecr get-login-password --region us-east-1 --profile softmax-admin | docker login --username AWS --password-stdin 751442549699.dkr.ecr.us-east-1.amazonaws.com && \
docker tag softmax-dashboard:local 751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-dashboard:${TAG} && \
docker push 751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-dashboard:${TAG} && \
helm upgrade -n monitoring dashboard-cronjob ./devops/charts/dashboard-cronjob --set image.tag=${TAG} && \
echo "✓ Deployed with tag: ${TAG}"
```

### Troubleshooting Local Deployment

**Build Fails:**

```bash
# Clean Docker build cache
docker builder prune -af

# Check platform
docker buildx ls
# Should show linux/amd64 support

# Try with verbose output
docker buildx build --platform linux/amd64 --progress=plain -f softmax/Dockerfile -t softmax-dashboard:local --load .
```

**Push to ECR Fails:**

```bash
# Re-authenticate
aws ecr get-login-password --region us-east-1 --profile softmax-admin | \
  docker login --username AWS --password-stdin 751442549699.dkr.ecr.us-east-1.amazonaws.com

# Verify credentials
aws sts get-caller-identity --profile softmax-admin

# Check repository exists
aws ecr describe-repositories --repository-names softmax-dashboard --region us-east-1
```

**Deployment Fails:**

```bash
# Check Helm release status
helm status -n monitoring dashboard-cronjob

# View Helm history
helm history -n monitoring dashboard-cronjob

# Check pod events
kubectl get pods -n monitoring
kubectl describe pod <pod-name> -n monitoring

# Rollback if needed
helm rollback -n monitoring dashboard-cronjob
```

**Collectors Not Running:**

```bash
# Check CronJob is active
kubectl get cronjob -n monitoring dashboard-cronjob -o yaml | grep suspend
# Should show: suspend: false

# Force a manual run
kubectl create job --from=cronjob/dashboard-cronjob test-$(date +%s) -n monitoring

# Check logs
kubectl logs -n monitoring -l app.kubernetes.io/instance=dashboard-cronjob --tail=100
```

## Building the Docker Image

### Platform Compatibility

**Important**: The production environment runs on **Linux AMD64** (x86_64). If you're building on macOS ARM64 (Apple
Silicon), you must build for the correct platform:

```bash
# ✅ Correct - Build for linux/amd64 (production platform)
docker buildx build --platform linux/amd64 -f softmax/Dockerfile -t softmax-dashboard:local .

# ❌ Incorrect - Builds for your local platform (macOS ARM64)
docker build -f softmax/Dockerfile -t softmax-dashboard:local .
```

### Build Locally

```bash
# From repo root - build for linux/amd64 (production platform)
docker buildx build --platform linux/amd64 -f softmax/Dockerfile -t softmax-dashboard:local --load .

# Verify the image
docker images | grep softmax-dashboard

# Test locally (requires AWS credentials and secrets)
docker run --rm \
  -e AWS_PROFILE=softmax-admin \
  -v ~/.aws:/root/.aws:ro \
  softmax-dashboard:local \
  uv run python /workspace/metta/devops/datadog/run_collector.py github --verbose
```

### Build with Specific Tag

```bash
# Build with version tag (for linux/amd64)
docker buildx build --platform linux/amd64 -f softmax/Dockerfile -t softmax-dashboard:v1.0.0 --load .

# Build with git commit SHA (for linux/amd64)
GIT_SHA=$(git rev-parse --short HEAD)
docker buildx build --platform linux/amd64 -f softmax/Dockerfile -t softmax-dashboard:sha-${GIT_SHA} --load .
```

**Note**: The `--load` flag loads the image into your local Docker daemon. Without it, the image is only built but not
available locally.

## Pushing to ECR

### Login to ECR

```bash
# Get ECR login password and login
aws ecr get-login-password --region us-east-1 --profile softmax-admin | \
  docker login --username AWS --password-stdin 751442549699.dkr.ecr.us-east-1.amazonaws.com
```

### Tag and Push

```bash
# Set ECR registry
export ECR_REGISTRY=751442549699.dkr.ecr.us-east-1.amazonaws.com
export IMAGE_NAME=softmax-dashboard

# Tag the local image
docker tag softmax-dashboard:local ${ECR_REGISTRY}/${IMAGE_NAME}:latest

# For specific versions
GIT_SHA=$(git rev-parse --short HEAD)
docker tag softmax-dashboard:local ${ECR_REGISTRY}/${IMAGE_NAME}:sha-${GIT_SHA}

# Push to ECR
docker push ${ECR_REGISTRY}/${IMAGE_NAME}:latest
docker push ${ECR_REGISTRY}/${IMAGE_NAME}:sha-${GIT_SHA}
```

### Verify Push

```bash
# List images in ECR
aws ecr describe-images --repository-name softmax-dashboard --region us-east-1 | \
  jq -r '.imageDetails[] | "\(.imageTags[0]) - \(.imagePushedAt)"'
```

## Deploying with Helm

### Single Collector Deployment

The default Helm chart deploys a single CronJob running the GitHub collector.

```bash
# Deploy using the helper script
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring \
  --cluster main

# Deploy with specific image tag
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring \
  --cluster main \
  --set image.tag=sha-abc123

# Dry run (template only)
./devops/k8s/deploy-helm-chart.sh \
  --chart devops/charts/dashboard-cronjob \
  --release dashboard-cronjob \
  --namespace monitoring \
  --cluster main \
  --dry-run
```

### Multi-Collector Deployment

Deploy separate CronJobs for each collector:

```bash
# GitHub collector (every 15 minutes)
helm upgrade -n monitoring github-collector ./devops/charts/dashboard-cronjob \
  --set command[0]=uv \
  --set command[1]=run \
  --set command[2]=python \
  --set command[3]=/workspace/metta/devops/datadog/run_collector.py \
  --set command[4]=github \
  --set command[5]=--push \
  --set schedule="*/15 * * * *" \
  --set datadog.service=github-collector \
  --install

# Skypilot collector (every 10 minutes - faster for job monitoring)
helm upgrade -n monitoring skypilot-collector ./devops/charts/dashboard-cronjob \
  --set command[0]=uv \
  --set command[1]=run \
  --set command[2]=python \
  --set command[3]=/workspace/metta/devops/datadog/run_collector.py \
  --set command[4]=skypilot \
  --set command[5]=--push \
  --set schedule="*/10 * * * *" \
  --set datadog.service=skypilot-collector \
  --install

# Asana collector (every 30 minutes - less frequent updates needed)
helm upgrade -n monitoring asana-collector ./devops/charts/dashboard-cronjob \
  --set command[0]=uv \
  --set command[1]=run \
  --set command[2]=python \
  --set command[3]=/workspace/metta/devops/datadog/run_collector.py \
  --set command[4]=asana \
  --set command[5]=--push \
  --set schedule="*/30 * * * *" \
  --set datadog.service=asana-collector \
  --install
```

### Update Existing Deployment

```bash
# After building and pushing a new image, update the deployment
helm upgrade -n monitoring dashboard-cronjob ./devops/charts/dashboard-cronjob \
  --set image.tag=sha-${GIT_SHA}
```

## Monitoring Deployments

### Check CronJob Status

```bash
# List all cronjobs in monitoring namespace
kubectl get cronjobs -n monitoring

# Get detailed info about a cronjob
kubectl describe cronjob dashboard-cronjob -n monitoring

# Check cronjob configuration
kubectl get cronjob dashboard-cronjob -n monitoring -o yaml
```

### Monitor Job Execution

```bash
# List recent jobs (sorted by creation time)
kubectl get jobs -n monitoring --sort-by=.metadata.creationTimestamp

# Watch for new jobs
kubectl get jobs -n monitoring --watch

# Get jobs for specific cronjob
kubectl get jobs -n monitoring -l app.kubernetes.io/instance=dashboard-cronjob
```

### View Logs

```bash
# View logs from the most recent job
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=100

# Follow logs in real-time
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob -f

# View logs from a specific job
kubectl logs -n monitoring job/<job-name>

# View logs from a failed job
kubectl get jobs -n monitoring --field-selector status.successful=0
kubectl logs -n monitoring job/<failed-job-name>
```

## Troubleshooting

### Image Pull Errors

```bash
# Check if the image exists in ECR
aws ecr describe-images --repository-name softmax-dashboard --region us-east-1

# Verify the service account has correct IAM role annotation
kubectl get serviceaccount -n monitoring -o yaml | grep eks.amazonaws.com/role-arn

# Check pod events
kubectl get pods -n monitoring
kubectl describe pod <pod-name> -n monitoring
```

### Authentication Errors

The collectors authenticate to AWS Secrets Manager using the service account's IAM role:

```bash
# Verify the service account has the correct role
kubectl get sa -n monitoring dashboard-cronjob-sa -o yaml

# Expected annotation:
# eks.amazonaws.com/role-arn: arn:aws:iam::751442549699:role/dashboard-cronjob

# Test secret access from a pod
kubectl run -n monitoring test-secrets --rm -it --image=amazon/aws-cli --restart=Never \
  --serviceaccount=dashboard-cronjob-sa -- \
  secretsmanager get-secret-value --secret-id github/dashboard-token --region us-east-1
```

### Collector Failures

```bash
# Check recent job failures
kubectl get jobs -n monitoring --field-selector status.successful=0

# View logs from failed job
kubectl logs -n monitoring job/<failed-job-name>

# Test collector locally with verbose output
docker run --rm \
  -e AWS_PROFILE=softmax-admin \
  -v ~/.aws:/root/.aws:ro \
  softmax-dashboard:latest \
  uv run python /workspace/metta/devops/datadog/run_collector.py github --verbose

# Test from within the cluster
kubectl run -n monitoring test-collector --rm -it \
  --image=751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-dashboard:latest \
  --serviceaccount=dashboard-cronjob-sa \
  --restart=Never -- \
  uv run python /workspace/metta/devops/datadog/run_collector.py github --verbose
```

### Manual Job Trigger

```bash
# Trigger a job manually for testing
kubectl create job --from=cronjob/dashboard-cronjob manual-test-$(date +%s) -n monitoring

# Watch the job
kubectl get jobs -n monitoring --watch

# View logs
kubectl logs -n monitoring job/manual-test-<timestamp>
```

### Rollback Deployment

```bash
# List deployment history
helm history -n monitoring dashboard-cronjob

# Rollback to previous version
helm rollback -n monitoring dashboard-cronjob

# Rollback to specific revision
helm rollback -n monitoring dashboard-cronjob 3
```

## CI/CD Pipeline

The automated pipeline runs on every push to `main`:

1. **Build**: Builds Docker image with git SHA tag
2. **Push**: Pushes to ECR with both SHA and `latest` tags
3. **Deploy**: Runs `helm upgrade` to update the deployment

**Workflow file**: `.github/workflows/build-dashboard-image.yml`

### Trigger Manual Build

```bash
# Trigger workflow manually via GitHub CLI
gh workflow run "Build Dashboard CronJob Docker Image"

# Check workflow status
gh run list --workflow "Build Dashboard CronJob Docker Image"

# View logs
gh run view <run-id> --log
```

### Workflow Triggers

The workflow runs automatically when:

- Pushing to `main` branch
- Changes to `softmax/**`
- Changes to `common/**`
- Changes to `devops/datadog/**` (needs to be added)
- Changes to `devops/tools/install-system.sh`
- Changes to `.github/workflows/build-dashboard-image.yml`

## Adding New Dependencies

When adding a new collector or dependency:

1. **Update pyproject.toml**: Add dependencies to appropriate package
2. **Update Dockerfile**: Ensure dependencies are installed (via `uv sync`)
3. **Test locally**: Build and test Docker image
4. **Push to main**: CI/CD will rebuild and deploy automatically

Example: Adding the asana package

```toml
# In pyproject.toml or appropriate workspace package
[project.dependencies]
asana = "^5.0.0"
```

```bash
# Test locally (build for linux/amd64)
docker buildx build --platform linux/amd64 -f softmax/Dockerfile -t softmax-dashboard:test --load .
docker run --rm softmax-dashboard:test \
  uv run python -c "import asana; print('OK')"
```

## Quick Reference

### One-Line Deploy

```bash
# Build, push, and deploy in one go (for linux/amd64 platform)
docker buildx build --platform linux/amd64 -f softmax/Dockerfile -t softmax-dashboard:local --load . && \
  docker tag softmax-dashboard:local 751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-dashboard:latest && \
  docker push 751442549699.dkr.ecr.us-east-1.amazonaws.com/softmax-dashboard:latest && \
  helm upgrade -n monitoring dashboard-cronjob ./devops/charts/dashboard-cronjob
```

### Common Commands

```bash
# Check what's running
kubectl get cronjobs,jobs,pods -n monitoring

# View recent logs
kubectl logs -n monitoring -l app.kubernetes.io/name=dashboard-cronjob --tail=50

# Force a run
kubectl create job --from=cronjob/dashboard-cronjob test-$(date +%s) -n monitoring

# Clean up old jobs
kubectl delete jobs -n monitoring --field-selector status.successful=1
```

## Related Documentation

- **Collector Development**: `devops/datadog/collectors/README.md`
- **Deployment Helpers**: `devops/k8s/README.md`
- **Helm Chart**: `devops/charts/dashboard-cronjob/`
- **CI/CD Workflow**: `.github/workflows/build-dashboard-image.yml`
