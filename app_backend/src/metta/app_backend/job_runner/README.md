# Job Runner Local Dev

## Setup

```bash
# Build image + Kind cluster + create jobs namespace
metta observatory kind build

# Start postgres + backend + reconciler (uses same image)
metta observatory backend up -d
```

## Submit test jobs

```bash
uv run python app_backend/scripts/submit_test_jobs.py
```

## Monitor

```bash
# Watch k8s jobs
kubectl get jobs -n jobs -w

# Reconciler logs
metta observatory backend logs reconciler

# Backend logs
metta observatory backend logs observatory-backend
```

## Teardown

```bash
metta observatory backend down
metta observatory kind clean  # optional, deletes cluster
```
