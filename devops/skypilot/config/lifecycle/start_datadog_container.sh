#!/usr/bin/env bash

# Simple script to start Datadog agent as a Docker container

echo "[DATADOG] Starting Datadog Agent container..."

# Check if docker command is available
if ! command -v docker &> /dev/null; then
    echo "[DATADOG] Docker CLI not found in container - cannot start Datadog container"
    echo "[DATADOG] Consider adding docker-cli to the application container image"
    exit 0
fi

# Get DD_API_KEY from environment or AWS Secrets Manager
if [ -z "${DD_API_KEY:-}" ]; then
    if command -v aws &> /dev/null; then
        DD_API_KEY=$(aws secretsmanager get-secret-value \
            --secret-id datadog/api-key \
            --region us-east-1 \
            --query SecretString \
            --output text 2>/dev/null || true)
    fi
fi

if [ -z "$DD_API_KEY" ]; then
    echo "[DATADOG] Warning: DD_API_KEY not found - skipping Datadog container"
    exit 0
fi

# Check if Datadog container is already running
if docker ps | grep -q dd-agent; then
    echo "[DATADOG] Datadog container already running"
    exit 0
fi

# Start Datadog agent container
# Using the recommended flags for Amazon Linux v2 / Ubuntu
docker run -d \
    --cgroupns host \
    --pid host \
    --name dd-agent \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    -v /proc/:/host/proc/:ro \
    -v /sys/fs/cgroup/:/host/sys/fs/cgroup:ro \
    -e DD_API_KEY="$DD_API_KEY" \
    -e DD_SITE="${DD_SITE:-datadoghq.com}" \
    -e DD_LOGS_ENABLED=true \
    -e DD_LOGS_CONFIG_CONTAINER_COLLECT_ALL=true \
    -e DD_PROCESS_AGENT_ENABLED=true \
    -e DD_CONTAINER_EXCLUDE="name:dd-agent" \
    -e DD_TAGS="env:production service:skypilot-worker metta_run_id:${METTA_RUN_ID:-unknown} skypilot_task_id:${SKYPILOT_TASK_ID:-unknown}" \
    gcr.io/datadoghq/agent:7

# Wait for container to start
sleep 5

# Check if running
if docker ps | grep -q dd-agent; then
    echo "[DATADOG] ✅ Datadog container started successfully"
    echo "[DATADOG] Metrics will appear in Datadog within 1-2 minutes"
else
    echo "[DATADOG] ⚠️ Failed to start Datadog container"
    docker logs dd-agent 2>/dev/null | head -20
fi