#!/usr/bin/env bash

# Use error handling but don't exit on systemctl failures
set -uo pipefail

echo "[DATADOG] Starting Datadog Agent installation on host..."

# Get Datadog API key from AWS Secrets Manager or environment
get_dd_api_key() {
    if [ -n "${DD_API_KEY:-}" ]; then
        echo "$DD_API_KEY"
        return 0
    fi

    # Try to get from AWS Secrets Manager
    if command -v aws &> /dev/null; then
        aws secretsmanager get-secret-value \
            --secret-id datadog/api-key \
            --region us-east-1 \
            --query SecretString \
            --output text 2>/dev/null || true
    fi
}

DD_API_KEY=$(get_dd_api_key)

if [ -z "$DD_API_KEY" ]; then
    echo "[DATADOG] Warning: DD_API_KEY not found in environment or AWS Secrets Manager"
    echo "[DATADOG] Skipping Datadog agent installation"
    exit 0
fi

# Check if Datadog agent is already installed and running
if command -v datadog-agent >/dev/null 2>&1; then
    echo "[DATADOG] Datadog agent is already installed"
    # Check if it's running
    if pgrep -f "datadog-agent" >/dev/null 2>&1; then
        echo "[DATADOG] Agent is already running"
        exit 0
    fi
fi

echo "[DATADOG] Installing Datadog Agent with Docker integration..."

# Set installation environment variables
export DD_API_KEY="$DD_API_KEY"
export DD_SITE="${DD_SITE:-datadoghq.com}"
export DD_INSTALL_ONLY="true"  # Don't try to start in setup phase - will start in run phase

# Install Datadog Agent
DD_AGENT_MAJOR_VERSION=7 bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script_agent7.sh)"

# Configure Datadog Agent for Docker monitoring
echo "[DATADOG] Configuring Docker integration..."

# Create Docker integration configuration
sudo tee /etc/datadog-agent/conf.d/docker.d/docker_daemon.yaml > /dev/null << 'EOF'
init_config:

instances:
  - url: "unix://var/run/docker.sock"
    # Collect events from Docker
    collect_events: true
    # Collect container size
    collect_container_size: true
    # Collect container labels as tags
    collect_labels_as_tags:
      - "com.docker.compose.project"
      - "com.docker.compose.service"
    # Collect metrics for all containers
    include:
      - ".*"
EOF

# Update main Datadog configuration (only if not already configured)
if ! grep -q "container_collection_enabled" /etc/datadog-agent/datadog.yaml 2>/dev/null; then
    echo "[DATADOG] Adding container monitoring configuration..."
    sudo tee -a /etc/datadog-agent/datadog.yaml > /dev/null << EOF

# Enable Docker check
process_config:
  enabled: true

# Enable container monitoring
container_collection_enabled: true

# Collect Docker metrics
docker_root: /var/lib/docker

# Log collection from containers
logs_enabled: true
logs_config:
  container_collect_all: true
  docker_container_use_file: false

# APM configuration for containers
apm_config:
  enabled: true
  apm_non_local_traffic: true

# Add tags from SkyPilot environment
tags:
  - env:production
  - service:skypilot-worker
  - metta_run_id:${METTA_RUN_ID:-unknown}
  - skypilot_task_id:${SKYPILOT_TASK_ID:-unknown}
  - skypilot_node_rank:${SKYPILOT_NODE_RANK:-0}
  - skypilot_num_nodes:${SKYPILOT_NUM_NODES:-1}
  - git_ref:${METTA_GIT_REF:-unknown}
EOF
else
    echo "[DATADOG] Container monitoring already configured in datadog.yaml"
fi

# Add datadog-agent user to docker group for socket access
sudo usermod -a -G docker dd-agent 2>/dev/null || true

# Agent is installed but not started (DD_INSTALL_ONLY=true)
echo "[DATADOG] ✅ Datadog Agent installed successfully"
echo "[DATADOG] ✅ Docker integration configured"
echo "[DATADOG] ℹ️  Agent will start automatically in the run phase"
echo "[DATADOG] Metrics will appear in Datadog 1-2 minutes after job starts"
