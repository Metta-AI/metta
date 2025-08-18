#!/usr/bin/env bash

set -e
cd /workspace/metta

echo "[SETUP] Fetching latest from origin..."
git fetch origin "$METTA_GIT_REF" || git fetch --depth=1000 origin
git checkout "$METTA_GIT_REF"
echo "[SETUP] Checked out: $(git rev-parse HEAD)"

# Note that different sets of skypilot environment variables are available in "run" vs "setup"
# see https://docs.skypilot.co/en/latest/running-jobs/environment-variables.html

# Install Datadog Agent if DD_API_KEY is provided
if [ -n "${DD_API_KEY:-}" ]; then
    echo "[SETUP] Installing Datadog Agent..."

    # Install Datadog Agent using the official script
    DD_AGENT_MAJOR_VERSION=7 DD_API_KEY="$DD_API_KEY" DD_SITE="${DD_SITE:-datadoghq.com}" bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"

    # Configure Datadog to collect Docker metrics
    if [ -f /etc/datadog-agent/datadog.yaml ]; then
        echo "[SETUP] Configuring Datadog for Docker monitoring..."

        # Enable Docker integration
        sudo sed -i 's/# process_config:/process_config:/' /etc/datadog-agent/datadog.yaml
        sudo sed -i '/process_config:/a\  enabled: true' /etc/datadog-agent/datadog.yaml

        # Enable log collection
        sudo sed -i 's/# logs_enabled: false/logs_enabled: true/' /etc/datadog-agent/datadog.yaml

        # Add Docker check configuration
        sudo tee /etc/datadog-agent/conf.d/docker.d/conf.yaml > /dev/null <<EOF
init_config:

instances:
  - collect_container_size: true
    collect_container_count: true
    collect_images_stats: true
    collect_image_size: true
    collect_disk_stats: true
    collect_exit_codes: true
    tags:
      - "env:${DD_ENV:-production}"
      - "service:${DD_SERVICE:-skypilot-worker}"
EOF

        # Restart Datadog Agent to apply configuration
        sudo systemctl restart datadog-agent

        # Check agent status
        sleep 5
        sudo datadog-agent status || echo "[WARN] Datadog agent status check failed, continuing anyway"

        echo "[SETUP] Datadog Agent installed and configured successfully"
    else
        echo "[WARN] Datadog configuration file not found, agent may not be properly configured"
    fi
else
    echo "[SETUP] Skipping Datadog Agent installation (DD_API_KEY not provided)"
fi
