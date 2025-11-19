#!/usr/bin/env bash
# Start Datadog agent as a background daemon in SkyPilot jobs
# This script is called during the run phase after environment setup

set -e # Exit on error, but allow commands that might fail

# Try multiple possible locations for the agent binary
AGENT_BINARY=""
for path in \
  "/opt/datadog-agent/bin/agent/agent" \
  "/opt/datadog-agent/embedded/bin/agent" \
  "/usr/bin/datadog-agent" \
  "$(which datadog-agent 2> /dev/null)"; do
  if [ -n "$path" ] && [ -f "$path" ]; then
    AGENT_BINARY="$path"
    break
  fi
done

if [ -z "$AGENT_BINARY" ]; then
  echo "[DATADOG] Agent binary not found in any standard location, attempting to find it..."
  # Try to find it anywhere
  FOUND_BINARY=$(find /opt /usr -name "agent" -type f -path "*/datadog-agent/*" 2> /dev/null | head -1)
  if [ -n "$FOUND_BINARY" ] && [ -f "$FOUND_BINARY" ]; then
    AGENT_BINARY="$FOUND_BINARY"
    echo "[DATADOG] Found agent binary at: $AGENT_BINARY"
  else
    echo "[DATADOG] Agent binary not found, skipping startup"
    echo "[DATADOG] This may indicate the agent was not properly installed"
    exit 0
  fi
fi

# Check if agent is already running
if pgrep -f "datadog-agent.*run" > /dev/null 2>&1; then
  echo "[DATADOG] Agent is already running"
  # Always restart to ensure it picks up latest config (logs_enabled, log collection config, etc.)
  # This is safe because we're in the run phase and configs are already set
  echo "[DATADOG] Restarting agent to ensure latest configuration is loaded..."
  pkill -f "datadog-agent.*run" || true
  sleep 2
fi

echo "[DATADOG] Starting Datadog agent daemon..."

# Start agent in background, redirect output to logs
nohup "$AGENT_BINARY" run > /tmp/datadog-agent.log 2>&1 &
AGENT_PID=$!

# Wait a moment and verify it started
sleep 2

if ps -p "$AGENT_PID" > /dev/null; then
  echo "[DATADOG] Agent started successfully (PID: $AGENT_PID)"

  # Wait a bit more for agent to fully initialize
  sleep 3

  # Try to verify status and log collection
  if "$AGENT_BINARY" status > /tmp/datadog-agent-status.log 2>&1; then
    echo "[DATADOG] Agent status: Running"
    # Check if log collection is enabled
    if grep -q "Logs Agent" /tmp/datadog-agent-status.log 2>/dev/null; then
      echo "[DATADOG] Log collection appears to be enabled"
    else
      echo "[DATADOG] WARNING: Could not verify log collection status"
    fi
  else
    echo "[DATADOG] Agent started but status check failed (may need more time)"
    echo "[DATADOG] Status output:"
    cat /tmp/datadog-agent-status.log 2>/dev/null || echo "  (no status output)"
  fi
  
  # Verify log collection config exists and show contents
  if [ -f "/etc/datadog-agent/conf.d/skypilot_training.d/conf.yaml" ]; then
    echo "[DATADOG] Log collection config found: /etc/datadog-agent/conf.d/skypilot_training.d/conf.yaml"
    echo "[DATADOG] Config file contents (first 20 lines):"
    head -20 "/etc/datadog-agent/conf.d/skypilot_training.d/conf.yaml" | sed 's/^/  /'
    
    # Check if agent can see the config
    if "$AGENT_BINARY" configcheck > /tmp/datadog-configcheck.log 2>&1; then
      if grep -q "skypilot_training" /tmp/datadog-configcheck.log 2>/dev/null; then
        echo "[DATADOG] ✓ Agent recognizes skypilot_training config"
      else
        echo "[DATADOG] ⚠️ Agent configcheck doesn't show skypilot_training config"
        echo "[DATADOG] Configcheck output:"
        cat /tmp/datadog-configcheck.log | head -30 | sed 's/^/  /'
      fi
    fi
  else
    echo "[DATADOG] WARNING: Log collection config not found!"
    echo "[DATADOG] Checking what config files exist:"
    ls -la /etc/datadog-agent/conf.d/*/conf.yaml 2>/dev/null | head -10 | sed 's/^/  /' || echo "  (no config files found)"
  fi
  
  # Check if log files exist and have content
  if [ -f "/tmp/training_logs/training_combined.log" ]; then
    LOG_SIZE=$(stat -f%z /tmp/training_logs/training_combined.log 2>/dev/null || stat -c%s /tmp/training_logs/training_combined.log 2>/dev/null || echo "0")
    echo "[DATADOG] Training log file exists: /tmp/training_logs/training_combined.log (size: ${LOG_SIZE} bytes)"
  else
    echo "[DATADOG] WARNING: Training log file not found: /tmp/training_logs/training_combined.log"
  fi
else
  echo "[DATADOG] WARNING: Agent process died immediately after startup"
  echo "[DATADOG] Check logs: /tmp/datadog-agent.log"
  if [ -f /tmp/datadog-agent.log ]; then
    echo "[DATADOG] Last 20 lines of agent log:"
    tail -20 /tmp/datadog-agent.log || true
  fi
  exit 1
fi
