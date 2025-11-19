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
  # Check if log collection config exists - if so, restart to pick it up
  if [ -f "/etc/datadog-agent/conf.d/custom_logs.d/conf.yaml" ]; then
    echo "[DATADOG] Log collection config found, restarting agent to pick up changes..."
    pkill -f "datadog-agent.*run" || true
    sleep 2
  else
    echo "[DATADOG] No log collection config found, agent running with default config"
    exit 0
  fi
fi

echo "[DATADOG] Starting Datadog agent daemon..."

# Start agent in background, redirect output to logs
nohup "$AGENT_BINARY" run > /tmp/datadog-agent.log 2>&1 &
AGENT_PID=$!

# Wait a moment and verify it started
sleep 2

if ps -p "$AGENT_PID" > /dev/null; then
  echo "[DATADOG] Agent started successfully (PID: $AGENT_PID)"

  # Try to verify status (non-blocking)
  if "$AGENT_BINARY" status > /dev/null 2>&1; then
    echo "[DATADOG] Agent status: Running"
  else
    echo "[DATADOG] Agent started but status check failed (may need more time)"
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
