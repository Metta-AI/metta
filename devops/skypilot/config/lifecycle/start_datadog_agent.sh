#!/usr/bin/env bash
# Start Datadog agent as a background daemon in SkyPilot jobs
# This script is called during the run phase after environment setup
# This script is non-fatal - failures won't break training

set +e  # Don't exit on error - Datadog is optional, training must continue

# Update log collection config with run-specific tags
# This must happen in the run phase when METTA_RUN_ID and SKYPILOT_TASK_ID are available
if [ -f "$(dirname "$0")/update_datadog_log_config.sh" ]; then
  bash "$(dirname "$0")/update_datadog_log_config.sh" || true
elif [ -f "./devops/skypilot/config/lifecycle/update_datadog_log_config.sh" ]; then
  bash ./devops/skypilot/config/lifecycle/update_datadog_log_config.sh || true
fi

# Find agent binary
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
  FOUND_BINARY=$(find /opt /usr -name "agent" -type f -path "*/datadog-agent/*" 2> /dev/null | head -1)
  if [ -n "$FOUND_BINARY" ] && [ -f "$FOUND_BINARY" ]; then
    AGENT_BINARY="$FOUND_BINARY"
  else
    exit 0  # Agent not found, skip silently
  fi
fi

# Restart agent if already running to pick up latest config
if pgrep -f "datadog-agent.*run" > /dev/null 2>&1; then
  pkill -f "datadog-agent.*run" || true
  sleep 2
fi

# Delay startup to prevent interference with training process initialization
# This gives Ray/Torchrun time to bind ports and start up fully
sleep 60

# CRITICAL: Set DD_LOGS_ENABLED=true - this environment variable takes precedence over config file
# This ensures the Logs Agent component starts even if the config file has logs_enabled: false
export DD_LOGS_ENABLED="true"

# Start agent with DD_LOGS_ENABLED=true to ensure Logs Agent component starts
DD_LOGS_ENABLED=true nohup "$AGENT_BINARY" run > /tmp/datadog-agent.log 2>&1 &
AGENT_PID=$!

sleep 2

if ! ps -p "$AGENT_PID" > /dev/null 2>&1; then
  exit 0  # Agent failed to start, skip silently
fi

# Ensure logs_enabled is set in config file (for next restart, though env var takes precedence)
MAIN_CONFIG="/etc/datadog-agent/datadog.yaml"
if [ -f "$MAIN_CONFIG" ] && ! grep -q "logs_enabled: true" "$MAIN_CONFIG" 2>/dev/null; then
  if grep -q "logs_enabled: false" "$MAIN_CONFIG" 2>/dev/null; then
    sed -i 's/logs_enabled: false/logs_enabled: true/' "$MAIN_CONFIG"
  elif ! grep -q "logs_enabled:" "$MAIN_CONFIG" 2>/dev/null; then
    echo "logs_enabled: true" >> "$MAIN_CONFIG"
  fi

  # If we updated the config, restart the agent to pick up the change
  # (though DD_LOGS_ENABLED env var already ensures logs are enabled)
  pkill -f "datadog-agent.*run" || true
  sleep 3
  DD_LOGS_ENABLED=true nohup "$AGENT_BINARY" run > /tmp/datadog-agent.log 2>&1 &
  NEW_AGENT_PID=$!
  sleep 2
  if ! ps -p "$NEW_AGENT_PID" > /dev/null 2>&1; then
    # Agent restart failed, but this is non-fatal - continue anyway
    # The initial agent start (line 53) may still be running
    true  # no-op, just for syntax
  fi
fi

exit 0
