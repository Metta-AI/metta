#!/usr/bin/env bash
# Start Datadog agent daemon
# Non-fatal script

set +e  # Don't exit on error

# Update log config
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

# Restart if running
if pgrep -f "datadog-agent.*run" > /dev/null 2>&1; then
  pkill -f "datadog-agent.*run" || true
  sleep 2
fi

# Delay for training init
sleep 60

# Set DD_LOGS_ENABLED=true
export DD_LOGS_ENABLED="true"

# Start agent
DD_LOGS_ENABLED=true nohup "$AGENT_BINARY" run > /tmp/datadog-agent.log 2>&1 &
AGENT_PID=$!

sleep 2

if ! ps -p "$AGENT_PID" > /dev/null 2>&1; then
  exit 0  # Agent failed
fi

# Set logs_enabled in config
MAIN_CONFIG="/etc/datadog-agent/datadog.yaml"
if [ -f "$MAIN_CONFIG" ] && ! grep -q "logs_enabled: true" "$MAIN_CONFIG" 2>/dev/null; then
  if grep -q "logs_enabled: false" "$MAIN_CONFIG" 2>/dev/null; then
    sed -i 's/logs_enabled: false/logs_enabled: true/' "$MAIN_CONFIG"
  elif ! grep -q "logs_enabled:" "$MAIN_CONFIG" 2>/dev/null; then
    echo "logs_enabled: true" >> "$MAIN_CONFIG"
  fi

  # Restart after config update
  pkill -f "datadog-agent.*run" || true
  sleep 3
  DD_LOGS_ENABLED=true nohup "$AGENT_BINARY" run > /tmp/datadog-agent.log 2>&1 &
  NEW_AGENT_PID=$!
  sleep 2
  if ! ps -p "$NEW_AGENT_PID" > /dev/null 2>&1; then
    true  # Non-fatal restart failure
  fi
fi

exit 0
